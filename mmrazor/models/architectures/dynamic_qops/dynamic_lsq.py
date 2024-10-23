# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
try:
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver
except ImportError:
    from mmrazor.utils import get_placeholder
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')

from mmrazor.registry import MODELS
from mmrazor.models import LearnableFakeQuantize

from ..dynamic_ops import DynamicMixin
from ...task_modules.tracer.fx.graph_utils import update_qdype_qmin_qmax


def fix_calib_stats(mod):
    """Fix learning of quantization parameters, if applicable. Example
    usage::

    # model is any PyTorch model model.apply(fix_calib_stats)
    """
    if isinstance(mod, DynamicBatchLearnableFakeQuantize):
        mod.calib_stats_fixed[0] = 1

def unfix_calib_stats(mod):
    """Unfix learning of quantization parameters, if applicable. Example
    usage::

    # model is any PyTorch model model.apply(fix_calib_stats)
    """
    if isinstance(mod, DynamicBatchLearnableFakeQuantize):
        mod.calib_stats_fixed[0] = 0


@MODELS.register_module()
class DynamicLearnableFakeQuantize(LearnableFakeQuantize, DynamicMixin):
    """This is an extension of the FakeQuantize module in fake_quantize.py,
    which supports learning of the scale and zero point parameters through
    backpropagation.

    In addition to the attributes in the original FakeQuantize module, the
    DynamicLearnableFakeQuantize module also includes the following attributes to
    support quantization parameter learning.
    """
    accepted_mutable_attrs = {'quant_bits'}
    FLOAT_BITS = 32
    accepted_param_share_modes = {0, 1, 2, 3, 4}

    def __init__(self, *args, param_share_mode = 1, M=0.4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # mode 0: Unshared
        # mode 1: Full shared with all the bits sharing the same scale
        # mode 2: Reparameter and partially shared
        # mode 3: Full shared with `1.0 / 2 ** (quant_bits - self.BASE_BITS)`
        # mode 4: mode 2 with adelta constrained
        # mode 5: mode 4 with channel-wise theta learnable
        assert param_share_mode in self.accepted_param_share_modes, f'Unexpected param_share_mode: {param_share_mode}'
        self.param_share_mode = param_share_mode
        if self.param_share_mode in [2, 4]:
            self.scale_theta = Parameter(torch.tensor([0.]))
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()
        self.M = M
        self.BASE_BITS = self.bitwidth

    @property
    def static_op_factory(self):
        return LearnableFakeQuantize

    def convert_from(cls, module):
        """Convert a nn.Linear module to a DynamicLinear.

        Args:
            module (:obj:`torch.nn.Linear`): The original Linear module.
        """
        raise NotImplementedError()

    def to_static_op(self) -> nn.Module:
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return

        scale, zero_point = self.calculate_qparams()
        if len(scale) == 1:
            observer = MinMaxObserver
        else:
            observer = PerChannelMinMaxObserver
        lsq = LearnableFakeQuantize(observer.with_args())
        lsq.scale.data = torch.ones_like(scale)
        lsq.zero_point.data = torch.zeros_like(zero_point.float())
        lsq.scale.data.copy_(scale.data)
        lsq.zero_point.data.copy_(zero_point.data)
        lsq.fake_quant_enabled.data.copy_(self.fake_quant_enabled.data)
        lsq.static_enabled.data.copy_(self.static_enabled.data)
        lsq.learning_enabled.copy_(self.learning_enabled.data)
        lsq.eps.copy_(self.eps.data)
        update_qdype_qmin_qmax(lsq, self.bitwidth, quant_min=self.quant_min, quant_max=self.quant_max, qdtype=self.dtype)
        return lsq

    @torch.jit.export
    def calculate_qparams(self):
        quant_bits, index = self.get_dynamic_params()

        mult = self.mult_factor(quant_bits)
        # Check whether is initialized or not. We choose scale as indicator
        # since zero_point may also be zero after initialized.
        scale = self.scale[:, index] if self.param_share_mode == 0 else self.scale
        zero_point = self.zero_point[:, index]
        scale_initialized = not torch.equal(scale, torch.ones_like(scale))
        # scale_initialized = scale_initialized or not self.training
        if not scale_initialized or self.static_enabled[0] == 1:
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            # self.activation_post_process.reset_min_max_vals()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            if self.qscheme in (torch.per_channel_symmetric,
                                torch.per_channel_affine) and scale.shape != _scale.shape:
                if self.param_share_mode == 0:
                    self.scale.data = self.scale.data.repeat(len(_scale), 1)
                else:
                    self.scale.data = torch.ones_like(_scale)
                    if self.param_share_mode in [2, 4]:
                        self.scale_theta.data = self.scale_theta.data.repeat(len(_scale), 1)
                self.zero_point.data = self.zero_point.data.repeat(len(_zero_point), 1)
                scale = self.scale[:, index] if self.param_share_mode == 0 else self.scale
                zero_point = self.zero_point[:, index]

            _scale = _scale / mult
            scale.data.copy_(_scale)
            zero_point.data.copy_(_zero_point)

        if index is None:
            zero_point = self.zero_point

        if self.param_share_mode == 0:
            scale = self.scale[:, index]
        else:
            scale = self.scale
            scale = mult * scale
            if self.param_share_mode in [2, 4]:
                if self.param_share_mode == 4:
                    clip_m2 = mult * scale * ((1 + self.M) ** (quant_bits - self.BASE_BITS) - 1)
                    clip_m1 = torch.zeros_like(clip_m2)
                    self.scale_theta.data[:, index] = torch.clamp(self.scale_theta.data[:, index], clip_m1, clip_m2)
                scale = scale + self.scale_theta[:, index]
        scale.data.abs_()
        scale.data.clamp_(min=self.eps.item())
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return scale, zero_point.float()

    def register_mutable_attr(self, attr, mutable):
        assert hasattr(self, 'mutable_attrs')
        # assert not self.zero_point_trainable, 'Unsupport zero_point_trainable now.'
        if attr == 'quant_bits':
            device = self.scale.device
            if self.param_share_mode == 0:
                self.scale.data = torch.ones(1, len(mutable.choices)).to(device)
            elif self.param_share_mode in [2, 4]:
                self.scale_theta.data = torch.zeros(1, len(mutable.choices)).to(device)
            self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)
            self.mutable_attrs['quant_bits'] = mutable
        else:
            raise NotImplementedError
        self.BASE_BITS = self.mutable_attrs['quant_bits'].choices[0]

    def get_dynamic_params(self):
        if 'quant_bits' in self.mutable_attrs:
            quant_bits = self.mutable_attrs['quant_bits'].current_choice
            update_qdype_qmin_qmax(self, quant_bits)
            index = self.mutable_attrs['quant_bits'].choices.index(quant_bits)
        else:
            # TODO: handle
            if abs(self.quant_min) == abs(self.quant_max):
                quant_bits = int(math.log(self.quant_max - self.quant_min + 2, 2))
            else:
                quant_bits = int(math.log(self.quant_max - self.quant_min + 1, 2))
            index = None
            # assert self.param_share_mode == 1, f'param_share_mode can only be "1", when index is None, get {self.param_share_mode}'
        return quant_bits, index

    @torch.jit.export
    def observe_quant_params(self):
        """Calculate the quantization parameters."""
        raise NotImplementedError()

    def mult_factor(self, quant_bits):
        factor = 1.0
        if self.param_share_mode in [2, 3, 4]:
            factor = 1.0 / 2 ** (quant_bits - self.BASE_BITS)
        return factor

    def forward(self, X):
        """Forward computation. """
        if X.numel() == 0:
            return X
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return X

        self.activation_post_process(X.detach())
        # import pdb; pdb.set_trace()
        if self.fake_quant_enabled[0] == 1:
            scale, zero_point = self.calculate_qparams()

            if self.qscheme in (torch.per_channel_symmetric,
                                torch.per_channel_affine):
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max)**0.5
                else:
                    grad_factor = 1.0
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, scale, zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max)**0.5
                else:
                    grad_factor = 1.0
                if not (self.quant_min <= zero_point <= self.quant_max):
                    print(self.quant_min, zero_point, self.quant_max)
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, self.quant_min,
                    self.quant_max, grad_factor)

        return X

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Removing this function throws an error that the the size of the
        loaded tensor does not match the original size i.e., These buffers
        start out with numel 0 and become numel 1 once they have their first
        forward pass.

        Modified from https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fake_quantize.py  # noqa:E501
        """
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                local_val = getattr(self, name)
                # import pdb; pdb.set_trace()
                if local_val.shape == val.shape:
                    continue

                if name == 'scale':
                    self.scale.data.repeat(val.size(0), 1)
                else:
                    assert name == 'zero_point'
                    self.zero_point.data.repeat(val.size(0), 1)
                state_dict[key] = val.repeat(1, local_val.size(1))
                # For torchscript module we need to update the attributes here
                # since we do not call the `_load_from_state_dict` function
                # defined module.py
                # TODO: handle torch.jit.is_scripting
                # if torch.jit.is_scripting():
                #     if name == 'scale':
                #         self.scale.copy_(val)
                #     else:
                #         assert name == 'zero_point'
                #         self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super(LearnableFakeQuantize,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)


@MODELS.register_module()
class DynamicBatchLearnableFakeQuantize(DynamicLearnableFakeQuantize):
    """This is implementation of BatchQuantizer.
    Note that the differences between LSQ and BatchLSQ:
        1. during training, instead of learning the scale and zero_point,
        we use extreme value estimators to caputre the range variation in the
        current batch, and learn a multiplicative residual scale_mult and
        an additive residual zero_point.
        2. during test, we have to recalibrate the minmax statics by
        enabling the observer updatable and forwarding a few batches fo data.
    """
    accepted_param_share_modes = {0, 1, 2, 3, 4, 5}

    def __init__(self, *args, M=0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer('calib_stats_fixed',
                             torch.tensor([0], dtype=torch.uint8))
        if self.param_share_mode == 5:
            self.scale_theta = Parameter(torch.tensor([0.]))
        self.M = M #0.2 #0.4 #0.3

    @torch.jit.export
    def calculate_qparams(self):
        quant_bits, index = self.get_dynamic_params()
        _scale, _zero_point = \
            self.activation_post_process.calculate_qparams()

        if index is None:
            zp_add = self.zero_point
            scale_mult = self.scale
        elif self.param_share_mode == 0:
            zp_add = self.zero_point[:, index]
            scale_mult = self.scale[:, index]
        elif self.param_share_mode == 1:
            zp_add = self.zero_point
            scale_mult = self.scale
        elif self.param_share_mode in [2, 4, 5]:
            zp_add = self.zero_point[:, index]
            scale_mult = self.scale

            scale_initialized = not torch.equal(scale_mult, torch.ones_like(scale_mult))
            # scale_initialized = scale_initialized or not self.training
            if not scale_initialized and self.param_share_mode == 5:
                if self.qscheme in (torch.per_channel_symmetric,
                                    torch.per_channel_affine) and scale_mult.shape != _scale.shape:
                    self.scale.data = self.scale.data.repeat(len(_scale))
                    self.scale_theta.data = self.scale_theta.data.repeat(len(_scale), 1)
                    self.zero_point.data = self.zero_point.data.repeat(len(_zero_point), 1)
                    zp_add = self.zero_point[:, index]
                    scale_mult = self.scale

            if self.param_share_mode in [4, 5]:
                m = (quant_bits - self.BASE_BITS)
                clip_m2 = ((1 + self.M) ** m - 1) * scale_mult
                clip_m1 = torch.zeros_like(clip_m2)
                self.scale_theta.data[:, index] = torch.clamp(self.scale_theta.data[:, index], clip_m1, clip_m2)
            scale_mult = scale_mult + self.scale_theta[:, index]
        scale = _scale * scale_mult
        zero_point = _zero_point + zp_add
        scale.data.abs_()
        scale.data.clamp_(min=self.eps.item())
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return scale, zero_point

    def register_mutable_attr(self, attr, mutable):
        assert hasattr(self, 'mutable_attrs')
        if attr == 'quant_bits':
            device = self.scale.device
            if self.param_share_mode == 0:
                self.scale.data = torch.ones(1, len(mutable.choices)).to(device)
                self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)
            elif self.param_share_mode in [2, 4, 5]:
                self.scale_theta.data = torch.zeros(1, len(mutable.choices)).to(device)
                self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)
            self.mutable_attrs['quant_bits'] = mutable
        else:
            raise NotImplementedError
        self.BASE_BITS = self.mutable_attrs['quant_bits'].choices[0]

    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        """
        if X.numel() == 0:
            return X
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return X
        # import pdb; pdb.set_trace()
        if self.calib_stats_fixed[0] == 0:
            self.activation_post_process(X.detach())

        if self.fake_quant_enabled[0] == 1:
            scale, zero_point = self.calculate_qparams()

            if self.qscheme in (torch.per_channel_symmetric,
                                torch.per_channel_affine):
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max)**0.5
                else:
                    grad_factor = 1.0
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, scale, zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max)**0.5
                else:
                    grad_factor = 1.0
                if not (self.quant_min <= zero_point <= self.quant_max):
                    print(self.quant_min, zero_point, self.quant_max)
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, self.quant_min,
                    self.quant_max, grad_factor)

        return X

    def to_static_op(self) -> nn.Module:
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return

        scale, zero_point = self.calculate_qparams()
        if len(scale) == 1:
            observer = MinMaxObserver
        else:
            observer = PerChannelMinMaxObserver
        lsq = LearnableFakeQuantize(observer.with_args())
        lsq.scale.data = torch.ones_like(scale)
        lsq.zero_point.data = torch.zeros_like(zero_point.float())
        lsq.scale.data.copy_(scale.data)
        lsq.zero_point.data.copy_(zero_point.data)
        lsq.fake_quant_enabled.data.copy_(self.fake_quant_enabled.data)
        lsq.static_enabled.data.copy_(self.static_enabled.data)
        lsq.learning_enabled.copy_(self.learning_enabled.data)
        lsq.eps.copy_(self.eps.data)
        update_qdype_qmin_qmax(lsq, self.bitwidth, quant_min=self.quant_min, quant_max=self.quant_max, qdtype=self.dtype)
        return lsq
