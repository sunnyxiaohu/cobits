from typing import Any, Dict, Optional, Set, Tuple

import copy
import torch
from torch import Tensor, nn

import torch.nn.functional as F
from torch.nn.utils.parametrize import (
    is_parametrized,
    type_before_parametrizations,
    transfer_parametrizations_and_params,
)

try:
    import torch.nn.intrinsic as nni
    import torch.nn.intrinsic.qat as nniqat
    import torch.nn.qat as nnqat
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    nni = get_package_placeholder('torch>=1.13')
    nniqat = get_package_placeholder('torch>=1.13')
    nnqat = get_package_placeholder('torch>=1.13')

from mmrazor.models import BaseMutable
from ..dynamic_ops import DynamicLinear, DynamicMixin, DynamicLinearMixin, DynamicBatchNorm1d
from .dynamic_fused import DynamicLinearBn1d, DynamicLinearReLU


class DynamicQLinear(nnqat.Linear, DynamicLinearMixin):

    _FLOAT_MODULE = DynamicLinear
    _FLOAT_LINEAR_MODULE = DynamicLinear

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        weight, bias, out_mask = self.get_dynamic_params(self.weight_fake_quant(self.weight), self.bias)
        return F.linear(input, weight, bias)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type_before_parametrizations(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if type_before_parametrizations(mod) == DynamicLinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, qconfig=qconfig)

        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        for attr, value in mod.mutable_attrs.items():
            qat_linear.register_mutable_attr(attr, value)

        return qat_linear

    @property
    def static_op_factory(self):
        return nnqat.Linear

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

    def get_dynamic_params(self: nn.Linear, orig_weight=None, orig_bias=None) -> Tuple[Tensor, Optional[Tensor]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Sliced weight and bias.
        """
        if orig_weight is None and orig_bias is None:
            orig_weight, orig_bias = self.weight, self.bias

        if 'in_features' in self.mutable_attrs:
            in_mask = self.mutable_attrs['in_features'].current_mask.to(
                orig_weight.device)
        else:
            in_mask = torch.ones(orig_weight.size(1)).bool().to(
                orig_weight.device)
        if 'out_features' in self.mutable_attrs:

            out_mask = self.mutable_attrs['out_features'].current_mask.to(
                orig_weight.device)
        else:
            out_mask = torch.ones(orig_weight.size(0)).bool().to(
                orig_weight.device)

        weight = orig_weight[out_mask][:, in_mask]
        bias = orig_bias[out_mask] if orig_bias is not None else None

        return weight, bias, out_mask

    def to_static_op(self):
        weight, bias, out_mask = self.get_dynamic_params(self.weight, self.bias)
        out_channels = weight.size(0)
        in_channels = weight.size(1)

        linear = nn.Linear(  # type: ignore[attr-defined]
            in_channels,
            out_channels,
            bias=self.bias is not None)
        linear.weight = nn.Parameter(weight)
        if bias is not None:
            linear.bias = nn.Parameter(bias)

        fake_quant = self.weight_fake_quant.to_static_op()
        if len(fake_quant.scale) > 1 and len(fake_quant.scale) != out_channels:
          fake_quant.scale.data = fake_quant.scale.data[out_mask]
          fake_quant.zero_point.data = fake_quant.zero_point.data[out_mask]

        mod = linear
        cls = self.static_op_factory
        if self._FLOAT_MODULE == DynamicLinearReLU:
            linear.qconfig = self.qconfig
            modules = [linear, nn.ReLU()]
            mod = cls._FLOAT_MODULE(*modules)
        mod.qconfig = self.qconfig
        mod.train(self.training)
        mod = cls.from_float(mod)
        mod.weight_fake_quant = fake_quant
        return mod


class DynamicQLinearBn1d(nniqat.LinearBn1d, DynamicLinearMixin):
    r"""
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = DynamicLinearBn1d

    def __init__(self,
                 # Linear args
                 in_features, out_features, bias=True,
                 # BatchNorm1d args
                 # num_features: out_features
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        assert qconfig, 'qconfig must be provded for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = DynamicBatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        assert self.bn.running_var is not None

        # Scale the linear weights by BN's running statistics to reduce
        # weight jitter, see https://arxiv.org/pdf/1806.08342.pdf, page 18
        # for motivation.
        #
        # Instead of
        #
        #   x1 = F.linear(x0, fq(w), b)
        #   x2 = self.bn(x1)
        #
        # We have
        #
        #   # scale the weight by previous batch's running statistics
        #   scale_factor = bn.w / bn.running_std_from_prev_batch
        #   # do the linear transformation without bias
        #   x1_scaled = F.linear(x0, fq(w * scale_factor), 0)
        #   # reverse the scaling and add original bias
        #   x1_orig = x1_scaled / scale_factor + b
        #   x2 = self.bn(x1_orig)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        scaled_weight, bias, out_mask = self.get_dynamic_params(scaled_weight, self.bias)
        scale_factor = scale_factor[out_mask]
        if bias is not None:
            zero_bias = torch.zeros_like(bias)
        else:
            zero_bias = torch.zeros(scaled_weight.size(0), device=scaled_weight.device)
        linear_out = F.linear(input, scaled_weight, zero_bias)
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)
        if bias is not None:
            linear_out_orig = linear_out_orig + bias.reshape(bias_shape)
        bn_out = self.bn(linear_out_orig)
        return bn_out

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod' a float module, either produced by torch.ao.quantization
            utilities or directly from user
        """
        assert type(mod) == DynamicLinearBn1d, 'qat.' + cls.__name__ + \
            '.from_float only works for ' + DynamicLinearBn1d.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid config'
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(linear.in_features, linear.out_features, linear.bias is not None,
                           bn.eps, bn.momentum,
                           False, qconfig)
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked

        for attr, value in linear.mutable_attrs.items():
            qat_linearbn.register_mutable_attr(attr, value)
        for attr, value in bn.mutable_attrs.items():
            qat_linearbn.bn.register_mutable_attr(attr, value)

        return qat_linearbn

    @property
    def static_op_factory(self):
        return nniqat.LinearBn1d

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

    def get_dynamic_params(self: nn.Linear, orig_weight=None, orig_bias=None) -> Tuple[Tensor, Optional[Tensor]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Sliced weight and bias.
        """
        if orig_weight is None and orig_bias is None:
            orig_weight, orig_bias = self.weight, self.bias

        if 'in_features' in self.mutable_attrs:
            in_mask = self.mutable_attrs['in_features'].current_mask.to(
                orig_weight.device)
        else:
            in_mask = torch.ones(orig_weight.size(1)).bool().to(
                orig_weight.device)
        if 'out_features' in self.mutable_attrs:

            out_mask = self.mutable_attrs['out_features'].current_mask.to(
                orig_weight.device)
        else:
            out_mask = torch.ones(orig_weight.size(0)).bool().to(
                orig_weight.device)

        weight = orig_weight[out_mask][:, in_mask]
        bias = orig_bias[out_mask] if orig_bias is not None else None

        return weight, bias, out_mask

    def to_static_op(self):
        weight, bias, out_mask = self.get_dynamic_params(self.weight, self.bias)
        out_channels = weight.size(0)
        in_channels = weight.size(1)

        linear = nn.Linear(  # type: ignore[attr-defined]
            in_channels,
            out_channels,
            bias=self.bias is not None)
        linear.weight = nn.Parameter(weight)
        if bias is not None:
            linear.bias = nn.Parameter(bias)

        running_mean, running_var, weight, bias = self.bn.get_dynamic_params()
        bn = nn.BatchNorm1d(
            num_features=out_channels,
            eps=self.bn.eps,
            momentum=self.bn.momentum,
            affine=self.bn.affine,
            track_running_stats=self.bn.track_running_stats)
        if running_mean is not None:
            bn.running_mean.copy_(running_mean)
            bn.running_mean = bn.running_mean.to(running_mean.device)
        if running_var is not None:
            bn.running_var.copy_(running_var)
            bn.running_var = bn.running_var.to(running_var.device)
        if weight is not None:
            bn.weight = nn.Parameter(weight)
        if bias is not None:
            bn.bias = nn.Parameter(bias)

        fake_quant = self.weight_fake_quant.to_static_op()
        if len(fake_quant.scale) > 1 and len(fake_quant.scale) != out_channels:
          fake_quant.scale.data = fake_quant.scale.data[out_mask]
          fake_quant.zero_point.data = fake_quant.zero_point.data[out_mask]

        modules = [linear, bn]
        cls = self.static_op_factory
        mod = nni.LinearBn1d(*modules)
        mod.qconfig = self.qconfig
        mod.train(self.training)
        mod = cls.from_float(mod)
        mod.weight_fake_quant = fake_quant
        return mod


class DynamicQLinearReLU(DynamicQLinear):

    _FLOAT_MODULE = DynamicLinearReLU

    def forward(self, input):
        return F.relu(super().forward(input))

    @property
    def static_op_factory(self):
        return nniqat.LinearReLU
