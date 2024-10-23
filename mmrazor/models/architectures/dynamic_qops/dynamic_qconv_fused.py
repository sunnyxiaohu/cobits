from typing import Any, Callable, Iterable, Optional, Tuple, Union

import copy
from contextlib import contextmanager

import math
import torch
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from typing import Callable

try:
    import torch.ao.nn.intrinsic as nni
    import torch.nn.intrinsic.qat as nniqat
    from torch.nn.intrinsic.qat.modules.conv_fused import _BN_CLASS_MAP
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    nni = get_package_placeholder('torch>=1.13')
    nniqat = get_package_placeholder('torch>=1.13')
    _BN_CLASS_MAP = {}

from mmrazor.models import BaseMutable

from ..dynamic_ops import (DynamicConvMixin, DynamicMixin, BigNasConv2d,
                           DynamicBatchNorm1d, DynamicBatchNorm2d,
                           DynamicBatchNorm3d)
from .dynamic_fused import DynamicConvReLU2d, DynamicConvBn2d, DynamicConvBnReLU2d



@contextmanager
def substitute_bn_class_map():
    org_bn_class_map = copy.deepcopy(_BN_CLASS_MAP)
    _BN_CLASS_MAP[1] = DynamicBatchNorm1d
    _BN_CLASS_MAP[2] = DynamicBatchNorm2d
    _BN_CLASS_MAP[3] = DynamicBatchNorm3d
    yield
    _BN_CLASS_MAP.clear()
    _BN_CLASS_MAP.update(org_bn_class_map)


def traverse_children(module: nn.Module) -> None:
    for name, mutable in module.items():
        if isinstance(mutable, DynamicMixin):
            module[name] = mutable.to_static_op()
        if hasattr(mutable, '_modules'):
            traverse_children(mutable._modules)


class DynamicQConvBn2d(nniqat.ConvBn2d, DynamicConvMixin):

    _FLOAT_MODULE = DynamicConvBn2d
    _FLOAT_CONV_MODULE = BigNasConv2d
    _FLOAT_BN_MODULE = DynamicBatchNorm2d
    _FLOAT_RELU_MODULE = None

    def __init__(self, *args, **kwarg):
        with substitute_bn_class_map():
            super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()
        self._enable_slow_path_for_better_numerical_stability = False # True

    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        # import pdb; pdb.set_trace()
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight * scale_factor.reshape(weight_shape)
        scaled_weight = self.weight_fake_quant(scaled_weight)
        scaled_weight, bias, padding, out_mask = self.get_dynamic_params(scaled_weight, self.bias)
        scale_factor = scale_factor[out_mask]
        # using zero bias here since the bias for original conv
        # will be added later
        if bias is not None:
            zero_bias = torch.zeros_like(bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(scaled_weight.size(0), device=scaled_weight.device, dtype=input.dtype)
        conv = self.conv_func(input, scaled_weight, zero_bias,
                              self.stride, padding, self.dilation, groups)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if bias is not None:
            conv_orig = conv_orig + bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def _forward_slow(self, input):
        """
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        assert self.bn.running_var is not None
        assert self.bn.running_mean is not None

        # using zero bias here since the bias for original conv
        # will be added later
        zero_bias = torch.zeros(self.out_channels, device=self.weight.device, dtype=input.dtype)

        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        weight, bias, padding, out_mask = self.get_dynamic_params(self.weight_fake_quant(self.weight), self.bias)
        zero_bias = zero_bias[out_mask]

        if self.bn.training:
            # import pdb; pdb.set_trace()
            # needed to compute batch mean/std
            conv_out = self.conv_func(input, weight, zero_bias,
                                      self.stride, padding, self.dilation, groups)
            # update bn statistics
            with torch.no_grad():
                conv_out_bias = (
                    conv_out if bias is None else conv_out + bias.reshape(bias_shape)
                )
                self.bn(conv_out_bias)

        # fused conv + bn without bias using bn running statistics
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scale_factor = scale_factor[out_mask]
        scaled_weight = weight * scale_factor.reshape(weight_shape)
        # fused conv without bias for inference: (r * W / running_std) * X
        conv_bn = self.conv_func(input, scaled_weight, zero_bias,
                                 self.stride, padding, self.dilation, groups)

        if self.bn.training:
            avg_dims = [0] + list(range(2, len(self.weight.shape)))
            batch_mean = conv_out.mean(avg_dims)
            batch_var = torch.square(conv_out - batch_mean.reshape(bias_shape)).mean(
                avg_dims
            )
            batch_std = torch.sqrt(batch_var + self.bn.eps)

            # scale to use batch std in training mode
            # conv(X, r * W / std_Y) = conv(X, r * W / running_std) * (running_std / std_Y)
            unscale_factor = running_std[out_mask] / batch_std
            conv_bn *= unscale_factor.reshape(bias_shape)

            fused_mean = batch_mean
            fused_std = batch_std
        else:
            fused_mean = self.bn.running_mean[out_mask] - (bias if bias is not None else 0)
            fused_std = running_std[out_mask]

        # fused bias = beta - r * mean / std
        fused_bias = self.bn.bias[out_mask] - self.bn.weight[out_mask] * fused_mean / fused_std
        conv_bn += fused_bias.reshape(bias_shape)

        # HACK to let conv bias particpiate in loss to avoid DDP error (parameters
        #   were not used in producing loss)
        if bias is not None:
            conv_bn += (bias - bias).reshape(bias_shape)

        return conv_bn

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        qat_conv = super(DynamicQConvBn2d, cls).from_float(mod)

        for attr, value in mod[0].mutable_attrs.items():
            qat_conv.register_mutable_attr(attr, value)

        for attr, value in mod[1].mutable_attrs.items():
            qat_conv.bn.register_mutable_attr(attr, value)

        return qat_conv

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        return nniqat.ConvBn2d

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

    def to_static_op(self):
        weight, bias, padding, out_mask = self.get_dynamic_params(self.weight, self.bias)
        groups = self.groups
        if groups == self.in_channels == self.out_channels and \
                self.mutable_in_channels is not None:
            mutable_in_channels = self.mutable_attrs['in_channels']
            groups = mutable_in_channels.current_mask.sum().item()
        out_channels = weight.size(0)
        in_channels = weight.size(1) * groups
        kernel_size = tuple(weight.shape[2:])

        cls = self.static_op_factory
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode)
        conv.weight = torch.nn.Parameter(weight)
        if bias is not None:
            conv.bias = torch.nn.Parameter(bias)

        running_mean, running_var, weight, bias = self.bn.get_dynamic_params()
        bn = cls._FLOAT_BN_MODULE(
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

        modules = [conv, bn]
        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)

        fake_quant = self.weight_fake_quant.to_static_op()
        if len(fake_quant.scale) > 1 and len(fake_quant.scale) != out_channels:
          fake_quant.scale.data = fake_quant.scale.data[out_mask]
          fake_quant.zero_point.data = fake_quant.zero_point.data[out_mask]

        mod = cls._FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
        mod.qconfig = self.qconfig
        mod.train(self.training)
        mod = cls.from_float(mod)
        mod.weight_fake_quant = fake_quant

        return mod

    def get_dynamic_params(
            self: _ConvNd, orig_weight, orig_bias) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tuple[int]]: Sliced weight, bias
                and padding.
        """
        # slice in/out channel of weight according to
        # mutable in_channels/out_channels
        weight, bias = self._get_dynamic_params_by_mutable_channels(
            orig_weight, orig_bias)

        if 'out_channels' in self.mutable_attrs:
            mutable_out_channels = self.mutable_attrs['out_channels']
            out_mask = mutable_out_channels.current_mask.to(orig_weight.device)
        else:
            out_mask = torch.ones(orig_weight.size(0)).bool().to(orig_weight.device)

        return weight, bias, self.padding, out_mask


class DynamicQConvBnReLU2d(DynamicQConvBn2d):

    # base class defines _FLOAT_MODULE as "ConvBn2d"
    _FLOAT_MODULE = DynamicConvBnReLU2d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = BigNasConv2d
    _FLOAT_BN_MODULE = DynamicBatchNorm2d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = DynamicConvReLU2d

    def forward(self, input):
        return F.relu(DynamicQConvBn2d._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(DynamicQConvBnReLU2d, cls).from_float(mod)

    @property
    def static_op_factory(self):
        return nniqat.ConvBnReLU2d


class DynamicQConvReLU2d(nniqat.ConvReLU2d, DynamicConvMixin):

    _FLOAT_MODULE = DynamicConvReLU2d
    _FLOAT_CONV_MODULE = BigNasConv2d
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        # import pdb; pdb.set_trace()
        qat_conv = super(DynamicQConvReLU2d, cls).from_float(mod)

        for attr, value in mod[0].mutable_attrs.items():
            qat_conv.register_mutable_attr(attr, value)

        return qat_conv

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        return nniqat.ConvReLU2d

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

    def forward(self, input):
        import pdb; pdb.set_trace()
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias, padding, out_mask = self.get_dynamic_params(
            self.weight_fake_quant(self.weight), self.bias)
        return F.relu(
            self.conv_func(input, weight, bias,
                           self.stride, padding, self.dilation, groups))

    def to_static_op(self):
        weight, bias, padding, out_mask = self.get_dynamic_params(self.weight, self.bias)
        groups = self.groups
        if groups == self.in_channels == self.out_channels and \
                self.mutable_in_channels is not None:
            mutable_in_channels = self.mutable_attrs['in_channels']
            groups = mutable_in_channels.current_mask.sum().item()
        out_channels = weight.size(0)
        in_channels = weight.size(1) * groups
        kernel_size = tuple(weight.shape[2:])

        cls = self.static_op_factory
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode)
        conv.weight = torch.nn.Parameter(weight)
        if bias is not None:
            conv.bias = torch.nn.Parameter(bias)

        modules = [conv, cls._FLOAT_RELU_MODULE()]

        fake_quant = self.weight_fake_quant.to_static_op()
        if len(fake_quant.scale) > 1 and len(fake_quant.scale) != out_channels:
          fake_quant.scale.data = fake_quant.scale.data[out_mask]
          fake_quant.zero_point.data = fake_quant.zero_point.data[out_mask]

        mod = cls._FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
        mod.qconfig = self.qconfig
        mod.train(self.training)
        mod = cls.from_float(mod)
        mod.weight_fake_quant = fake_quant

        return mod

    def get_dynamic_params(
            self: _ConvNd, orig_weight, orig_bias) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tuple[int]]: Sliced weight, bias
                and padding.
        """
        # slice in/out channel of weight according to
        # mutable in_channels/out_channels
        weight, bias = self._get_dynamic_params_by_mutable_channels(
            orig_weight, orig_bias)

        if 'out_channels' in self.mutable_attrs:
            mutable_out_channels = self.mutable_attrs['out_channels']
            out_mask = mutable_out_channels.current_mask.to(orig_weight.device)
        else:
            out_mask = torch.ones(orig_weight.size(0)).bool().to(orig_weight.device)

        return weight, bias, self.padding, out_mask

