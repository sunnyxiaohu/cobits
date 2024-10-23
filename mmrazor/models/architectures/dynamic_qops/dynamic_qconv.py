from typing import Any, Callable, Iterable, Optional, Tuple, Union

import copy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd

try:
    import torch.nn.qat as nnqat
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    nnqat = get_package_placeholder('torch>=1.13')

from mmrazor.models import BaseMutable
from ..dynamic_ops import BigNasConv2d, DynamicMixin, DynamicConvMixin

def traverse_children(module: nn.Module) -> None:
    for name, mutable in module.items():
        if isinstance(mutable, DynamicMixin):
            module[name] = mutable.to_static_op()
        if hasattr(mutable, '_modules'):
            traverse_children(mutable._modules)


class DynamicQConv2d(nnqat.Conv2d, DynamicConvMixin):

    _FLOAT_MODULE = BigNasConv2d
    _FLOAT_CONV_MODULE = BigNasConv2d

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias, padding, out_mask = self.get_dynamic_params(self.weight_fake_quant(self.weight), self.bias)
        return self.conv_func(input, weight, bias,
                              self.stride, padding, self.dilation, groups)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        qat_conv = super(DynamicQConv2d, cls).from_float(mod)

        for attr, value in mod.mutable_attrs.items():
            qat_conv.register_mutable_attr(attr, value)

        return qat_conv

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        return nnqat.Conv2d

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

    def get_dynamic_params(
            self: _ConvNd, orig_weight=None, orig_bias=None) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tuple[int]]: Sliced weight, bias
                and padding.
        """
        if orig_weight is None and orig_bias is None:
            orig_weight, orig_bias = self.weight, self.bias
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
        mod = cls._FLOAT_MODULE(  # type: ignore[attr-defined]
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode)
        mod.weight = torch.nn.Parameter(weight)
        if bias is not None:
            mod.bias = torch.nn.Parameter(bias)

        fake_quant = self.weight_fake_quant.to_static_op()
        if len(fake_quant.scale) > 1 and len(fake_quant.scale) != out_channels:
          fake_quant.scale.data = fake_quant.scale.data[out_mask]
          fake_quant.zero_point.data = fake_quant.zero_point.data[out_mask]

        mod.qconfig = self.qconfig
        mod.train(self.training)
        mod = cls.from_float(mod)
        mod.weight_fake_quant = fake_quant
        return mod
