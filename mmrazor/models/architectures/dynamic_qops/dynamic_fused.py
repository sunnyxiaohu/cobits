from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, Linear

from torch.nn.utils.parametrize import type_before_parametrizations

try:
    import torch.ao.nn.intrinsic as nni
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    nni = get_package_placeholder('torch>=1.13')

from ..dynamic_ops import BigNasConv2d, DynamicLinear, DynamicBatchNorm2d, DynamicBatchNorm1d


class DynamicConvReLU2d(nni._FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert type_before_parametrizations(conv) == BigNasConv2d and type_before_parametrizations(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type_before_parametrizations(conv), type_before_parametrizations(relu))
        super().__init__(conv, relu)

class DynamicLinearReLU(nni._FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, relu):
        assert type_before_parametrizations(linear) == DynamicLinear and type_before_parametrizations(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type_before_parametrizations(linear), type_before_parametrizations(relu))
        super().__init__(linear, relu)

class DynamicConvBn2d(nni._FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type_before_parametrizations(conv) == BigNasConv2d and type_before_parametrizations(bn) == DynamicBatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type_before_parametrizations(conv), type_before_parametrizations(bn))
        super().__init__(conv, bn)

class DynamicConvBnReLU2d(nni._FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type_before_parametrizations(conv) == BigNasConv2d and type_before_parametrizations(bn) == DynamicBatchNorm2d and \
            type_before_parametrizations(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type_before_parametrizations(conv), type_before_parametrizations(bn), type_before_parametrizations(relu))
        super().__init__(conv, bn, relu)

class DynamicLinearBn1d(nni._FusedModule):
    r"""This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, bn):
        assert type_before_parametrizations(linear) == DynamicLinear and type_before_parametrizations(bn) == DynamicBatchNorm1d, \
            'Incorrect types for input modules{}{}'.format(type_before_parametrizations(linear), type_before_parametrizations(bn))
        super().__init__(linear, bn)
