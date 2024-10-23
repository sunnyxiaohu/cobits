from .dynamic_qconv_fused import DynamicQConvBnReLU2d, DynamicQConvBn2d, DynamicQConvReLU2d
from .dynamic_qlinear import DynamicQLinearBn1d
import torch.nn.intrinsic.qat as nniqat

bn_modules = [DynamicQConvBnReLU2d, DynamicQConvBn2d, DynamicQConvReLU2d, DynamicQLinearBn1d]


def update_bn_stats(mod):
    if type(mod) in bn_modules:
        mod.update_bn_stats()
    else:
        nniqat.update_bn_stats(mod)

def freeze_bn_stats(mod):
    if type(mod) in bn_modules:
        mod.freeze_bn_stats()
    else:
        nniqat.freeze_bn_stats(mod)
