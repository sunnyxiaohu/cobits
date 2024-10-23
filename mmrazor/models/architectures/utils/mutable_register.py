# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence, Tuple

from mmrazor.models.architectures.ops.mobilenet_series import MBBlock
from ...mutables.base_mutable import BaseMutable


def mutate_conv_module(
        conv_module,
        mutable_in_channels: Optional[BaseMutable] = None,
        mutable_out_channels: Optional[BaseMutable] = None,
        mutable_kernel_size: Optional[Tuple[BaseMutable,
                                            Sequence[int]]] = None):
    """Mutate a conv module."""
    if mutable_in_channels is not None:
        conv_module.conv.register_mutable_attr(
            'in_channels', mutable_in_channels)

    if mutable_out_channels is not None:
        conv_module.conv.register_mutable_attr(
            'out_channels', mutable_out_channels)

        if hasattr(conv_module, 'bn'):
            conv_module.bn.register_mutable_attr(
                'num_features', mutable_out_channels)

    if mutable_kernel_size is not None:
        conv_module.conv.register_mutable_attr('kernel_size',
                                               mutable_kernel_size)


def mutate_mobilenet_layer(mb_layer: MBBlock,
                           mutable_in_channels,
                           mutable_out_channels,
                           mutable_expand_ratio,
                           mutable_kernel_size,
                           fine_grained_mode: bool = False):
    """Mutate MobileNet layers."""
    derived_expand_channels = mutable_expand_ratio * mutable_in_channels

    if mb_layer.with_expand_conv:
        mutate_conv_module(
            mb_layer.expand_conv,
            mutable_in_channels=mutable_in_channels,
            mutable_out_channels=derived_expand_channels)

    mutate_conv_module(
        mb_layer.depthwise_conv,
        mutable_in_channels=derived_expand_channels,
        mutable_out_channels=derived_expand_channels,
        mutable_kernel_size=mutable_kernel_size)

    if mb_layer.with_se:
        if fine_grained_mode:
            mutable_expand_ratio2 = copy.deepcopy(mutable_expand_ratio)
            mutable_expand_ratio2.alias += '_se'
            derived_se_channels = mutable_expand_ratio2 * mutable_in_channels
            derived_se_channels = \
                derived_se_channels.derive_divide_mutable(4, 8)
        else:
            derived_se_channels = \
                derived_expand_channels.derive_divide_mutable(4, 8)

        mutate_conv_module(
            mb_layer.se.conv1,
            mutable_in_channels=derived_expand_channels,
            mutable_out_channels=derived_se_channels)
        mutate_conv_module(
            mb_layer.se.conv2,
            mutable_in_channels=derived_se_channels,
            mutable_out_channels=derived_expand_channels)

    if not mb_layer.with_res_shortcut:
        if mb_layer.with_attentive_shortcut:
            mb_layer.shortcut.conv.register_mutable_attr(
                'in_channels', mutable_in_channels)
            mb_layer.shortcut.conv.register_mutable_attr(
                'out_channels', mutable_out_channels)

    mutate_conv_module(
        mb_layer.linear_conv,
        mutable_in_channels=derived_expand_channels,
        mutable_out_channels=mutable_out_channels)
