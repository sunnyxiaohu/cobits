# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import TASK_UTILS
from ..flops_params_counter import get_model_parameters_number
from .base_counter import BaseCounter


@TASK_UTILS.register_module()
class DynamicQConv2dCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the dynamic channels of conv
        layers."""
        if hasattr(module, '_DEPTH_SCOPE') and module._DEPTH_MUTABLE.current_choice < module._DEPTH_SCOPE:
            return

        input = input[0]

        batch_size = input.shape[0]
        output_dims = list(output.shape[2:])

        kernel_dims = list(module.kernel_size)

        if 'out_channels' in module.mutable_attrs:
            out_channels = module.mutable_attrs['out_channels'].current_choice
        else:
            out_channels = module.out_channels
        if 'in_channels' in module.mutable_attrs:
            in_channels = module.mutable_attrs['in_channels'].current_choice
        else:
            in_channels = module.in_channels

        groups = module.groups

        filters_per_channel = out_channels / groups
        conv_per_position_flops = \
            np.prod(kernel_dims) * in_channels * filters_per_channel

        active_elements_count = batch_size * int(np.prod(output_dims))

        overall_conv_flops = conv_per_position_flops * active_elements_count
        overall_params = conv_per_position_flops

        bias_flops = 0
        overall_params = conv_per_position_flops
        if module.bias is not None:
            bias_flops = out_channels * active_elements_count
            overall_params += out_channels

        overall_flops = overall_conv_flops + bias_flops

        w_quant_bits = module.weight_fake_quant.mutable_attrs['quant_bits'].current_choice
        act_quant_bits = module._ACT_QUANT_BITS.current_choice
        overall_flops = overall_flops * w_quant_bits * act_quant_bits # / 64
        assert w_quant_bits <= 8 and act_quant_bits <= 8
        module.__flops__ += overall_flops
        module.__params__ += int(overall_params * w_quant_bits)

@TASK_UTILS.register_module()
class DynamicQConvBnReLU2dCounter(DynamicQConv2dCounter):
    pass

@TASK_UTILS.register_module()
class DynamicQConvBn2dCounter(DynamicQConv2dCounter):
    pass

@TASK_UTILS.register_module()
class DynamicQConvReLU2dCounter(DynamicQConv2dCounter):
    pass

@TASK_UTILS.register_module()
class DynamicQLinearCounter(BaseCounter):
    """FLOPs/params counter for Linear operation series."""

    @staticmethod
    def add_count_hook(module, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        if hasattr(module, '_DEPTH_SCOPE') and module._DEPTH_MUTABLE.current_choice < module._DEPTH_SCOPE:
            return
        
        input = input[0]
        output_last_dim = output.shape[
            -1]  # pytorch checks dimensions, so here we don't care much
        w_quant_bits = module.weight_fake_quant.mutable_attrs['quant_bits'].current_choice
        # import pdb; pdb.set_trace()
        act_quant_bits = module._ACT_QUANT_BITS.current_choice
        overall_flops = np.prod(input.shape) * output_last_dim * w_quant_bits * act_quant_bits # / 64
        assert w_quant_bits <= 8 and act_quant_bits <= 8
        module.__flops__ += overall_flops
        module.__params__ += int(get_model_parameters_number(module) * w_quant_bits)

@TASK_UTILS.register_module()
class DynamicQLinearBn1dCounter(DynamicQLinearCounter):
    pass

@TASK_UTILS.register_module()
class DynamicQLinearReLUCounter(DynamicQLinearCounter):
    pass
