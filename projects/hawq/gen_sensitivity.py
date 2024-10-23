# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmrazor.utils import register_all_modules

import numpy as np
from typing import List, Tuple
import torch
from torch.nn import Module
from hessian_per_layer import hessian_per_layer, hessian_per_layer_acti
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicLinearMixin)

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMRazor test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def mixprecision_profiling(model: Module, quantized_model: Module, data: Tuple, algo='naive'):
    """
    Get layer sensitive index under a list of bitwidth.
    A lot of algorithms can do the same thing.
    HAWQ is the most useful one.
    Naive is the most straight forward one.
    """
    layer_parameters_dict = model_size_analysis(model)
    w_sensetive_dict = {}
    # layer_act_dict = act_size_analysis(model)
    a_sensetive_dict = {}
    if algo == 'hawq_eigen':
        eigen_values_dict = hawq(model, data, type='eigenvalues')
        # Do normalize.
        for layer, eigen_val in eigen_values_dict.items():
            eigen_values_dict[layer] = eigen_val / layer_parameters_dict[layer]
        for name, max_eignevalues in eigen_values_dict.items():
            print("Layer {} with max eigen values: {}".format(name, max_eignevalues))
        delta_w = get_delta_w(quantized_model)
        for layer, max_eignevalues in eigen_values_dict.items():
            # max_eigne_val: Float
            # delta_w: List shape = bitwidth_list
            w_sensetive_dict[layer] = max_eignevalues * delta_w[layer]
    elif algo == 'hawq_trace':
        trace_value_dict = hawq(model, data, type='trace')
        # Do normalize.
        for layer, trace in trace_value_dict.items():
            trace_value_dict[layer] = trace / layer_parameters_dict[layer]
        for name, trace in trace_value_dict.items():
            print("Layer {} with trace: {}".format(name, trace))
        delta_w = get_delta_w(quantized_model)
        for layer, trace in trace_value_dict.items():
            # max_eigne_val: Float
            # delta_w: List shape = bitwidth_list
            w_sensetive_dict[layer] = trace * delta_w[layer]
    elif algo == 'naive':
        w_sensetive_dict = prec_degradation_by_layer(model, quantized_model, data)
    else:
        print("Unknown algorithm!")
    return w_sensetive_dict


def get_delta_w(quantized_model: Module):

    def square_mean(ta, tb):
        return torch.pow((ta - tb), 2.0).mean().detach().cpu().numpy()

    delta_w = {}
    for name, mod in quantized_model.named_modules():
        if hasattr(mod, 'weight_fake_quant'):
            delta_w[name] = []
            mod.weight_fake_quant.fake_quant_enabled[0] = 1
            bitwidth_list = mod.weight_fake_quant.mutable_attrs['quant_bits'].choices
            for bits in bitwidth_list:
                mod.weight_fake_quant.mutable_attrs['quant_bits'].current_choice = bits
                mod.weight_fake_quant.get_dynamic_params()
                delta_w[name].append(square_mean(mod.weight, mod.weight_fake_quant(mod.weight)))
            delta_w[name] = np.array(delta_w[name])
            mod.weight_fake_quant.fake_quant_enabled[0] = 0

    return delta_w


def model_size_analysis(model):
    layer_parameters_dict = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (DynamicConvMixin, DynamicLinearMixin)):
            layer_parameters_dict[name] = mod.weight.numel()
    return layer_parameters_dict


def model_latency_analysis(model):
    pass


def model_flops_analyze(model):
    pass


def mp_model_size(model: Module):
    """
    Calcualte model size in different bitwidth.
    """
    mp_size = 0
    for mod in model.modules():
        if hasattr(mod, 'weight_fake_quant'):
            bitwidth = mod.weight_fake_quant.bitwidth
            mp_size += mod.weight.numel() * bitwidth
        elif hasattr(mod, 'weight'):
            mp_size += mod.weight.numel() * 32
    return mp_size / 8 / 1024 / 1024


def prec_degradation_by_layer(model: Module, quantized_model: Module, data: Tuple):
    """
    Calculate degradation of each layer in different bitwidth.
    """
    data = model.data_preprocessor(data, True)
    w_sensetive_dict = {}
    losses = model(**data, mode='loss')
    fp_loss = losses['loss']

    import pdb; pdb.set_trace()       
    
    for name, mod in quantized_model.named_modules():
        if hasattr(mod, 'weight_fake_quant'):
            w_sensetive_dict[name] = []
            mod.weight_fake_quant.fake_quant_enabled[0] = 1
            bitwidth_list = mod.weight_fake_quant.mutable_attrs['quant_bits'].choices
            for bits in bitwidth_list:
                mod.weight_fake_quant.mutable_attrs['quant_bits'].current_choice = bits
                mod.weight_fake_quant.get_dynamic_params()
                with torch.no_grad():
                    loss = quantized_model(**data)['loss']

                w_sensetive_dict[name].append(loss - fp_loss)
                print("Layer {} under bit {} with sensetive {}".format(name, bits, loss - fp_loss))
            mod.weight_fake_quant.fake_quant_enabled[0] = 0

    return w_sensetive_dict


def hawq(model: Module, data: Tuple, type='trace'):
    """
    HAWQ layer sensetive indicator. Using extend PyHessian to calculate.
    """
    # inputs, targets = data
    hessian_comp = hessian_per_layer(model, data=data)
    hessian_comp_act= hessian_per_layer_acti(model, data=data)
    if type == 'eigenvalues':
        return hessian_comp.layer_eigenvalues(), hessian_comp_act.layer_eigenvalues()
    elif type == 'trace':
        return hessian_comp.layer_trace(), hessian_comp_act.layer_trace()
    else:
        raise NotImplementedError("{} is not supported, only trace and eigenvalues.".format(type))


def main():
    register_all_modules(False)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.checkpoint == 'none':
        # NOTE: In this case, `args.checkpoint` isn't specified. If you haven't
        # specified a checkpoint in the `init_cfg` of the model yet, it may
        # cause the invalid results.
        cfg.load_from = None
    else:
        cfg.load_from = args.checkpoint
        if 'type' in cfg.test_cfg and cfg.test_cfg.type.endswith('PTQLoop'):
            cfg.test_cfg.only_val = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    import pdb; pdb.set_trace()
    # start testing
    # runner.test()
    model = runner.model.module.architecture.architecture
    quantized_model = runner.model.module.architecture.qmodels.loss
    data = next(iter(runner.train_dataloader))

    trace_w_sensetive_dict = mixprecision_profiling(model, quantized_model,
                                                    data=data, algo='hawq_trace')

if __name__ == '__main__':
    main()
