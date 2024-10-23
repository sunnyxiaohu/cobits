# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any
import logging

import torch
import torch.distributed as dist
from mmengine.logging import print_log
from mmengine.runner import autocast
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from mmrazor.models.utils import get_module_device


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg: Tensor = 0
        self.sum: Tensor = 0
        self.count: int = 0

    def update(self, val: Any, batch_size: int = 1) -> None:
        if dist.is_initialized() and dist.is_available():
            dist.all_reduce(val, dist.ReduceOp.SUM, async_op=False)
            batch_size_tensor = torch.tensor([batch_size], device=val.device)
            dist.all_reduce(
                batch_size_tensor, dist.ReduceOp.SUM, async_op=False)
            total_batch_size = batch_size_tensor.item()

            val /= (total_batch_size / batch_size)
            batch_size = total_batch_size

        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


class CalibrateBNMixin:
    fp16: bool = False

    @torch.no_grad()
    def calibrate_bn_statistics(self,
                                dataloader: DataLoader,
                                calibrate_sample_num: int = 2000,
                                model = None) -> None:
        if model is None:
            model = self.runner.model

        def record_input_statistics_hook(bn_module: _BatchNorm, input: Tensor,
                                         output: Tensor) -> None:
            mean_average_meter: AverageMeter = bn_module.__mean_average_meter__
            var_average_meter: AverageMeter = bn_module.__var_average_meter__

            real_input = input[0]
            x_dim = real_input.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[1] = 0
            new_axis_list[0] = 1
            real_input = real_input.permute(new_axis_list).flatten(start_dim=1)
            mean = real_input.mean((1,))
            var = real_input.var((1,), unbiased=True)

            mean_average_meter.update(mean, x_dim[0])
            var_average_meter.update(var, x_dim[0])

        hook_handles = []

        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                print_log('register `record_input_statistics_hook` to module: '
                          f'{name}', logger='current', level=logging.DEBUG)
                module.__mean_average_meter__ = AverageMeter()
                module.__var_average_meter__ = AverageMeter()
                handle = module.register_forward_hook(
                    record_input_statistics_hook)
                hook_handles.append(handle)

        model.train()
        print_log('Start calibrating batch norm statistics', logger='current')
        print_log(f'Total sample number for calibration: {calibrate_sample_num}',
                  logger='current')
        remaining = calibrate_sample_num
        for data_batch in dataloader:
            # TODO: Handle remaining can not be divided evenly by total data_batch
            # if len(data_batch) >= remaining:
            #     data_batch = data_batch[:remaining]
            if isinstance(data_batch, torch.Tensor):
                data_batch_nums = len(data_batch)
            else:
                data_batch_nums = len(data_batch['inputs'])
            if dist.is_initialized() and dist.is_available():
                data_batch_tensor = torch.tensor(
                    [data_batch_nums], device=get_module_device(model))
                dist.all_reduce(
                    data_batch_tensor, dist.ReduceOp.SUM, async_op=False)
                data_batch_nums = data_batch_tensor.item()
            remaining -= data_batch_nums

            print_log(f'Remaining samples for calibration: {remaining}',
                      logger='current', level=logging.DEBUG)
            with autocast(enabled=self.fp16):
                model.test_step(data_batch)

            if remaining <= 0:
                break

        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                mean_average_meter = module.__mean_average_meter__
                var_average_meter = module.__var_average_meter__
                if mean_average_meter.count == 0 or \
                        var_average_meter.count == 0:
                    assert mean_average_meter.count == 0 and \
                        var_average_meter.count == 0
                    print_log(f'layer {name} is not chosen, ignored',
                              logger='current', level=logging.DEBUG)
                    continue

                calibrated_bn_mean = mean_average_meter.avg
                calibrated_bn_var = var_average_meter.avg

                feature_dim = calibrated_bn_mean.size(0)

                print_log(
                    f'layer: {name}, '
                    f'current feature dimension: {feature_dim}, '
                    'number of samples for calibration: '
                    f'{mean_average_meter.count}, '
                    'l2 norm of calibrated running mean: '
                    f'{calibrated_bn_mean.norm()}, '
                    'l2 norm of calibrated running var: '
                    f'{calibrated_bn_var.norm()}, '
                    'l2 norm of original running mean: '
                    f'{module.running_mean[:feature_dim].norm()}, '
                    'l2 norm of original running var: '
                    f'{module.running_var[:feature_dim].norm()}, ',
                    logger='current', level=logging.DEBUG)

                module.running_mean[:feature_dim].copy_(calibrated_bn_mean)
                module.running_var[:feature_dim].copy_(calibrated_bn_var)

                del module.__mean_average_meter__
                del module.__var_average_meter__

        print_log('Remove all hooks for calibration',
                  logger='current', level=logging.DEBUG)
        print_log('Calibrate batch norm statistics done', logger='current')
        for handle in hook_handles:
            handle.remove()
