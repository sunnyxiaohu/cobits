# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import logging
import os
import os.path as osp
import math
import random
import time
import warnings
from functools import partial
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pulp import (LpProblem, lpSum, LpVariable, LpInteger, LpMinimize, value,
                  LpMaximize, PULP_CBC_CMD)

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from mmengine import fileio
from mmengine.dist import broadcast_object_list, all_reduce_params, is_distributed
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop, autocast
from mmengine.utils import is_list_of, import_modules_from_strings

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer, disable_fake_quant,
                                       FakeQuantizeBase)
except ImportError:
    from mmrazor.utils import get_placeholder

    disable_observer = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    FakeQuantizeBase = get_placeholder('torch>=1.13')

from mmrazor.models import (BaseAlgorithm, register_torch_fake_quants,
                            register_torch_observers, LearnableFakeQuantize,
                            BatchLSQObserver, OneShotMutableChannel, DerivedMutable)
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicLinearMixin)
from mmrazor.models.architectures.dynamic_qops import (freeze_bn_stats,
    fix_calib_stats, unfix_calib_stats, DynamicLearnableFakeQuantize)                                                      
from mmrazor.models.fake_quants import (enable_param_learning,
                                        enable_static_estimate, enable_val)
from mmrazor.models.task_modules.tracer.fx.graph_utils import _get_attrs
from mmrazor.models.utils import add_prefix, get_module_device
from mmrazor.structures import Candidates, export_fix_subnet, convert_fix_subnet
from mmrazor.structures.subnet.fix_subnet import _load_fix_subnet_by_mutable

from mmrazor.registry import LOOPS, TASK_UTILS
from mmrazor.utils import SupportRandomSubnet
from mmrazor.engine.runner import QATEpochBasedLoop, EvolutionSearchLoop
from mmrazor.engine.runner.utils import check_subnet_resources, crossover
from mmrazor.engine.runner.utils.calibrate_bn_mixin import AverageMeter

TORCH_observers = register_torch_observers()
TORCH_fake_quants = register_torch_fake_quants()


class FWDSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self):
        self.store_input = None

    def __call__(self, module, input_batch, output_batch):
        output_batch.requires_grad_()
        output_batch.retain_grad()
        self.store_input = output_batch

class BWDSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self):
        self.store_input = None

    def __call__(self, module, input_batch, output_batch):
        self.store_input = output_batch


@contextlib.contextmanager
def adabn_context(model):
    running_means = []
    running_vars = []
    num_batches_trackeds = []
    for mod in model.modules():
        if isinstance(mod, _BatchNorm) and mod.track_running_stats:
            running_means.append(mod.running_mean.data.clone())
            running_vars.append(mod.running_var.data.clone())
            num_batches_trackeds.append(mod.num_batches_tracked.clone())
            # mod.reset_running_stats()
    yield
    i = 0
    for mod in model.modules():
        if isinstance(mod, _BatchNorm) and mod.track_running_stats:
            mod.running_mean.data.copy_(running_means[i])
            mod.running_var.data.copy_(running_vars[i])
            mod.num_batches_tracked.data.copy_(num_batches_trackeds[i])
            i += 1

class CalibrateMixin:
    fp16: bool = False

    @torch.no_grad()
    def calibrate_bn_observer_statistics(self,
                                dataloader: DataLoader,
                                calibrate_sample_num: int = 2000,
                                model = None) -> None:
        if model is None:
            model = self.runner.model
        model.apply(unfix_calib_stats)

        def record_bn_statistics_hook(bn_module: _BatchNorm, input: Tensor,
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

            mean_average_meter.update(mean, real_input.size(0))
            var_average_meter.update(var, real_input.size(0))

        def record_observer_statistics_hook(observer_module, input: Tensor,
                                            output: Tensor) -> None:
            max_average_meter: AverageMeter = observer_module.__max_average_meter__
            min_average_meter: AverageMeter = observer_module.__min_average_meter__

            real_input = input[0]
            max_val = observer_module.max_val
            min_val = observer_module.min_val
            max_average_meter.update(max_val, real_input.size(0))
            min_average_meter.update(min_val, real_input.size(0))

        hook_handles = []

        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                print_log('register `record_bn_statistics_hook` to module: '
                          f'{name}', logger='current', level=logging.DEBUG)
                module.__mean_average_meter__ = AverageMeter()
                module.__var_average_meter__ = AverageMeter()
                handle = module.register_forward_hook(
                    record_bn_statistics_hook)
                hook_handles.append(handle)
            if isinstance(module, BatchLSQObserver):
                print_log('register `record_observer_statistics_hook` to module: '
                          f'{name}', logger='current', level=logging.DEBUG)
                module.__max_average_meter__ = AverageMeter()
                module.__min_average_meter__ = AverageMeter()
                module.reset_min_max_vals()
                handle = module.register_forward_hook(
                    record_observer_statistics_hook)
                hook_handles.append(handle)

        model.train()
        print_log('Start calibrating bn and observer statistics', logger='current')
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

            if isinstance(module, BatchLSQObserver):
                max_average_meter = module.__max_average_meter__
                min_average_meter = module.__min_average_meter__
                if max_average_meter.count == 0 or \
                        min_average_meter.count == 0:
                    assert max_average_meter.count == 0 and \
                        min_average_meter.count == 0
                    print_log(f'layer {name} is not chosen, ignored',
                              logger='current', level=logging.DEBUG)
                    continue

                calibrated_max_val = max_average_meter.avg
                calibrated_min_val = min_average_meter.avg

                module.max_val.copy_(calibrated_max_val)
                module.min_val.copy_(calibrated_min_val)

                del module.__max_average_meter__
                del module.__min_average_meter__

        print_log('Remove all hooks for calibration',
                  logger='current', level=logging.DEBUG)
        print_log('Calibrate bn and observer statistics done', logger='current')
        for handle in hook_handles:
            handle.remove()
        model.apply(fix_calib_stats)


@LOOPS.register_module()
class QNASEpochBasedLoop(QATEpochBasedLoop):
    """`EpochBasedLoop` for `LEARNED STEP SIZE QUANTIZATION`

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats. Defaults to -1, which means no need to freeze bn.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            qat_begin: int = 1,
            freeze_bn_begin: int = -1,
            is_first_batch: bool = True,
            calibrate_steps: int = -1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            freeze_bn_begin=freeze_bn_begin,
            dynamic_intervals=dynamic_intervals)

        self._is_first_batch = is_first_batch
        self.distributed = is_distributed()
        self.qat_begin = qat_begin
        self.calibrate_steps = calibrate_steps

    def prepare_for_run_epoch(self):
        """Toggle the state of the observers and fake quantizers before qat
        training."""
        model = self.runner.model
        model = getattr(model, 'module', model)        
        if (self.freeze_bn_begin > 0
                and self._epoch + 1 >= self.freeze_bn_begin):
            self.runner.model.apply(freeze_bn_stats)
        if (self.qat_begin > 0
                and self._epoch + 1 >= self.qat_begin):
            self.runner.model.apply(enable_param_learning)
            model.current_stage = 'qat'
        else:
            self.runner.model.apply(enable_val)
            self.runner.model.apply(disable_fake_quant)
            model.current_stage = 'float'
        self.runner.model.apply(unfix_calib_stats)
        if self.is_first_batch and self.calibrate_steps != -1:
            # lsq observer init
            # import pdb; pdb.set_trace()
            self.runner.model.apply(enable_static_estimate)
            print_log('Star calibratiion...', logger='current')
            for idx, data_batch in enumerate(self.dataloader):
                if idx == self.calibrate_steps:
                    break
                _ = self.runner.model.calibrate_step(data_batch)
            if self.distributed:
                all_reduce_params(
                    self.runner.model.parameters(), op='mean')
            self.runner.model.sync_qparams(src_mode='predict')
            self.runner.model.apply(enable_param_learning)
            print_log('Finish calibratiion!', logger='current')

            self.prepare_for_val()
            self.runner.val_loop.run()
            self._is_first_batch = False

            self.runner.model.apply(unfix_calib_stats)
            if (self.qat_begin > 0
                    and self._epoch + 1 >= self.qat_begin):
                self.runner.model.apply(enable_param_learning)
                model.current_stage = 'qat'
            else:
                self.runner.model.apply(enable_val)
                self.runner.model.apply(disable_fake_quant)
                model.current_stage = 'float'
        print_log(f'Switch current_stage: "{model.current_stage}"', logger='current')

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)
        self.runner.model.apply(fix_calib_stats)

    @property
    def is_first_batch(self):
        return (self.qat_begin > 0
                    and self._epoch + 1 == self.qat_begin
                        and self._is_first_batch)

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            if self.is_first_batch:
                # lsq observer init
                self.runner.model.apply(enable_static_estimate)

            self.run_iter(idx, data_batch)

            if self.is_first_batch:
                # In the first batch, scale in LearnableFakeQuantize is
                # calculated through lsq observer. As the values of `scale` of
                # different observers in different rank are usually different,
                # we have to sync the `scale` here.
                if self.distributed:
                    all_reduce_params(
                        self.runner.model.parameters(), op='mean')

                # Change back to param learning mode
                self._is_first_batch = False
                self.runner.model.apply(enable_param_learning)

        self.runner.model.sync_qparams(src_mode='loss')
        # Make sure the registered buffer such as `observer_enabled` is
        # correct in the saved checkpoint.
        self.prepare_for_val()
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class QNASValLoop(ValLoop, CalibrateMixin):
    """`ValLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 evaluate_fixed_subnet: bool = False,
                 calibrate_sample_num: int = 4096,
                 estimator_cfg: Optional[Dict] = dict(type='mmrazor.ResourceEstimator'),
                 quant_bits = None,
                 only_quantized = False,
                 show_indicator = False,
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        if self.runner.distributed:
            assert hasattr(self.runner.model.module, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.module.data_preprocessor
            model = self.runner.model.module
        else:
            assert hasattr(self.runner.model, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.data_preprocessor
            model = self.runner.model
        self.model = model
        self.architecture = model.architecture.architecture
        self.architecture.data_preprocessor = data_preprocessor
        self.evaluate_fixed_subnet = evaluate_fixed_subnet
        self.calibrate_sample_num = calibrate_sample_num
        default_args = dict()
        default_args['dataloader'] = self.dataloader
        self.estimator = TASK_UTILS.build(estimator_cfg, default_args=default_args)
        self.quant_bits = quant_bits
        self.only_quantized = only_quantized
        self.show_indicator = show_indicator

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        all_metrics = dict()
        if self.model.current_stage == 'float':
            evaluate_once_func = self._evaluate_once
        else:
            evaluate_once_func = self._qat_evaluate_once

        if self.evaluate_fixed_subnet:
            metrics = evaluate_once_func()
            all_metrics.update(add_prefix(metrics, 'fix_subnet'))
        elif hasattr(self.model, 'sample_kinds'):
            if self.model.current_stage == 'float':
                sample_kinds = ['max', 'min']
            else:
                search_groups = self.model.mutator.search_groups
                quant_bits = list(filter(lambda x: x[1][0].alias and 'quant_bits' in x[1][0].alias, search_groups.items()))
                if len(quant_bits) == 0:
                    sample_kinds = ['max', 'min']
                else:
                    sample_kinds = []
                    if self.quant_bits is None:
                        for qb in quant_bits[1:]:
                            assert qb[1][0].choices == quant_bits[0][1][0].choices
                        choices = quant_bits[0][1][0].choices
                    else:
                        choices = self.quant_bits
                    for bit in choices:
                        skinds = [f'max_q{bit}'] if self.only_quantized else [f'max_q{bit}', f'min_q{bit}']
                        sample_kinds.extend(skinds)

            def qmaxmin(bit=32, is_max=True):
                def sample(mutables):
                    choice = mutables[0].max_choice if is_max else mutables[0].min_choice
                    if mutables[0].alias and 'quant_bits' in mutables[0].alias:
                        if bit in mutables[0].choices:
                            choice = bit
                        elif self.quant_bits is not None:
                            pass
                        else:
                            assert bit in mutables[0].choices
                    return choice
                return sample

            for kind in sample_kinds:  # self.model.sample_kinds:
                if kind == 'max':
                    self.model.mutator.set_max_choices()
                elif kind == 'min':
                    self.model.mutator.set_min_choices()
                elif 'random' in kind:
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices())
                elif kind.startswith('max_q') or kind.startswith('min_q'):
                    is_max = kind.split('_')[0] == 'max'
                    bit = int(kind.split('_')[1].replace('q', ''))
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices(kind=qmaxmin(bit, is_max)))
                metrics = evaluate_once_func(kind=kind)
                all_metrics.update(add_prefix(metrics, f'{kind}_subnet'))
        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')
        if self.show_indicator:
            self.get_sensitive_indicators()

    def get_sensitive_indicators(self):
        indicators_results = defaultdict(int)
        prefix = 'architecture.qmodels.tensor'
        prepared_model = _get_attrs(self.model, prefix)
        kinds = ['max', 'min']
        kinds += ['random'] * 5
        calibrate_sample_num = self.calibrate_sample_num
        for kind in kinds:
            self.model.mutator.set_choices(self.model.mutator.sample_choices(kind=kind))
            with adabn_context(self.model):
                if calibrate_sample_num > 0:
                    self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
                                                          model=self.model,
                                                          calibrate_sample_num=calibrate_sample_num)
                for node in prepared_model.graph.nodes:
                    if node.op == 'call_module':
                        maybe_lsq = _get_attrs(prepared_model, node.target)
                        if hasattr(maybe_lsq, 'weight_fake_quant'):
                            # weights
                            maybe_lsq = maybe_lsq.weight_fake_quant
                        if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                            quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                            bit = quant_bits.current_choice
                            scale, zero_point = maybe_lsq.calculate_qparams()
                            assert quant_bits.alias, f'quant_bits: {quant_bits.alias} already exists.'
                            indicators_results[quant_bits.alias] += scale.mean().item() * (2**bit - 1)
        indicators_strs = '\n Indicators Results: '
        sorted_results = sorted(indicators_results.items(), key=lambda x: x[1])
        for ind, rst in sorted_results:
            indicators_strs += f'\n {ind}: {rst}'
        self.runner.logger.info(indicators_strs)
        if self.runner.rank == 0:
            save_name = 'indicators.json'
            fileio.dump(indicators_results,
                        osp.join(self.runner.work_dir, save_name), indent=4)
        return indicators_results

    def _evaluate_once(self, kind='') -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        with adabn_context(self.architecture):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
                                                      model = self.architecture,
                                                      calibrate_sample_num = self.calibrate_sample_num)
            self.architecture.eval()
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch, self.architecture)

            _, sliced_model = export_fix_subnet(
                self.architecture, slice_weight=True)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            resource_metrics = self.estimator.estimate(sliced_model)

            metrics.update(resource_metrics)
        return metrics

    def _qat_evaluate_once(self, kind='') -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        qat_metrics = dict()
        with adabn_context(self.model):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
                                                      model = self.model,
                                                      calibrate_sample_num = self.calibrate_sample_num)
            self.model.eval()
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch, self.model)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            for key, value in metrics.items():
                qat_key = 'qat.' + key
                # ori_key = 'original.' + key
                qat_metrics[qat_key] = value
                # self.runner.message_hub.log_scalars.pop(f'val/{ori_key}', None)

        with adabn_context(self.architecture):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
                                                    model = self.architecture,
                                                    calibrate_sample_num = self.calibrate_sample_num)
            self.architecture.eval()
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch, self.architecture)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            for key, value in metrics.items():
                # qat_key = 'qat.' + key
                ori_key = 'original.' + key
                qat_metrics[ori_key] = value
                # self.runner.message_hub.log_scalars.pop(f'val/{qat_key}', None)

        _, sliced_model = export_fix_subnet(
            self.model, slice_weight=True)
        resource_metrics = self.estimator.estimate(sliced_model)
        qat_metrics.update(resource_metrics)

        return qat_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], model):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement

        outputs = model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class QNASEvolutionSearchLoop(EvolutionSearchLoop, CalibrateMixin):
    """Loop for evolution searching."""

    def __init__(self, *args, export_fix_subnet=None, solve_mode='evo_org', w_act_alphas=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.export_fix_subnet = export_fix_subnet
        self.solve_mode = solve_mode
        self.w_act_alphas = w_act_alphas
        assert solve_mode in ['evo', 'prob', 'ilp', 'evo_org', 'ilp_hawq_eigen', 'ilp_hawq_trace']
        auxiliary_estimator_cfg = dict(type='mmrazor.ResourceEstimator',
                                       input_shape=self.estimator.input_shape)
        # Auxiliary estimator could help estimate spec module's resources.
        self.auxiliary_estimator = TASK_UTILS.build(auxiliary_estimator_cfg)

    def get_hassian_probs(self, algo='hawq_trace', maxIter=100, tol=1e-3):
        from pyhessian import hessian_vector_product, group_product, normalization, orthnormal
        # import pdb; pdb.set_trace()
        # 1. get all the spec_modules.
        spec_modules = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, (DynamicConvMixin, DynamicLinearMixin)):
                spec_modules.append(name)
        # 2. forward and get it's corresponding flops / params. we use its min_choices as reference.
        self.model.mutator.set_min_choices()
        results = self.auxiliary_estimator.estimate_separation_modules(
            model=self.model, flops_params_cfg=dict(spec_modules=spec_modules, seperate_return=True))
        # filter results
        filterd_results = {}
        for key, val in results.items():
            if val['flops'] != 0 or val['params'] != 0:
                filterd_results[key] = val
        # 3. get every FakeQaunt's qrange
        qrange_results = defaultdict(int)
        prefix = 'architecture.qmodels.loss'
        prepared_model = _get_attrs(self.model, prefix)
        prob_results = {}
        # 3.1 fw_hook/bw_hook get all the values/grads for fakequant
        grad_savers, value_savers = {}, {}
        grad_dict, value_dict = {}, {}
        hook_handles = []
        quant_bits_to_node = {}
        for node in prepared_model.graph.nodes:
            if node.op == 'call_module':
                maybe_lsq = _get_attrs(prepared_model, node.target)
                if hasattr(maybe_lsq, 'weight_fake_quant'):
                    # weights
                    maybe_lsq = maybe_lsq.weight_fake_quant

                if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                    quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                    grad_savers[quant_bits.alias] = BWDSaverHook()
                    value_savers[quant_bits.alias] = FWDSaverHook()
                    handle_f = maybe_lsq.register_forward_hook(value_savers[quant_bits.alias])
                    handle_b = maybe_lsq.register_full_backward_hook(grad_savers[quant_bits.alias])
                    hook_handles.append((handle_f, handle_b))
                    quant_bits_to_node[quant_bits.alias] = node.target

        data = next(iter(self.dataloader))
        data = self.model.data_preprocessor(data)
        self.model.eval()
        self.model.apply(disable_fake_quant)
        self.model.apply(freeze_bn_stats)
        self.model.zero_grad()
        losses = self.model(**data, mode='loss')
        losses['loss'].backward(create_graph=True)
        device = losses['loss'].device
        grad_dict = {key: grad_savers[key].store_input[0] + 0. for key in grad_savers}
        value_dict = {key: value_savers[key].store_input for key in value_savers}
        for handle_f, handle_b in hook_handles:
            handle_f.remove()
            handle_b.remove()

        # 3.2 compute sensitive via hessian
        def compute_sensitive(algo, value, first_order_grad):
            if algo == 'hawq_trace':
                trace = 0.
                trace_vhv = []
                for i in range(maxIter):
                    self.model.zero_grad()
                    v = torch.randint_like(value, high=2, device=device)
                    # generate Rademacher random variables
                    v[v == 0] = -1
                    v = [v]
                    Hv = hessian_vector_product(first_order_grad, value, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                sensitive = trace
            elif algo == 'hawq_eigen':
                v = [torch.randn_like(value)]
                v = normalization(v)
                # eigenvectors = []
                eigenvalue = None
                for i in range(maxIter):
                    # v = orthnormal(v, eigenvectors)
                    self.model.zero_grad()
                    Hv = hessian_vector_product(first_order_grad, value, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()
                    v = normalization(Hv)

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                sensitive = eigenvalue
            torch.cuda.empty_cache()
            return sensitive

        def get_sensitive_delta(bit_name):

            def square_mean(ta, tb):
                return torch.pow((ta - tb), 2.0).mean().detach().cpu().numpy()

            lsq_name = quant_bits_to_node[bit_name]
            maybe_lsq = _get_attrs(prepared_model, lsq_name)
            if hasattr(maybe_lsq, 'weight_fake_quant'):
                # weights
                maybe_lsq = maybe_lsq.weight_fake_quant
            sensitive_delta = {}
            maybe_lsq.fake_quant_enabled[0] = 1
            assert maybe_lsq.mutable_attrs['quant_bits'].alias == bit_name
            bitwidth_list = maybe_lsq.mutable_attrs['quant_bits'].choices
            for bits in bitwidth_list:
                maybe_lsq.mutable_attrs['quant_bits'].current_choice = bits
                maybe_lsq.get_dynamic_params()
                value = value_dict[bit_name]
                sensitive_delta[bits] = square_mean(value, maybe_lsq(value))
            maybe_lsq.fake_quant_enabled[0] = 0

            return sensitive_delta

        sensitive_dict = {}
        for name, value in value_dict.items():
            first_order_grad = grad_dict[name]
            sensitive = compute_sensitive(algo, value, first_order_grad)
            delta = get_sensitive_delta(name)
            value_size = value.numel()
            sensitive_dict[name] = {k: v * sensitive / value_size for k, v in delta.items()}

        return filterd_results, sensitive_dict

    def get_qrange_probs(self):
        # 1. get all the spec_modules.
        spec_modules = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, (DynamicConvMixin, DynamicLinearMixin)):
                spec_modules.append(name)
        # 2. forward and get it's corresponding flops / params. we use its min_choices as reference.
        self.model.mutator.set_min_choices()
        results = self.auxiliary_estimator.estimate_separation_modules(
            model=self.model, flops_params_cfg=dict(spec_modules=spec_modules, seperate_return=True))
        # filter results
        filterd_results = {}
        for key, val in results.items():
            if val['flops'] != 0 or val['params'] != 0:
                filterd_results[key] = val
        # 3. get every FakeQaunt's qrange
        qrange_results = defaultdict(int)
        prefix = 'architecture.qmodels.tensor'
        prepared_model = _get_attrs(self.model, prefix)
        kinds = ['max', 'min']
        kinds += ['random'] * 5
        calibrate_sample_num = self.calibrate_sample_num
        if self.solve_mode in ['evo', 'evo_org']:
            calibrate_sample_num = 0
        for kind in kinds:
            self.model.mutator.set_choices(self.model.mutator.sample_choices(kind=kind))
            with adabn_context(self.model):
                if calibrate_sample_num > 0:
                    self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                          model=self.model,
                                                          calibrate_sample_num=calibrate_sample_num)
                for node in prepared_model.graph.nodes:
                    if node.op == 'call_module':
                        maybe_lsq = _get_attrs(prepared_model, node.target)
                        if hasattr(maybe_lsq, 'weight_fake_quant'):
                            # weights
                            maybe_lsq = maybe_lsq.weight_fake_quant
                        if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                            quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                            bit = quant_bits.current_choice
                            scale, zero_point = maybe_lsq.calculate_qparams()
                            assert quant_bits.alias, f'quant_bits: {quant_bits.alias} already exists.'
                            qrange_results[quant_bits.alias] += scale.mean().item() * (2**bit - 1)
        # 4. normalize qrange_results to range [-1, 1]
        belta, eps = 2.0, 10e-6
        filter_fns = [
            # lambda x: 'act_quant_bits_' in x[0],
            # lambda x: 'act_quant_bits_' not in x[0]
            lambda x: True,
        ]
        for fn in filter_fns:
            f_qrange_results = dict(filter(fn, qrange_results.items()))
            values = np.array(list(f_qrange_results.values()))
            values = belta * ((values - values.min()) / (values.max() - values.min() + eps) - 0.5)
            f_qrange_results = dict(zip(f_qrange_results.keys(), values))
            qrange_results.update(f_qrange_results)

        def adjust_function(fixed_point, ratio, inputs, ftype='linear'):
            assert ftype in ['linear', 'exp', 'log']
            def softmax(x, T=1.0):
                return np.exp(x / T) / sum(np.exp(x / T))

            outputs = []
            for input in inputs:
                if ftype == 'linear':
                    output = ratio * (input - fixed_point[0]) + fixed_point[1]
                elif ftype == 'exp':
                    output = np.exp(ratio * (input - fixed_point[0])) + (1 - fixed_point[1])
                outputs.append(output)
            outputs = softmax(np.array(outputs))
            return dict(zip(inputs, outputs))

        # 5. get each quant_bits's prob according `qrange_results`
        # Suppose fakequant A and fakequant B,
        # if qrange(A) > qrange(B), then the higher bit-width of A should get more selection prob
        # if qrange(A) < qrange(B), then the lower bit-width of A should get less selection prob.
        prob_results = {}
        for node in prepared_model.graph.nodes:
            if node.op == 'call_module':
                maybe_lsq = _get_attrs(prepared_model, node.target)
                if hasattr(maybe_lsq, 'weight_fake_quant'):
                    # weights
                    maybe_lsq = maybe_lsq.weight_fake_quant
                if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                    # activation
                    quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                    fixed_point = (np.mean(quant_bits.choices), 0.5)
                    ratio = qrange_results[quant_bits.alias]
                    prob_results[quant_bits.alias] = adjust_function(fixed_point, ratio, quant_bits.choices, ftype='linear')
                    # indicators = {}
                    # for bit in quant_bits.choices:
                    #     quant_bits.current_choice = bit
                    #     scale, zero_point = maybe_lsq.calculate_qparams()
                    #     indicators[bit] = scale.mean().item()
                    # prob_results[quant_bits.alias] = indicators

        return filterd_results, prob_results

    def _map_actfqnode2outputs(self, prepared_model):
        actfqnode2outputs = defaultdict(list)
        for node in prepared_model.graph.nodes:
            for maybe_act_fq in node.args:
                if not hasattr(maybe_act_fq, 'op') or maybe_act_fq.op != 'call_module':
                    continue
                maybe_act_fq = _get_attrs(prepared_model, maybe_act_fq.target)
                if not isinstance(maybe_act_fq, DynamicLearnableFakeQuantize):
                    continue
                quant_bits = maybe_act_fq.mutable_attrs['quant_bits']
                actfqnode2outputs[quant_bits].append(node)

        return actfqnode2outputs

    def _get_mod_mutable_or_choices(self, mod, chmutable2mods=None):
        if isinstance(mod, (DynamicConvMixin, DynamicLinearMixin)):
            out_channels = [mod.out_channels] if isinstance(mod, DynamicConvMixin) else [mod.out_features]
            out_mutable = mod.get_mutable_attr('out_channels')
            if out_mutable is not None:
                out_channels = list(set(out_mutable.choices))
            in_channels = [mod.in_channels] if isinstance(mod, DynamicConvMixin) else [mod.in_features]
            in_mutable = mod.get_mutable_attr('in_channels')
            if in_mutable is not None:
                in_channels = list(set(in_mutable.choices))
        else:
            raise TypeError(f'Unsupported mod {type(mod)}')
        w_quant_bits = mod.weight_fake_quant.mutable_attrs['quant_bits']
        act_quant_bits = mod._ACT_QUANT_BITS

        if chmutable2mods is not None:
            if out_mutable is not None:
                chmutable2mods[out_mutable].append((mod, True))
            if in_mutable is not None:
                chmutable2mods[in_mutable].append((mod, False))
        return out_channels, in_channels, w_quant_bits, act_quant_bits

    def _set_mod_chs(self, mod, ich, och, derived_mutables, origin_mutables):
        if isinstance(mod, (DynamicConvMixin, DynamicLinearMixin)):
            out_mutable = mod.get_mutable_attr('out_channels')
            if isinstance(out_mutable, DerivedMutable):
                if out_mutable in derived_mutables and derived_mutables[out_mutable] != och:
                    raise ValueError(f'Multiple mutable get different choices')
                derived_mutables[out_mutable] = och
            elif out_mutable is not None:
                if out_mutable in origin_mutables and origin_mutables[out_mutable] != och:
                    raise ValueError(f'Multiple mutable get different choices')
                origin_mutables[out_mutable] = och
                out_mutable.current_choice = och
            in_mutable = mod.get_mutable_attr('in_channels')
            if isinstance(in_mutable, DerivedMutable):
                if in_mutable in derived_mutables and derived_mutables[in_mutable] != ich:
                    raise ValueError(f'Multiple mutable get different choices')
                derived_mutables[in_mutable] = ich
            elif in_mutable is not None:
                if in_mutable in origin_mutables and origin_mutables[in_mutable] != ich:
                    raise ValueError(f'Multiple mutable get different choices')
                origin_mutables[in_mutable] = ich
                in_mutable.current_choices = ich
            str = ''
            if out_mutable is not None:
                str += f'set out_mutable: {out_mutable.alias} to {och}'
            if in_mutable is not None:
                str += f'; ... set in_mutable: {in_mutable.alias} ot {ich}'
            self.runner.logger.debug(str)
        else:
            raise TypeError(f'Unsupported mod {type(mod)}')

    def sample_ilp_once(self, filterd_results, prob_results, w_alpha=1.0, act_alpha=1.0) -> None:
        others_prob_results = deepcopy(prob_results)
        filterd_mods = dict((name, mod) for name, mod in self.model.named_modules() if name in filterd_results)

        # build interger linear programming and solve it.
        problem = LpProblem('Bit-width allocation', LpMinimize)
        variables = {}
        target, bitops, bitparams = 0, 0, 0
        key_temp = 'och.{och}-ich.{ich}-{wb_alias}.{wb}-{ab_alias}.{ab}'
        chmutable2mods = defaultdict(list)
        bitops_speedup = {}
        for name, mod in filterd_mods.items():
            out_channels, in_channels, \
                w_quant_bits, act_quant_bits = self._get_mod_mutable_or_choices(mod, chmutable2mods)
            all_choices = product(out_channels, in_channels, w_quant_bits.choices, act_quant_bits.choices)
            key_temp1 = partial(key_temp.format, wb_alias=w_quant_bits.alias, ab_alias=act_quant_bits.alias)
            bitops_speedup[w_quant_bits.alias] = filterd_results[name]['flops'] / min(w_quant_bits.choices) / min(act_quant_bits.choices)
            bitops_speedup[act_quant_bits.alias] = filterd_results[name]['flops'] / min(w_quant_bits.choices) / min(act_quant_bits.choices)
            for och, ich, wb, ab in all_choices:
                key = key_temp1(och=och, ich=ich, wb=wb, ab=ab)
                variables[key] = LpVariable(key, 0, 1, LpInteger)
                # target
                # TODO(shiguang): make different channel get different prob_results.
                if act_quant_bits.alias not in prob_results and len(act_quant_bits.choices) == 1:
                    target += (w_alpha * prob_results[w_quant_bits.alias][wb]) * variables[key]
                else:
                    target += (w_alpha * prob_results[w_quant_bits.alias][wb] + act_alpha * prob_results[act_quant_bits.alias][ab]) * variables[key]
                bitops += filterd_results[name]['flops'] / min(out_channels) * och / min(in_channels) * ich / min(w_quant_bits.choices) * wb / min(act_quant_bits.choices) * ab * variables[key]
                bitparams += filterd_results[name]['params'] / min(out_channels) * och / min(in_channels) * ich  / min(w_quant_bits.choices) * wb * variables[key]
            others_prob_results.pop(w_quant_bits.alias, None)
            others_prob_results.pop(act_quant_bits.alias, None)
            # constraint 1: only select one bit for weight, one bit for activation, one bit for output channel and one bit for input channel.
            problem += sum(variables[key_temp1(och=och, ich=ich, wb=wb, ab=ab)]
                           for och in out_channels for ich in in_channels for wb in w_quant_bits.choices for ab in act_quant_bits.choices) == 1

        problem += target
        # import pdb; pdb.set_trace()
        # save_name = 'bitops_speedup.json'
        # fileio.dump(bitops_speedup,
        #             osp.join(self.runner.work_dir, save_name), indent=4)
        # constraint 2: make sure multiple targets share the same quant_bits.
        prefix = 'architecture.qmodels.tensor'
        prepared_model = _get_attrs(self.model, prefix)
        # find activation fakequant that have multiple targets, which is also in filterd_results.
        actfqnode2outputs = self._map_actfqnode2outputs(prepared_model)
        filterd_actfqnode2outputs = defaultdict(list)
        for fq, outputs in actfqnode2outputs.items():
            count = 0
            for output in outputs:
                if output.op == 'call_module' and _get_attrs(prepared_model, output.target) in filterd_mods.values():
                    filterd_actfqnode2outputs[fq].append(_get_attrs(prepared_model, output.target))
                    count += 1
            if count == 1:
                del filterd_actfqnode2outputs[fq]
        for act_quant_bits, outputs in filterd_actfqnode2outputs.items():
            base_out_channels, base_in_channels, \
                base_w_quant_bits, _ = self._get_mod_mutable_or_choices(outputs[0])
            key_temp1 = partial(key_temp.format, wb_alias=base_w_quant_bits.alias, ab_alias=act_quant_bits.alias)
            for ab in act_quant_bits.choices:
                base_w_value = sum(variables[key_temp1(och=och, ich=ich, wb=wb, ab=ab)]
                                   for och in base_out_channels for ich in base_in_channels for wb in base_w_quant_bits.choices)
                for output in outputs[1:]:
                    select_out_channels, select_in_channels, \
                        selected_w_quant_bits, _ = self._get_mod_mutable_or_choices(output)
                    key_temp2 = partial(key_temp.format, wb_alias=selected_w_quant_bits.alias, ab_alias=act_quant_bits.alias)
                    selected_w_value = sum(variables[key_temp2(och=och, ich=ich, wb=wb, ab=ab)]
                                           for och in select_out_channels for ich in select_in_channels for wb in selected_w_quant_bits.choices)
                    problem += base_w_value == selected_w_value

        # constraint 3: make sure multiple mod share the same channel mutable.
        for chmutable, mod_list in chmutable2mods.items():
            if len(mod_list) < 2:
                continue
            base_mod, base_is_out_channel = mod_list[0]
            base_out_channels, base_in_channels, \
                base_w_quant_bits, base_act_quant_bits = self._get_mod_mutable_or_choices(base_mod)

            ref_channels = list(set(chmutable.choices))
            for rch in ref_channels:
                key_temp1 = partial(key_temp.format, wb_alias=base_w_quant_bits.alias, ab_alias=base_act_quant_bits.alias)
                if base_is_out_channel:
                    base_ref_value = sum(variables[key_temp1(och=rch, ich=ich, wb=wb, ab=ab)]
                                         for ich in base_in_channels for wb in base_w_quant_bits.choices for ab in base_act_quant_bits.choices)
                else:
                    base_ref_value = sum(variables[key_temp1(och=och, ich=rch, wb=wb, ab=ab)]
                                         for och in base_out_channels for wb in base_w_quant_bits.choices for ab in base_act_quant_bits.choices)
                for select_mod, select_is_out_channel in mod_list[1:]:
                    select_out_channels, select_in_channels, \
                        selected_w_quant_bits, selected_act_quant_bits = self._get_mod_mutable_or_choices(select_mod)
                    key_temp2 = partial(key_temp.format, wb_alias=selected_w_quant_bits.alias, ab_alias=selected_act_quant_bits.alias)
                    if select_is_out_channel:
                        select_ref_value = sum(variables[key_temp2(och=rch, ich=ich, wb=wb, ab=ab)]
                                               for ich in select_in_channels for wb in selected_w_quant_bits.choices for ab in selected_act_quant_bits.choices)
                    else:
                        select_ref_value = sum(variables[key_temp2(och=och, ich=rch, wb=wb, ab=ab)]
                                               for och in select_out_channels for wb in selected_w_quant_bits.choices for ab in selected_act_quant_bits.choices)

                    problem += base_ref_value == select_ref_value

        # # constraint 4(debuging): make sure the derived mutable and source mutables satisify derive rule.
        # for chmutable, mod_list in chmutable2mods.items():
        #     if not isinstance(chmutable, DerivedMutable):
        #         continue
        #     base_mod, base_is_out_channel = mod_list[0]
        #     base_out_channels, base_in_channels, \
        #         base_w_quant_bits, base_act_quant_bits = self._get_mod_mutable_or_choices(base_mod)

        #     all_choices = [m.choices for m in chmutable.source_mutables]
        #     product_choices = product(*all_choices)
        #     for item_choices in product_choices:
        #         for m, choice in zip(chmutable.source_mutables, item_choices):
        #             m.current_choice = choice
        #         derived_ch = chmutable.current_choice
        #         key_temp1 = partial(key_temp.format, wb_alias=base_w_quant_bits.alias, ab_alias=base_act_quant_bits.alias)
        #         if base_is_out_channel:
        #             base_ref_value = sum(variables[key_temp1(och=derived_ch, ich=ich, wb=wb, ab=ab)]
        #                                  for ich in base_in_channels for wb in base_w_quant_bits.choices for ab in base_act_quant_bits.choices)
        #         else:
        #             base_ref_value = sum(variables[key_temp1(och=och, ich=derived_ch, wb=wb, ab=ab)]
        #                                  for och in base_out_channels for wb in base_w_quant_bits.choices for ab in base_act_quant_bits.choices)
        #         for m, choice in zip(chmutable.source_mutables, item_choices):
        #             if m not in chmutable2mods:
        #                 continue
        #             select_mod, select_is_out_channel = chmutable2mods[m][0]
        #             select_out_channels, select_in_channels, \
        #                 selected_w_quant_bits, selected_act_quant_bits = self._get_mod_mutable_or_choices(select_mod)
        #             key_temp2 = partial(key_temp.format, wb_alias=selected_w_quant_bits.alias, ab_alias=selected_act_quant_bits.alias)
        #             if select_is_out_channel:
        #                 select_ref_value = sum(variables[key_temp2(och=choice, ich=ich, wb=wb, ab=ab)]
        #                                        for ich in select_in_channels for wb in selected_w_quant_bits.choices for ab in selected_act_quant_bits.choices)
        #             else:
        #                 select_ref_value = sum(variables[key_temp2(och=och, ich=choice, wb=wb, ab=ab)]
        #                                        for och in select_out_channels for wb in selected_w_quant_bits.choices for ab in selected_act_quant_bits.choices)

        #             problem += base_ref_value == select_ref_value

        # Note that there may be some modules are not searchable and not calculate
        # there `params` and `flops` in `estimate_separation_modules` phase,
        # however, the `estimate` phase will calculate all the `params` and `flops`,
        # hence, we have to subtract the differences.
        self.model.mutator.set_min_choices()
        results = self.auxiliary_estimator.estimate(model=self.model)
        # constraint 4
        if 'flops' in self.constraints_range:
            searchable_flops = sum(v['flops'] for k, v in filterd_results.items())
            fixed_flops = results['flops'] - searchable_flops
            problem += bitops >= (self.constraints_range['flops'][0] - fixed_flops)
            problem += bitops <= (self.constraints_range['flops'][1] - fixed_flops)
        # constraint 5
        if 'params' in self.constraints_range:
            searchable_params = sum(v['params'] for k, v in filterd_results.items())
            fixed_params = results['params'] - searchable_params
            problem += bitparams >= (self.constraints_range['params'][0] - fixed_params)
            problem += bitparams <= (self.constraints_range['params'][1] - fixed_params)
        # TODO: constraint 7: avg_quant_bits for weight and act, seperately.
        time_limit_in_seconds = 60
        ret = problem.solve(PULP_CBC_CMD(msg=1, maxSeconds=time_limit_in_seconds))
        if ret != 1:
            return None
        # get results for weight-only nn.Modules
        origin_mutables = dict()
        derived_mutables = dict()
        for name, mod in filterd_mods.items():
            out_channels, in_channels, \
                w_quant_bits, act_quant_bits = self._get_mod_mutable_or_choices(mod)
            all_choices = product(out_channels, in_channels, w_quant_bits.choices, act_quant_bits.choices)
            key_temp1 = partial(key_temp.format, wb_alias=w_quant_bits.alias, ab_alias=act_quant_bits.alias)
            matched_item_choices = []
            for och, ich, wb, ab in all_choices:
                key = key_temp1(och=och, ich=ich, wb=wb, ab=ab)
                if value(variables[key]) == 1.0:
                    matched_item_choices.append((och, ich, wb, ab))
            assert len(matched_item_choices) == 1, f'Unmatched or multiple matched item choices: {len(matched_item_choices)}'
            och, ich, wb, ab = matched_item_choices[0]
            self._set_mod_chs(mod, ich, och, derived_mutables, origin_mutables)
            w_quant_bits.current_choice = wb
            act_quant_bits.current_choice = ab

        self._solve_derived_mutables(derived_mutables, origin_mutables)

        # get results for Others quant_bits
        others_select_mode = 2
        for node in prepared_model.graph.nodes:
            if node.op == 'call_module':
                maybe_lsq = _get_attrs(prepared_model, node.target)
                if hasattr(maybe_lsq, 'weight_fake_quant'):
                    # weights
                    maybe_lsq = maybe_lsq.weight_fake_quant
                if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                    # activation
                    quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                    if quant_bits.alias not in others_prob_results:
                        continue
                    if others_select_mode == 1:
                        # select the highest bit;
                        quant_bits.current_choice = list(others_prob_results[quant_bits.alias].keys())[-1]
                    elif others_select_mode == 2:
                        # select the bit with the smallest prob;
                        keys = list(others_prob_results[quant_bits.alias].keys())
                        values = list(others_prob_results[quant_bits.alias].values())
                        quant_bits.current_choice = keys[values.index(min(values))]
                    elif others_select_mode == 3:
                        # select the highest bit;
                        quant_bits.current_choice = max(quant_bits.choices)
                    else:
                        # random
                        quant_bits.current_choice = random.choice(quant_bits.choices)

        candidate = self.model.mutator.current_choices
        return candidate

    def _solve_derived_mutables(self, derived_mutables, origin_mutables):
        derived_mutables_choices = defaultdict(list)
        # get all candidate choices.
        for dm, target_choice in derived_mutables.items():
            all_choices = [m.choices for m in dm.source_mutables]
            product_choices = product(*all_choices)
            current_choices = [m.current_choice for m in dm.source_mutables]
            for item_choices in product_choices:
                for m, choice in zip(dm.source_mutables, item_choices):
                    m.current_choice = choice

                if dm.current_choice == target_choice:
                    derived_mutables_choices[dm].append(item_choices)
                # reset back
                for m, choice in zip(dm.source_mutables, current_choices):
                    m.current_choice = choice
        # infer final choices from the candidates.
        derived_mutables = list(derived_mutables.keys())
        for dm, item_choices_list in derived_mutables_choices.items():
            if len(item_choices_list) == 1:
                item_choices = item_choices_list[0]
                for m, choice in zip(dm.source_mutables, item_choices):
                    if m in origin_mutables and origin_mutables[m] != choice:
                        import pdb; pdb.set_trace()
                        raise ValueError(f'mutable {m.alias} get different choices among multifple sets.')
                    m.current_choice = choice
                    origin_mutables[m] = choice
                derived_mutables.pop(derived_mutables.index(dm))

        idx = len(derived_mutables) * 20
        while(derived_mutables and idx > 0):
            idx -= 1
            dm = derived_mutables.pop(0)
            item_choices_list = derived_mutables_choices[dm]
            in_origin = np.array([True if m in origin_mutables else False for m in dm.source_mutables])
            current_choices = [m.current_choice for m in dm.source_mutables]
            match_mat = np.array(item_choices_list) == np.array(current_choices)
            match_mat = match_mat & in_origin
            # if there at most one origin is not visited, we can infer it.
            matched_item_choices = list(np.array(item_choices_list)[match_mat.sum(axis=1) + 1 >=len(in_origin)])
            if len(matched_item_choices) > 0:
                if len(matched_item_choices) > 1:
                    import pdb; pdb.set_trace()
                    print('Multiple matched, just take any one you what?')
                for m, choice in zip(dm.source_mutables, matched_item_choices[0]):
                    if m in origin_mutables and origin_mutables[m] != choice:
                        raise ValueError(f'Multiple mutable get different choices')
                    m.current_choice = choice
                    origin_mutables[m] = choice
            else:
                derived_mutables.append(dm)
        if len(derived_mutables) > 0:
            import pdb; pdb.set_trace()
            raise RuntimeError(f'Unable to infer all the origin mutables, maybe you can rewrite your searchable backbone')

    def sample_candidates(self, num_candidates=None) -> None:
        """Update candidate pool contains specified number of candicates."""
        num_candidates = num_candidates or self.num_candidates
        if self.solve_mode == 'evo_org':
            return super().sample_candidates(num_candidates=num_candidates)

        candidates_resources = []
        init_candidates = len(self.candidates)
        idx = 0
        if self.solve_mode in ['evo', 'prob', 'ilp']:
            filterd_results, prob_results = self.get_qrange_probs()
        if 'hawq' in self.solve_mode:
            algo = self.solve_mode.split('ilp_')[-1]
            filterd_results, prob_results = self.get_hassian_probs(algo=algo)
        if 'ilp' in self.solve_mode:
            num_candidates = num_candidates - init_candidates
            if self.w_act_alphas is not None:
                assert len(self.w_act_alphas) == num_candidates, 'Unmatched w_act_alphas'
                w_act_alphas = self.w_act_alphas
            else:
                act_alphas = [3*(i+1.0) / num_candidates for i in range(num_candidates)]
                w_act_alphas = [(1.0, 0.0), (0.0, 1.0)]
                w_act_alphas += [(1.0, i) for i in act_alphas]
        if self.runner.rank == 0:
            while len(self.candidates) < num_candidates:
                idx += 1
                # Note that not all the seachable mutables will be searched by `sample_ilp_once`,
                # we first make a random choice for all the mutables, and then use the solve_mode
                # to modify the specific mutables (DynamicQConv, DynamicQLinear).
                self.model.mutator.set_choices(self.model.mutator.sample_choices())
                if 'ilp' in self.solve_mode:
                    if idx > len(w_act_alphas):
                        break
                    candidate = self.sample_ilp_once(
                        filterd_results, prob_results, *w_act_alphas[idx-1])
                elif self.solve_mode == 'evo':
                    this_prob_results = deepcopy(prob_results)
                    for k, v in this_prob_results.items():
                        for b in v:
                            p = random.random()
                            this_prob_results[k][b] = p
                    candidate = self.sample_ilp_once(
                        filterd_results, this_prob_results)

                if candidate is None:
                    continue
                is_pass, result = self._check_constraints(
                    random_subnet=candidate)
                if is_pass:
                    self.candidates.append(candidate)
                    candidates_resources.append(result)
            self.candidates = Candidates(self.candidates.data)
        else:
            self.candidates = Candidates([dict(a=0)] * num_candidates)

        if len(candidates_resources) > 0:
            self.candidates.update_resources(
                candidates_resources,
                start=len(self.candidates.data) - len(candidates_resources))
            assert init_candidates + len(
                candidates_resources) == num_candidates

        # broadcast candidates to val with multi-GPUs.
        broadcast_object_list(self.candidates.data)

    @torch.no_grad()
    def _val_candidate(self, use_predictor: bool = False) -> Dict:
        """Run validation.

        Args:
            use_predictor (bool): Whether to use predictor to get metrics.
                Defaults to False.
        """
        self.prepare_for_val()
        if use_predictor:
            assert self.predictor is not None
            metrics = self.predictor.predict(self.model)
        else:
            with adabn_context(self.model):
                if self.calibrate_sample_num > 0:
                    self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                          model = self.model,
                                                          calibrate_sample_num = self.calibrate_sample_num)
                self.runner.model.eval()
                for data_batch in self.dataloader:
                    outputs = self.runner.model.val_step(data_batch)
                    self.evaluator.process(
                        data_samples=outputs, data_batch=data_batch)
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        return metrics

    def _check_constraints(
            self, random_subnet: SupportRandomSubnet) -> Tuple[bool, Dict]:
        """Check whether is beyond constraints.

        Returns:
            bool, result: The result of checking.
        """
        is_pass = True
        results = dict()
        constraints_range = deepcopy(self.constraints_range)
        if self.solve_mode != 'evo_org' and self._epoch == 0:
            default_constraints = dict()
            for k in self.constraints_range:
                if k in ('flops', 'params'):
                    default_constraints[k] = constraints_range.pop(k)
            assert len(default_constraints) > 0
            is_pass1, results1 = check_subnet_resources(
                model=self.model,
                subnet=random_subnet,
                estimator=self.auxiliary_estimator,
                constraints_range=default_constraints,
                export=False)
            is_pass &= is_pass1
            results.update(results1)
        if type(self.auxiliary_estimator) != type(self.estimator) or self._epoch > 0:
            is_pass1, results1 = check_subnet_resources(
                model=self.model,
                subnet=random_subnet,
                estimator=self.estimator,
                constraints_range=constraints_range,
                export=True)
            is_pass &= is_pass1
            results.update(results1)
        # import pdb; pdb.set_trace()
        return is_pass, results

    def _save_best_fix_subnet(self):
        """Save best subnet in searched top-k candidates."""
        best_random_subnet = self.top_k_candidates.subnets[0]
        self.model.mutator.set_choices(best_random_subnet)

        with adabn_context(self.model):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                      model=self.model,
                                                      calibrate_sample_num=self.calibrate_sample_num)
                self.model.sync_qparams(src_mode='predict')
            best_fix_subnet, sliced_model = \
                export_fix_subnet(self.model.architecture, slice_weight=True)
        if self.runner.rank == 0:
            timestamp_subnet = time.strftime('%Y%m%d_%H%M', time.localtime())
            model_name = f'subnet_{timestamp_subnet}.pth'
            save_path = osp.join(self.runner.work_dir, model_name)
            torch.save({
                'state_dict': sliced_model.state_dict(),
                'meta': {}
            }, save_path)
            self.runner.logger.info(f'Subnet checkpoint {model_name} saved in '
                                    f'{self.runner.work_dir}')
            save_name = 'best_fix_subnet.yaml'
            best_fix_subnet = convert_fix_subnet(best_fix_subnet)
            fileio.dump(best_fix_subnet,
                        osp.join(self.runner.work_dir, save_name))
            self.runner.logger.info(
                f'Subnet config {save_name} saved in {self.runner.work_dir}.')
            save_name = 'best_fix_subnet_by_module_name.yaml'
            best_fix_subnet = self.convert_fix_subnet_by_module_name(sliced_model.qmodels)
            fileio.dump(best_fix_subnet,
                        osp.join(self.runner.work_dir, save_name))
            self.runner.logger.info(
                f'Subnet config {save_name} saved in {self.runner.work_dir}.')

            self.runner.logger.info('Search finished.')

    def convert_fix_subnet_by_module_name(self, model):
        fix_subnet = {}
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                quant_info = {
                    'bit': int(np.log2(module.quant_max - module.quant_min + 1)),
                    'quant_min': module.quant_min,
                    'quant_max': module.quant_max,
                }
                fix_subnet[name] = quant_info
        return fix_subnet

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')

        self.prepare_for_val()
        if self.export_fix_subnet:
            fix_subnet = fileio.load(self.export_fix_subnet)
            _load_fix_subnet_by_mutable(self.model, fix_subnet)
            fix_subnet = self.model.mutator.current_choices
            _, result = self._check_constraints(random_subnet=fix_subnet)
            self.candidates = Candidates([fix_subnet])
            self.candidates.update_resources([result])
            self.update_candidates_scores()
            self.top_k_candidates = self.candidates
            self._save_best_fix_subnet()
            return

        if self.predictor_cfg is not None:
            self._init_predictor()

        if self.resume_from:
            self._resume()

        while self._epoch < self._max_epochs:
            self.run_epoch()
            self._save_searcher_ckpt()

        self._save_best_fix_subnet()

        self.runner.call_hook('after_train')

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)
        self.runner.model.apply(fix_calib_stats)
