# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS, TASK_UTILS
from mmrazor.structures import export_fix_subnet

from .utils import CalibrateBNMixin


@LOOPS.register_module()
class SubnetExportValLoop(ValLoop, CalibrateBNMixin):
    """Loop for subnet validation in NAS with BN re-calibration.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
        evaluate_fixed_subnet (bool): Whether to evaluate a fixed subnet only
            or not. Defaults to False.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to 4096.
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to dict(type='mmrazor.ResourceEstimator').
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
        evaluate_fixed_subnet: bool = False,
        calibrate_sample_num: int = 4096,
        estimator_cfg: Optional[Dict] = dict(type='mmrazor.ResourceEstimator'),
        is_supernet=True,
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        self.model = model
        self.evaluate_fixed_subnet = evaluate_fixed_subnet
        self.calibrate_sample_num = calibrate_sample_num
        default_args = dict()
        default_args['dataloader'] = self.dataloader
        self.estimator = TASK_UTILS.build(
            estimator_cfg, default_args=default_args)
        self.is_supernet = is_supernet

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        all_metrics = dict()

        if self.evaluate_fixed_subnet:
            metrics = self._evaluate_once()
            all_metrics.update(add_prefix(metrics, 'fix_subnet'))
        elif hasattr(self.model, 'sample_kinds'):
            for kind in self.model.sample_kinds:
                if kind == 'max':
                    self.model.mutator.set_max_choices()
                    metrics = self._evaluate_once()
                    all_metrics.update(add_prefix(metrics, 'max_subnet'))
                elif kind == 'min':
                    self.model.mutator.set_min_choices()
                    metrics = self._evaluate_once()
                    all_metrics.update(add_prefix(metrics, 'min_subnet'))
                elif 'random' in kind:
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices())
                    metrics = self._evaluate_once()
                    all_metrics.update(add_prefix(metrics, f'{kind}_subnet'))

        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')

    def _evaluate_once(self) -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        if self.calibrate_sample_num > 0:
            self.calibrate_bn_statistics(self.runner.train_dataloader,
                                         self.calibrate_sample_num)
        self.runner.model.eval()
        num_infer = 0  #1000 # len(self.dataloader.dataset)
        for idx, data_batch in enumerate(self.dataloader):
            if idx >= num_infer:
                break
            self.run_iter(idx, data_batch)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        if self.is_supernet:
            # TODO(shiguang): figure out why deepcopy will cause `CustomTracer` degenerate to `fx.Tracer`
            _, model = export_fix_subnet(model, slice_weight=True)
        metrics = {} if num_infer == 0 else self.evaluator.evaluate(num_infer)
        resource_metrics = self.estimator.estimate(model)
        metrics.update(resource_metrics)

        return metrics
