# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
import zlib
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnx
import torch
from mmengine.dist import broadcast_object_list, get_rank
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from torch.utils.data import DataLoader

from mmrazor.registry import METRICS, TASK_UTILS
from .resource_estimator import ResourceEstimator

try:
    import npuc
except ImportError:
    from mmrazor.utils import get_placeholder
    npuc = get_placeholder('npuc')

logger = MMLogger.get_current_instance()


@TASK_UTILS.register_module()
class OVResourceEstimator(ResourceEstimator):
    """Estimator for calculating the resources consume.

    Args:
        ovmodel_cfg (dict): Cfg for estimating ovmodel.
        input_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224).
        units (dict): Dict that contains converted FLOPs/params/latency units.
            Default to dict(flops='M', params='M', latency='ms').
        as_strings (bool): Output FLOPs/params/latency counts in a string
            form. Default to False.
        flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
            Default to None.
        latency_cfg (dict): Cfg for estimating latency. Default to None.

    Examples:
        >>> # direct calculate resource consume of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> estimator = ResourceEstimator(input_shape=(1, 3, 64, 64))
        >>> estimator.estimate(model=conv2d)
        {'flops': 3.444, 'params': 0.001, 'latency': 0.0}

        >>> # direct calculate resource consume of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> estimator = ResourceEstimator()
        >>> flops_params_cfg = dict(input_shape=(1, 3, 32, 32))
        >>> estimator.estimate(model=conv2d, flops_params_cfg)
        {'flops': 0.806, 'params': 0.001, 'latency': 0.0}

        >>> # calculate resources of custom modules
        >>> class CustomModule(nn.Module):
        ...
        ...    def __init__(self) -> None:
        ...        super().__init__()
        ...
        ...    def forward(self, x):
        ...        return x
        ...
        >>> @TASK_UTILS.register_module()
        ... class CustomModuleCounter(BaseCounter):
        ...
        ...    @staticmethod
        ...    def add_count_hook(module, input, output):
        ...        module.__flops__ += 1000000
        ...        module.__params__ += 700000
        ...
        >>> model = CustomModule()
        >>> flops_params_cfg = dict(input_shape=(1, 3, 64, 64))
        >>> estimator.estimate(model=model, flops_params_cfg)
        {'flops': 1.0, 'params': 0.7, 'latency': 0.0}
        ...
        >>> # calculate resources of custom modules with disable_counters
        >>> flops_params_cfg = dict(input_shape=(1, 3, 64, 64),
        ...                         disabled_counters=['CustomModuleCounter'])
        >>> estimator.estimate(model=model, flops_params_cfg)
        {'flops': 0.0, 'params': 0.0, 'latency': 0.0}

        >>> # calculate resources of mmrazor.models
        NOTE: check 'EstimateResourcesHook' in
              mmrazor.engine.hooks.estimate_resources_hook for details.
    """

    def __init__(
        self,
        ovmodel_cfg: dict,
        input_shape: Tuple = (1, 3, 224, 224),
        units: Dict = dict(flops='M', params='M', latency='ms'),
        as_strings: bool = False,
        flops_params_cfg: Optional[dict] = None,
        latency_cfg: Optional[dict] = None,
        dataloader: Optional[DataLoader] = None,
    ):
        super().__init__(
            input_shape,
            units,
            as_strings,
            flops_params_cfg=flops_params_cfg,
            latency_cfg=latency_cfg,
            dataloader=dataloader)
        self.ovmodel_cfg = ovmodel_cfg
        self._check_update_ovmodel_cfg(self.ovmodel_cfg)
        self.ovmodel = OVModelWrapper(
            val_data=self.dataloader.dataset, **self.ovmodel_cfg)

    def _check_update_ovmodel_cfg(self, ovmodel_cfg: dict) -> None:
        qdef_file_dir = ovmodel_cfg.pop('qdef_file_dir', '')
        for key, value in ovmodel_cfg.items():
            if key not in OVModelWrapper.__init__.__code__.co_varnames[1:]:
                raise KeyError(f'Got invalid key `{key}` in ovmodel_cfg.')
            qdef_names = ['qfnodes', 'qfops']
            if key in qdef_names:
                ovmodel_cfg[key] = osp.join(qdef_file_dir, value)

    def estimate(self,
                 model: torch.nn.Module,
                 flops_params_cfg: dict = None,
                 latency_cfg: dict = None) -> Dict[str, Union[float, str]]:
        """Estimate the resources(flops/params/latency) of the given model.

        This method will first parse the merged :attr:`self.flops_params_cfg`
        and the :attr:`self.latency_cfg` to check whether the keys are valid.

        Args:
            model: The measured model.
            flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
                Default to None.
            latency_cfg (dict): Cfg for estimating latency. Default to None.

            NOTE: If the `flops_params_cfg` and `latency_cfg` are both None,
            this method will only estimate FLOPs/params with default settings.

        Returns:
            Dict[str, Union[float, str]]): A dict that contains the resource
                results(FLOPs, params and latency).
        """
        resource_metrics = super().estimate(
            model, flops_params_cfg=flops_params_cfg, latency_cfg=latency_cfg)

        if get_rank() == 0:
            self.ovmodel.import_torch(model)
            self.ovmodel.graph_opt()
            self.ovmodel.calibrate()
            self.ovmodel.gen_ovm()
            if self.ovmodel.infer_metric is not None:
                logger.info(
                    f'Floating inference: {self.ovmodel.floating_inference()}')
                logger.info(
                    f'Fixed inference: {self.ovmodel.fixed_inference()}')
            ov_metrics = self.ovmodel.sim_ppa()
            self.ovmodel.reset_model()
        else:
            ov_metrics = dict()
        # broadcast_object_list(ov_metrics)
        resource_metrics.update(ov_metrics)
        return resource_metrics


class OVModelWrapper:
    """NPUC Wrapper class.

    This is modified from example::09-mobilenetv2
    """
    ifmq_def = {'uq8': (8, False), 'q8': (8, True), 'q16': (16, True)}

    def __init__(self,
                 work_dir,
                 val_data,
                 ifm='data',
                 ifmq='q8',
                 qfops=None,
                 qfnodes=None,
                 compression=False,
                 num_infer=None,
                 num_calib=200,
                 infer_metric=None):
        name = self.__class__.__name__
        mkdir_or_exist(work_dir)
        self.onnx_file = osp.join(work_dir, f'{name}.onnx')
        self.calibrated_qfnodes = osp.join(work_dir, f'{name}.qfnodes')
        self.ovm_file = osp.join(work_dir,
                                 f'{name}_npuc{npuc.__version__}.ovm')
        self.crc32_file = osp.join(work_dir,
                                   f'{name}_npuc{npuc.__version__}.crc32')
        self.crc32_file_semihost = osp.join(
            work_dir, f'{name}_npuc{npuc.__version__}_semihost.crc32')

        if qfnodes is not None:
            assert ifmq == qfnodes.split('.')[0].split(
                '_')[-1], f'Mismatch ifmq: {ifmq} and qfnodes: {qfnodes}'

        ## General config.
        self.ifm = ifm
        self.val_data = val_data
        self.num_infer = num_infer
        self.num_calib = num_calib
        self.compression = compression

        ## Quantizer definition files.
        self.ifmq = self.ifmq_def[ifmq]
        self.qfops_def_file = qfops
        self.qfnodes_def_file = qfnodes

        ## Others.
        self.graph = None
        self.params = None
        self.data = None
        self.label = None
        self.output = None
        self.ovm_hdl = None
        self.gid = 0

        self.reset_data()
        self.shape = (1, ) + next(self.it_val_data)['inputs'].shape
        self.in_width = self.shape[3]
        self.in_height = self.shape[2]
        self.in_channel = self.shape[1]
        self.shape = {self.ifm: self.shape}
        self.reset_data()

        self.infer_metric = infer_metric
        if infer_metric is not None:
            self.infer_metric = METRICS.build(infer_metric)

    def reset_model(self):
        self.graph = None
        self.params = None
        self.data = None
        self.label = None
        self.output = None
        self.ovm_hdl = None

    def reset_data(self):
        """Reset dataset iterator."""
        self.it_val_data = iter(self.val_data)
        self.crc32_ofm = []
        self.crc32_ifm = []

    def pre_func(self, index):
        """Pre-processsing function; for loading input images."""
        print('\r{0}'.format(index), end='')
        ## Get the images and labels
        data_batch = next(self.it_val_data, (None))
        ## MMDataset return data after transforms pipeline.
        self.data = np.asarray(data_batch['inputs'])
        self.data = np.expand_dims(self.data, axis=0)
        self.label = data_batch['data_samples']
        ## Note: NPUC inference engine takes in numpy.ndarray object.
        x = self.data
        # print(type(x))
        # print(x.shape)
        # print(type(self.label))
        # print(self.label.shape)
        return x

    def pre_func_sim(self, index, yuv_helper_cls):
        """Pre-processsing function for `npuc.ovm.sim()`; for loading input
        images.

        yuv_helper_cls is not used here as image is RGB.

        !!! attention "Expecting uint8"
            * Note that pre_func_sim is expected to return numpy.ndarray of uint8 type.
            * This is because simulator input is prior to any standardization/normalization pre-processing.
        """
        x = self.pre_func(index)
        self.crc32_ifm.append(
            '0x{:08x}\n'.format(zlib.crc32(self.data.flatten()) & 0xffffffff))
        return x

    def post_func(self, index, layer_n, quan, data, lastlayer_flag, module):
        """Post processing function; whatever is required to be performed post
        inference."""
        if lastlayer_flag:
            assert self.infer_metric is not None
            output = module.get_output(0).asnumpy().squeeze()
            self.label = self.label.set_pred_score(output).to_dict()
            # TODO(shiguang): debug when data_batch is meanfull.
            self.infer_metric.process(self.data, [self.label])

    def post_func_sim(self, index, tids, ofms):
        """Post processing function for `npuc.ovm.sim()`; whatever is required
        to be performed post inference."""
        raise NotImplementedError()
        ## npuc.ovm.sim() and npu.ovm.run() always return in NCHW as simulator always view tensor shape as such.
        ## For this model which last layer is FullyConnected, H=W=1.
        # self.output = np.squeeze(ofms[0], axis=(2,3))
        # # print(self.output.shape)
        # # print(self.label.shape)
        # self.crc32_ofm.append("0x{:08x}\n".format(zlib.crc32(self.output) & 0xffffffff))

    def import_onnx(self):
        """Import ONNX Model."""
        self.graph, self.params = npuc.frontend.from_onnx(self.onnx_file)
        #print(self.graph.json())

    def import_torch(self, model):
        device = next(model.parameters()).device
        dummy_data = self.val_data[0]['inputs'].unsqueeze(0).to(device)
        torch.onnx.export(
            model,
            dummy_data,
            self.onnx_file,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=11)
        return self.import_onnx()

    def graph_opt(self):
        """Graph optimization."""
        if self.graph is None or self.params is None:
            logger.error('graph and/orparams is None; please include job 1')
            sys.exit()

        self.graph, self.params = npuc.graphopt(self.graph, self.params)
        # print(self.graph.json())

    def floating_inference(self):
        """Floating point inference."""
        num_infer = self.num_infer
        if self.graph is None or self.params is None:
            logger.error('graph and/orparams is None; please include job 1')
            sys.exit()

        if self.ovm_hdl is None:
            self.ovm_hdl = npuc.get_ovm_hdl(
                self.graph,
                self.params,
                self.shape,
                qfops_fn=self.qfops_def_file,
                qfnodes_fn=self.qfnodes_def_file)

        self.reset_data()
        import warnings
        with warnings.catch_warnings():
            ## Ignore warning from numpy "numpy\ctypeslib.py:436: RuntimeWarning: Invalid PEP 3118 format string"
            warnings.simplefilter('ignore', lineno=436)
            npuc.floating.run(
                self.ovm_hdl,
                len(self.val_data) if num_infer is None else num_infer,
                {'func': self.pre_func}, {'func': self.post_func})
        if self.infer_metric is not None:
            metrics = self.infer_metric.compute_metrics(
                self.infer_metric.results)
            logger.info(metrics)

    def calibrate(self):
        """Calibrate the quantizers."""
        num_infer = self.num_calib
        if self.graph is None or self.params is None:
            logger.error('graph and/orparams is None; please include job 1')
            sys.exit()

        if self.ovm_hdl is None:
            self.ovm_hdl = npuc.get_ovm_hdl(
                self.graph,
                self.params,
                self.shape,
                qfops_fn=self.qfops_def_file,
                qfnodes_fn=self.qfnodes_def_file)

        ## This will produce a quantization configuration based on built-in statistical analysis.
        self.reset_data()
        self.ovm_hdl = npuc.calib.run(
            self.ovm_hdl,
            len(self.val_data) if num_infer is None else num_infer,
            {'func': self.pre_func})

        ## Always use npuc.resolve_q() after each npuc.calib.run() to fix up quantization constraint rule violations.
        self.ovm_hdl = npuc.resolve_q(self.ovm_hdl)
        ## Save the qfnodes configuration; for subsequent import or reloading of ovm_handler.
        npuc.export_qfnodes(self.ovm_hdl, self.calibrated_qfnodes)

        logger.info(
            '\nGenerating {} done. qfnodes can be used by npuc tool to verify accuracy.'
            .format(self.calibrated_qfnodes))

    def fixed_inference(self, qfnodes=None):
        """Fixed point inference."""
        num_infer = self.num_infer
        if self.graph is None or self.params is None:
            logger.error('graph and/orparams is None; please include job 1')
            sys.exit()

        if qfnodes is None:
            qfnodes = self.calibrated_qfnodes
            logger.info('Using post calibration qfnodes {}'.format(qfnodes))
        else:
            logger.info('Using provided qfnodes {}'.format(qfnodes))

        if self.ovm_hdl is None:
            self.ovm_hdl = npuc.get_ovm_hdl(
                self.graph,
                self.params,
                self.shape,
                qfops_fn=self.qfops_def_file,
                qfnodes_fn=qfnodes)
        else:
            logger.info('Importing qfnodes:{}'.format(qfnodes))
            self.ovm_hdl = npuc.import_qfnodes(self.ovm_hdl, qfnodes)

        self.reset_data()
        import warnings
        with warnings.catch_warnings():
            ## Ignore warning from numpy "numpy\ctypeslib.py:436: RuntimeWarning: Invalid PEP 3118 format string"
            warnings.simplefilter('ignore', lineno=436)
            npuc.qfixed.run(
                self.ovm_hdl,
                len(self.val_data) if num_infer is None else num_infer,
                {'func': self.pre_func}, {'func': self.post_func})
        if self.infer_metric is not None:
            metrics = self.infer_metric.compute_metrics(
                self.infer_metric.results)
            logger.info(metrics)

    def gen_ovm(self, qfnodes=None):
        """Generate OVM."""
        compression = self.compression
        if self.graph is None or self.params is None:
            logger.error('graph and/orparams is None; please include job 1')
            sys.exit()

        if qfnodes is None:
            qfnodes = self.calibrated_qfnodes
            logger.info('Using post calibration qfnodes {}'.format(qfnodes))
        else:
            logger.info('Using provided qfnodes {}'.format(qfnodes))

        if self.ovm_hdl is None:
            self.ovm_hdl = npuc.get_ovm_hdl(
                self.graph,
                self.params,
                self.shape,
                qfops_fn=self.qfops_def_file,
                qfnodes_fn=qfnodes)
        else:
            logger.info('Importing qfnodes:{}'.format(qfnodes))
            self.ovm_hdl = npuc.import_qfnodes(self.ovm_hdl, qfnodes)

        logger.info('Compression {}'.format(compression))
        # sizes = [self.shape[self.ifm]]
        npuc.ovm.gen(
            self.ovm_file, self.ovm_hdl,
            compression=compression)  # , ifm_sizes=sizes)
        logger.info(
            'Generating {} done. OVM file can be used in simulator.'.format(
                self.ovm_file))

    def sim_inference(self):
        """Simulator inference.

        !!! note "Input normalization is off-loaded to Inference Engine."
        During deployment on Inference Engine, input pre-processing can     be
        off-loaded to Inference Engine.  However with respect to     the input
        feature map (IFM) to the CNN, it is still considered     as the same as
        `floating_inference()` and `fixed_inference()`.
        """
        num_infer = self.num_infer
        self.reset_data()

        verbose = npuc.ovm.Verbose.QUIET
        # verbose = npuc.ovm.Verbose.ERROR | npuc.ovm.Verbose.WARN | npuc.ovm.Verbose.INFO
        # verbose = npuc.ovm.Verbose.ERROR | npuc.ovm.Verbose.WARN | npuc.ovm.Verbose.INFO | npuc.ovm.Verbose.LOG
        # verbose = npuc.ovm.Verbose.ERROR | npuc.ovm.Verbose.WARN | npuc.ovm.Verbose.INFO | npuc.ovm.Verbose.LOG | npuc.ovm.Verbose.TRACE
        npuc.ovm.sim(
            self.ovm_file,
            self.gid,
            self.in_width,
            self.in_height,
            self.in_channel,
            self.ifmq[0],
            len(self.val_data) if num_infer is None else num_infer,
            {'func': self.pre_func_sim}, {'func': self.post_func_sim},
            infer_type=npuc.ovm.InferType.RGB)

        ## Log the checksum of IFMs
        # with open('input.crc32', 'w', newline='\n') as file:
        # file.writelines(self.crc32_ifm)

        ## Log the checksum of OFMs
        with open(self.crc32_file, 'w', newline='\n') as file:
            file.writelines(self.crc32_ofm)

    def sim_ppa(self):
        """Simulator PPA (power/performance/area)."""

        self.reset_data()

        ppa = npuc.ovm.ppa(self.ovm_file, self.gid, self.in_width,
                           self.in_height, self.in_channel, self.ifmq[0],
                           self.ifmq[1])
        logger.info('PPA information:')
        logger.info(
            'NPU processing time     = {0:8.3f} S    for 1 inference'.format(
                ppa.time_frame))
        logger.info(
            'NPU total ops           = {0:8.3f} GOPs for 1 inference'.format(
                ppa.ops_frame))
        logger.info(
            'NPU total ddr bandwidth = {0:8.3f} MB   for 1 inference'.format(
                ppa.bandwidth_frame))
        logger.info(
            'NPU average power       = {0:8.3f} mW    at 1 inference per second (indicative only)'
            .format(ppa.power_frame))
        # TODO(shiguang): unify units for ppa.
        results = {
            'ov_npu_time': ppa.time_frame,
            'ov_ops': ppa.ops_frame,
            'ov_ddr_bandwidth': ppa.bandwidth_frame,
            'ov_power': ppa.power_frame
        }
        return results

    def sim_cgen(self, c_name=None):
        """Simulator helper; generate a C boiler-plate test_dut()."""

        self.reset_data()

        c_path = npuc.ovm.cgen(
            self.ovm_file,
            self.gid,
            self.in_width,
            self.in_height,
            self.in_channel,
            self.ifmq[0],
            c_name,
            infer_type=npuc.ovm.InferType.RGB)
        logger.info(
            'Template test_dut() for C simulator created under {}'.format(
                c_path))

    def semihost_inference(self, ipaddr='cnn', port='50051'):
        """Semihosted inference.

        !!! note "Input normalization is off-loaded to Inference Engine."
        During deployment on Inference Engine, input pre-processing can     be
        off-loaded to Inference Engine.  However with respect to     the input
        feature map (IFM) to the CNN, it is still considered     as the same as
        `floating_inference()` and `fixed_inference()`.
        """
        num_infer = self.num_infer
        self.reset_data()

        verbose = npuc.ovm.Verbose.QUIET
        # verbose = npuc.ovm.Verbose.ERROR | npuc.ovm.Verbose.WARN | npuc.ovm.Verbose.INFO
        # verbose = npuc.ovm.Verbose.ERROR | npuc.ovm.Verbose.WARN | npuc.ovm.Verbose.INFO | npuc.ovm.Verbose.LOG
        # verbose = npuc.ovm.Verbose.ERROR | npuc.ovm.Verbose.WARN | npuc.ovm.Verbose.INFO | npuc.ovm.Verbose.LOG | npuc.ovm.Verbose.TRACE

        try:
            npuc.ovm.run(
                self.ovm_file,
                self.gid,
                self.in_width,
                self.in_height,
                self.in_channel,
                self.ifmq[0],
                len(self.val_data) if num_infer is None else num_infer,
                {'func': self.pre_func_sim}, {'func': self.post_func_sim},
                infer_type=npuc.ovm.InferType.RGB,
                ip_addr=ipaddr,
                port='50051')
        except RuntimeError as err:
            print('Please ensure semihosting server is running')
            sys.exit(0)
        else:
            ## Log the checksum of IFMs
            # with open('input.crc32', 'w', newline='\n') as file:
            # file.writelines(self.crc32_ifm)

            ## Log the checksum of OFMs
            with open(self.crc32_file_semihost, 'w', newline='\n') as file:
                file.writelines(self.crc32_ofm)
