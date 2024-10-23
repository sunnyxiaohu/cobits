# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union, Dict

import torch
from mmrazor.registry import MODELS
from mmrazor.models.utils import str2class
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicMixin,
                                                      DynamicLinearMixin,
                                                      DynamicSequential)
from mmrazor.models.task_modules.tracer import build_graphmodule
from mmrazor.models.task_modules.tracer.fx.graph_utils import (_get_attrs,
    modify_fakequant_bits, register_mutables_for_dynamic_fakequant)
from mmrazor.structures.quantization import BackendConfigs

try:
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.quantize_fx import _fuse_fx
except ImportError:
    from mmrazor.utils import get_placeholder
    prepare = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')

from mmrazor.structures.quantization import BackendConfigs
from mmrazor.structures.quantization.backend_config.snpe import get_snpe_backend_config
from .native_quantizer import TorchNativeQuantizer

BackendConfigs['snpe'] = get_snpe_backend_config()

@MODELS.register_module()
class SNPEQuantizer(TorchNativeQuantizer):
    """Quantizer for quantizing and deploying to SNPE backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    SNPE's some important features about quantization is as follows:
    * support_w_mode = ('per_tensor')
    * support_a_mode = ('per_tensor')
    * weight range could be asymmetric
    """
    def __init__(self, *args, tracer: Dict = dict(type='CustomTracer'),
                 w_bits=None, a_bits=None, quant_bits_skipped_module_names=None,
                 default_skipped_bit=8, nested_quant_bits_in_layer=False,
                 w_skip=True, a_skip=True, **kwargs):
        if 'skipped_module_classes' in tracer:
            tracer['skipped_module_classes'] = str2class(tracer['skipped_module_classes'])
        super().__init__(*args, tracer=tracer, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_skip = w_skip
        self.a_skip = a_skip
        if quant_bits_skipped_module_names is None:
            quant_bits_skipped_module_names = []
        self.quant_bits_skipped_module_names = quant_bits_skipped_module_names
        self.default_skipped_bit = default_skipped_bit
        self.nested_quant_bits_in_layer = nested_quant_bits_in_layer

    def prepare(self, model, concrete_args=None):
        """prepare graph to ObservedGraphModule.

        Returns:
            ObservedGraphModule: GraphModules after fuse and observer.

        Notes:
            'graph_module' after '_fuse_fx()' function will fuse conv, BN, ReLU
            into modules in SUPPORT_QAT_MODULES.
            'graph_module' after 'prepare()' function will become observed.

        Notes:
            Keep `is_qat` is True is because in Pytorch when `is_qat` is false,
            the `_fuse_fx()` function only fuse module into `nn.Squential`.
            In mmrazor, we aim to add more ptq algorithm into our pipeline such
            as Adaround, these kind of ptq method have some additional
            fake_quant  operations that we need it to be fused into our
            `SUPPORT_QAT_MODULES` type, which is a tricky way to deal with it.
        """
        self.swap_ff_with_fxff(model)
        traced_graph = self.tracer.trace(model, concrete_args=concrete_args)
        graph_module = build_graphmodule(model, traced_graph)

        # set the training modes of all modules to True to `_fuse_fx` correctly
        # todo: check freezebn
        self.sync_module_training_mode(graph_module, mode=True)
        # 1. fuse conv-bn
        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        # 2. convert to qat_module and insert fakequant
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)
        # 3. delete redundant fakequant
        prepared = self.del_redundant_fakequant(prepared)
        # import pdb; pdb.set_trace()
        # 4. modify fakequant bits (for input and output layer)
        prepared = modify_fakequant_bits(
            prepared, tuple(self.quant_bits_skipped_module_names), w_bit=self.default_skipped_bit,
            a_bit=self.default_skipped_bit, inplace=True)
        # 5. register mutables for mixed-precision search.
        if self.w_bits or self.a_bits:
            prepared = register_mutables_for_dynamic_fakequant(
                prepared, tuple(self.quant_bits_skipped_module_names), w_bits=self.w_bits, a_bits=self.a_bits,
                default_skipped_bit=self.default_skipped_bit, w_skip=self.w_skip, a_skip=self.a_skip,
                inplace=True, nested_quant_bits_in_layer=self.nested_quant_bits_in_layer)

        return prepared

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'snpe'

    @property
    def support_w_modes(self):
        """Supported quantization modes for weight about per_tensor or
        per_channel."""
        return ('per_tensor')

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor')

    def export_onnx(self,
                    model: Union[torch.nn.Module, torch.jit.ScriptModule,
                                 torch.jit.ScriptFunction],
                    args: Union[Tuple[Any, ...], torch.Tensor],
                    output_path: str,
                    opset_version: Optional[int] = 11,
                    **kwargs):
        """Export the onnx model that can be deployed to SNPE backend."""

        symbolic_output_path = output_path.replace('.onnx', '_symbolic.onnx')
        torch.onnx.export(
            model,
            args,
            symbolic_output_path,
            opset_version=opset_version,
            **kwargs)

        from .exporters import SNPEQuantizeExportor
        exporter = SNPEQuantizeExportor(symbolic_output_path, output_path)
        exporter.export()

    @property
    def module_prev_wo_fakequant(self):
        """Configurate the modules that their previous nodes are redundant
        fakequants."""
        return (torch.nn.ReLU6, torch.nn.Identity)

    @property
    def module_next_wo_fakequant(self):
        """Configurate the modules that their next nodes are redundant
        fakequants."""
        return (torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)

    @property
    def method_next_wo_fakequant(self):
        """Configurate the methods that their next nodes are redundant
        fakequants."""
        return ('flatten', )

    @property
    def op_prev_wo_fakequant(self):
        """Configurate the OPs that their previous nodes are redundant
        fakequants."""
        return ('output', )
