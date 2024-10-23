# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union, Dict
import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer)
    from torch.ao.quantization.quantize import propagate_qconfig_
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.utils import assert_and_get_unique_device
    from torch.ao.quantization.fx.prepare import qat_swap_modules
    from torch.ao.quantization.backend_config.utils import get_module_to_qat_module
    from torch.ao.quantization.qconfig_mapping_utils import update_qconfig_for_qat, get_flattened_qconfig_dict
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
    from torch.ao.quantization.qconfig_mapping import (
        _FIXED_QPARAMS_OP_TO_OBSERVER, FixedQParamsFakeQuantize, QConfig,
        QConfigMapping, default_weight_fake_quant)
    from torch.ao.quantization.quantize_fx import _fuse_fx
    from torch.fx.graph_module import GraphModule
    from torch.nn.intrinsic.qat import modules as qat_fused_modules
    from torch.nn.qat import modules as qat_modules
    from torch.onnx import register_custom_op_symbolic
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    GraphModule = get_placeholder('torch>=1.13')
    ObservedGraphModule = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    disable_observer = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')
    qat_fused_modules = get_package_placeholder('torch>=1.13')
    qat_modules = get_package_placeholder('torch>=1.13')
    _FIXED_QPARAMS_OP_TO_OBSERVER = get_package_placeholder('torch>=1.13')
    FixedQParamsFakeQuantize = get_package_placeholder('torch>=1.13')
    QConfig = get_package_placeholder('torch>=1.13')
    default_weight_fake_quant = get_package_placeholder('torch>=1.13')
    get_module_to_qat_module = get_package_placeholder('torch>=1.13')
    update_qconfig_for_qat = get_package_placeholder('torch>=1.13')
    propagate_qconfig_ = get_package_placeholder('torch>=1.13')
    get_flattened_qconfig_dict = get_package_placeholder('torch>=1.13')
    assert_and_get_unique_device = get_package_placeholder('torch>=1.13')

from mmrazor.registry import MODELS
from mmrazor.models import LearnableFakeQuantize
from mmrazor.models.utils import str2class
from mmrazor.models.task_modules.tracer import build_graphmodule
from mmrazor.models.task_modules.tracer.fx.graph_utils import (_get_attrs,
    modify_fakequant_bits, update_qdype_qmin_qmax, register_mutables_for_dynamic_fakequant)
from mmrazor.structures.quantization import BackendConfigs
from mmrazor.structures.quantization.backend_config.weightonly import get_weightonly_backend_config
from .native_quantizer import TorchNativeQuantizer

BackendConfigs['weightonly'] = get_weightonly_backend_config()


@MODELS.register_module()
class WeightOnlyQuantizer(TorchNativeQuantizer):
    """Quantizer for quantizing and deploying to WeightOnly backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    WeightOnly's some important features about quantization is as follows:
    * only quantize weight-matrix layer. e.g., conv, linear.
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

        flattened_qconfig_dict = get_flattened_qconfig_dict(self.qconfig_mapping)
        propagate_qconfig_(graph_module, flattened_qconfig_dict)
        # 1. insert post fakequant
        module_patterns = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
        prepared = insert_post_fakequant(graph_module, module_patterns)
        # import pdb; pdb.set_trace()
        # 2. convert modules to qat_modules
        module_to_qat_module = get_module_to_qat_module(self.backend_config)
        qat_swap_modules(prepared, module_to_qat_module)
        update_qconfig_for_qat(self.qconfig_mapping, {})
        # 3. modify fakequant bits (for input and output layer)
        prepared = modify_fakequant_bits(
            prepared, tuple(self.quant_bits_skipped_module_names), w_bit=self.default_skipped_bit,
            a_bit=self.default_skipped_bit, inplace=True)
        # 4. modify the input fakequant to dtype: torch.qint8
        update_qdype_qmin_qmax(prepared.activation_post_process_0, qdtype=torch.qint8)
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
        return 'weightonly'

    @property
    def support_w_modes(self):
        """Supported quantization modes for weight about per_tensor or
        per_channel."""
        return ('per_tensor', 'per_channel')

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor')


def insert_post_fakequant(prepared_model,
                          module_patterns: Tuple,
                          inplace: bool = True):
    """Delete useless fakequant before modules whose type are in
    `module_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before and inner the modules
            whose name in `module_patterns` will be modified.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after modified.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)

    new_graph = copy.deepcopy(prepared_model.graph)
    model_device = assert_and_get_unique_device(prepared_model)  
    idx = 0
    for node in new_graph.nodes:
        # TODO: deal with call_function
        if node.op == 'call_module':
            maybe_weight = _get_attrs(prepared_model, node.target)
            if not isinstance(maybe_weight, module_patterns):
                continue

            fakequant_name = f'activation_post_process_{idx}'
            new_fakequant = maybe_weight.qconfig.activation()
            if model_device:
                new_fakequant.to(model_device)
            setattr(prepared_model, fakequant_name, new_fakequant)
            inp = node.all_input_nodes
            with new_graph.inserting_before(node):
                inserted_node = new_graph.create_node("call_module", fakequant_name, (inp[0], ), {})
            node.args = (inserted_node, )
            idx += 1

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model
