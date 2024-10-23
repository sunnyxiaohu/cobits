# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
from typing import Any, List, Tuple

import torch

try:
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    from torch.fx import Node
    from mmengine import print_log
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')
    Node = get_placeholder('torch>=1.13')


def _get_attrs(target: torch.nn.Module, attr: str) -> Any:
    """Get the attribute from target.

    Args:
        target (torch.nn.Module): Get the attribute from target module.
        attr (str): The target attribute.

    Returns:
        Any: The target attribute.
    """

    attrs: List[str] = attr.split('.')

    for att in attrs:
        target = getattr(target, att, None)
    return target


def recursive_find_erased_nodes(node, prepared_model):
    """Find FakeQuant before target node recursively.

    Examples:
        head_fc = self.head.fc(activation_post_process_87);  \
            activation_post_process_87 = None
        activation_post_process_88 = \
            self.activation_post_process_88(head_fc);  head_fc = None
        head = self.head
        _get_loss = head._get_loss(activation_post_process_88,
            data_samples);  \
            head = activation_post_process_88 = data_samples = None
        return _get_loss

    node                       |           node.args
    --------------------
    output                     | (_get_loss, )
    _get_loss                  | (head, activation_post_process_88,
                                    data_samples)
    head                       | ()
    activation_post_process_88 | (head_fc, )
    data_samples               | (None, )
    """
    nodes_to_erase = []
    if isinstance(node, (List, Tuple)):
        for sub_node in node:
            nodes_to_erase.extend(recursive_find_erased_nodes(sub_node, prepared_model))
        return nodes_to_erase
    if node is None:
        return []

    if node.op == 'call_module' and isinstance(
            _get_attrs(prepared_model, node.target), FakeQuantizeBase):
        return [node]

    for prev_node in node.args:
        if isinstance(prev_node, Node):
            nodes_to_erase.extend(
                recursive_find_erased_nodes(prev_node, prepared_model))
        elif isinstance(prev_node,List) or isinstance(prev_node,Tuple):
            for sub_prev_node in prev_node:
                nodes_to_erase.extend(
                recursive_find_erased_nodes(sub_prev_node, prepared_model))
        else:
            print_log('Currently only support prev_node type in (List,tupe),you can fix this above')
    for prev_node in node.kwargs.values():
        if isinstance(prev_node, Node):
            nodes_to_erase.extend(
                recursive_find_erased_nodes(prev_node, prepared_model))

    return list(set(nodes_to_erase))


def del_fakequant_before_op(prepared_model,
                            target_ops: Tuple,
                            inplace: bool = True):
    """Delete useless fakequant before nodes whose ``op`` attribute (node.op)
    is in `target_ops`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_ops (tuple): Fakequants before nodes whose op attribute
            (node.op) is in `target_ops` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """

    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op in target_ops:
            # import pdb; pdb.set_trace()
            nodes_to_erase: List[Node] = recursive_find_erased_nodes(
                node, prepared_model)
            for to_erase in nodes_to_erase:
                assert to_erase.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, to_erase.target),
                    FakeQuantizeBase) and len(to_erase.args) == 1
                to_erase.replace_all_uses_with(to_erase.args[0])
                new_graph.erase_node(to_erase)
                delattr(prepared_model, to_erase.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_op(prepared_model,
                           target_ops: Tuple,
                           inplace: bool = True):
    """Delete useless fakequant after nodes whose ``op`` attribute (node.op) is
    in `target_ops`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_ops (tuple): Fakequants after nodes whose op attribute
            (node.op) is in `target_ops` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)

    target_nodes = []
    for node in new_graph.nodes:
        if node.op in target_ops:
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_before_method(prepared_model,
                                method_patterns: Tuple,
                                inplace: bool = True):
    """Delete useless fakequant before nodes whose op attribute (node.op) is
    `call_method` and target attribute (node.target) is in `target_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before nodes whose op attribute
            (node.op) is `call_method` and target attribute (node.target) is
            in `target_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_method' and node.target in method_patterns:
            nodes_to_erase: List[Node] = recursive_find_erased_nodes(
                node, prepared_model)
            for to_erase in nodes_to_erase:
                assert to_erase.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, to_erase.target),
                    FakeQuantizeBase) and len(to_erase.args) == 1
                to_erase.replace_all_uses_with(to_erase.args[0])
                new_graph.erase_node(to_erase)
                delattr(prepared_model, to_erase.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_method(prepared_model,
                               method_patterns: Tuple,
                               inplace: bool = True):
    """Delete useless fakequant after nodes whose op attribute (node.op) is
    `call_method` and target attribute (node.target) is in `target_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants after nodes whose op attribute
            (node.op) is `call_method` and target attribute (node.target)
            is in `target_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)

    target_nodes = []
    for node in new_graph.nodes:
        if node.op == 'call_method' and node.target in method_patterns:
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_before_function(prepared_model,
                                  function_patterns: Tuple,
                                  inplace: bool = True):
    """Delete useless fakequant before nodes whose op attribute (node.op) is
    `call_function` and target attribute (node.target) is in `target_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before nodes whose op attribute
            (node.op) is `call_function` and target attribute (node.target) is
            in `target_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_function' and node.target in function_patterns:
            nodes_to_erase: List[Node] = recursive_find_erased_nodes(
                node, prepared_model)
            for to_erase in nodes_to_erase:
                assert to_erase.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, to_erase.target),
                    FakeQuantizeBase) and len(to_erase.args) == 1
                to_erase.replace_all_uses_with(to_erase.args[0])
                new_graph.erase_node(to_erase)
                delattr(prepared_model, to_erase.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_function(prepared_model,
                                 function_patterns: Tuple,
                                 inplace: bool = True):
    """Delete useless fakequant after nodes whose op attribute (node.op) is
    `call_function` and target attribute (node.target) is in `target_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        function_patterns (tuple): Fakequants after nodes whose op attribute
            (node.op) is `call_function` and target attribute (node.target) is
            in `target_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)

    target_nodes = []
    for node in new_graph.nodes:
        if node.op == 'call_function' and node.target in function_patterns:
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_before_module(prepared_model,
                                module_patterns: Tuple,
                                inplace: bool = True):
    """Delete useless fakequant before modules whose type are in
    `module_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before modules whose type is in
            `module_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), module_patterns):
            to_erase = node.args[0]
            if not (to_erase.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, to_erase.target),
                    FakeQuantizeBase)):
                continue
            to_erase.replace_all_uses_with(to_erase.args[0])
            new_graph.erase_node(to_erase)
            delattr(prepared_model, to_erase.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_module(prepared_model,
                               module_patterns: Tuple,
                               inplace: bool = True):
    """Delete useless fakequant after modules whose type are in
    `module_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants after modules whose type is in
            `module_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    target_nodes = []
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), module_patterns):
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def update_qdype_qmin_qmax(fake_quant, bit=8, quant_min=None, quant_max=None, qdtype=None):
    # TODO: calc qdype according quant_min, quant_max (rely on backend support)
    # reduce_range is False by default.
    if qdtype is None:
        qdtype = fake_quant.dtype
    if quant_min is None or quant_max is None:
        quant_min = fake_quant.quant_min
        quant_max = fake_quant.quant_max

        is_symmetric_range = False
        if abs(quant_min) == abs(quant_max):
            is_symmetric_range = True
        if qdtype == torch.quint8:
            quant_min = 0
            quant_max = 2**bit - 1
        elif qdtype == torch.qint8:
            quant_max = 2**(bit - 1) - 1
            if is_symmetric_range:
                quant_min = -2**(bit - 1) + 1
            else:
                quant_min = -2**(bit - 1)
        else:
            raise ValueError(f'Only support qint8 and quint8, got {qdtype}')
    fake_quant.quant_max = \
        fake_quant.activation_post_process.quant_max = quant_max
    fake_quant.quant_min = \
        fake_quant.activation_post_process.quant_min = quant_min
    fake_quant.dtype = \
        fake_quant.activation_post_process.dtype = qdtype
    fake_quant.bitwidth = int(np.log2(quant_max - quant_min + 1))


def modify_fakequant_bits(prepared_model,
                          module_patterns: Tuple,
                          w_bit: int = 8,
                          a_bit: int = 8,
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
    def recursive_find_act_fakequant(prepared_model, dynamic_node):
        if len(dynamic_node.args) == 0:
            return None
        maybe_act = dynamic_node.args[0]
        if not (maybe_act.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, maybe_act.target), FakeQuantizeBase)):
            return recursive_find_act_fakequant(prepared_model, maybe_act)
        return  _get_attrs(prepared_model, maybe_act.target)

    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)

    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and node.target in module_patterns:
            maybe_weight = _get_attrs(prepared_model, node.target)
            if not (hasattr(maybe_weight, 'weight_fake_quant') and isinstance(
                    maybe_weight.weight_fake_quant, FakeQuantizeBase)):
                continue
            if w_bit is not None:
                maybe_weight = maybe_weight.weight_fake_quant
                update_qdype_qmin_qmax(maybe_weight, w_bit)
            if a_bit is not None:
                maybe_act = recursive_find_act_fakequant(prepared_model, node)
                if maybe_act is not None:
                    update_qdype_qmin_qmax(maybe_act, a_bit)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def register_mutables_for_dynamic_fakequant(prepared_model,
                                            module_patterns: Tuple,
                                            w_bits: List,
                                            a_bits: List,
                                            default_skipped_bit = 32,
                                            w_skip: bool = True,
                                            a_skip: bool = True,
                                            inplace: bool = True,
                                            nested_quant_bits_in_layer = False):
    """Register mutables for dynamic fakequant. It will follow the rules bellow
    to register mutables:
    1. if quant_bits contains `FLOAT_BITS`, all the dynamic fakequant will use
        register mutable by using `quant_bits` as candidate values.
    2. if not, all the dynamic fakequant except that matchs `module_patterns` will
        register mutable by using `quant_bits` as candidate values.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before and inner the modules
            whose name in `module_patterns` will be modified.
        quant_bits (List): Quant bits for building mutables.
        default_skipped_bit (int): If matched, the default skipped bit will be registered.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.
        nested_quant_bits_in_layer: Nested quant_bits of  Activations(input) to the
            corresponding layer, so that we could easily BitFlops.

    Returns:
        GraphModule: Prepared standalone module after modified.
    """
    from mmrazor.models.architectures.dynamic_qops import DynamicLearnableFakeQuantize
    from mmrazor.models.mutables import OneShotMutableValue
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)

    skipped_a_fake_quant = set()
    act_qbits_idx = 0
    other_act_nodes = set()  # record nodes that not nested in a module.
    for node in new_graph.nodes:
        if node.op == 'call_module':
            maybe_dynamic = _get_attrs(prepared_model, node.target)
            # for activations
            if isinstance(maybe_dynamic, DynamicLearnableFakeQuantize):
                # multiple nodes may share the same fakequant.
                if 'quant_bits' in maybe_dynamic.mutable_attrs:
                    continue
                alias = 'act_quant_bits_' + str(act_qbits_idx)  # node.target + '.quant_bits'
                qbits = OneShotMutableValue(alias=alias, value_list=a_bits)
                maybe_dynamic.register_mutable_attr('quant_bits', qbits)
                act_qbits_idx += 1
                other_act_nodes.add(node)
            # for weights
            elif hasattr(maybe_dynamic, 'weight_fake_quant') and isinstance(
                    maybe_dynamic.weight_fake_quant, DynamicLearnableFakeQuantize):
                maybe_dynamicw = maybe_dynamic.weight_fake_quant
                this_bits = w_bits
                if node.target in module_patterns:
                    if w_skip:
                        this_bits = [default_skipped_bit]
                    if a_skip:
                        maybe_act = recursive_find_act_fakequant(prepared_model, node)
                        if maybe_act is not None:
                            skipped_a_fake_quant.add(maybe_act)
                qbits = OneShotMutableValue(alias=node.target + '.weight_fake_quant.quant_bits', value_list=this_bits)
                maybe_dynamicw.register_mutable_attr('quant_bits', qbits)

                if nested_quant_bits_in_layer:
                    maybe_act = recursive_find_act_fakequant(prepared_model, node)
                    if maybe_act is None:
                        act_quant_bits = OneShotMutableValue(alias=node.target + '.default_act.quant_bits', value_list=[8])
                    else:
                        act_quant_bits = _get_attrs(prepared_model, maybe_act.target).mutable_attrs['quant_bits']
                        if maybe_act in other_act_nodes:
                            other_act_nodes.remove(maybe_act)
                    maybe_dynamic._ACT_QUANT_BITS = act_quant_bits
                    print_log(
                        f'Nested node w: {node} and a: {maybe_act}, with a_bits: {maybe_dynamic._ACT_QUANT_BITS.alias}', logger='current')

    if nested_quant_bits_in_layer:
        for node in new_graph.nodes:
            for consumer_node in node.args:
                # import pdb; pdb.set_trace()
                def print_other(cnode):
                    if cnode in other_act_nodes:
                        maybe_dynamic = _get_attrs(prepared_model, cnode.target)
                        print_log(f'Other node: {node}, consume a_bits: {maybe_dynamic.mutable_attrs["quant_bits"].alias}', logger='current')
                if isinstance(consumer_node, dict):
                    for k, cnode in consumer_node.items():
                        print_other(cnode)
                elif isinstance(consumer_node, (list, tuple)):
                    for cnode in consumer_node:
                        print_other(cnode)
                else:
                    print_other(consumer_node)

    for node in new_graph.nodes:
        if node in skipped_a_fake_quant:
            maybe_dynamic = _get_attrs(prepared_model, node.target)
            maybe_dynamic.mutable_attrs['quant_bits']._value_list = [default_skipped_bit]
            maybe_dynamic.mutable_attrs['quant_bits'].current_choice = default_skipped_bit

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def recursive_find_act_fakequant(prepared_model, dynamic_node):
    from mmrazor.models.architectures.dynamic_qops import DynamicLearnableFakeQuantize
    if len(dynamic_node.args) == 0:
        return None
    maybe_act = dynamic_node.args[0]
    if isinstance(maybe_act, tuple):
        maybe_act = maybe_act[0]
        print_log(f"[WARNING] Node ({dynamic_node.name})'s input is a tuple, is it a node caused by `torch.fx.wrap`?", logger='current')
    # TODO(shiguang): more general.
    if not (maybe_act.op == 'call_module' and isinstance(
            _get_attrs(prepared_model, maybe_act.target), DynamicLearnableFakeQuantize)):
        return recursive_find_act_fakequant(prepared_model, maybe_act)
    return maybe_act
