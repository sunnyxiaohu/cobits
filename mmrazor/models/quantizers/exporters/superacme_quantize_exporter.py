# Copyright (c) OpenMMLab. All rights reserved.

from typing import List
import numpy as np
import json
import logging

try:
    import onnx
    from onnx import helper, numpy_helper
except ImportError:
    from mmrazor.utils import get_package_placeholder
    onnx = get_package_placeholder('No module named onnx')
    numpy_helper = get_package_placeholder('No module named onnx.numpy_helper')
    helper = get_package_placeholder('No module named onnx.helper')

from mmrazor.models.quantizers.exporters.base_quantize_exporter import BaseQuantizeExportor ,PERCHANNEL_FAKEQUANTIZER,PERTENSOR_FAKEQUANTIZER
from mmengine import print_log


class SuperAcmeQuantizeExportor(BaseQuantizeExportor):

    def __init__(self, onnx_model, export_path) -> None:
        super().__init__(onnx_model, export_path)
        self.optimizer.replace_resize_op_with_upsample(self.onnx_model, self.output2node)
        self._remap_input_and_node()
        self._remap_output_and_node()

    def _insert_initializers_to_onnx(self, initializers: List):
        """Insert onnx initializers to the onnx graph."""
        inserted_init_names = set()
        for init in initializers:
            if init.name in inserted_init_names:
                continue

            self.onnx_model.graph.initializer.append(init)
            inserted_init_names.add(init.name)
            
    def clip_weight(self, node, name2data, named_initializer):
        tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node)
        data = name2data[tensor_name]
        clip_range_min = ((qmin - zero_point) * scale).astype(data.dtype)
        clip_range_max = ((qmax - zero_point) * scale).astype(data.dtype)
        if len(scale.shape) > 0 and scale.shape[0] > 1:
            new_data = []
            transposed = False
            if data.shape[0] != scale.shape[0]:
                transposed = True
                data = data.transpose(1, 0, 2, 3)
            for c in range(data.shape[0]):
                new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            if not np.allclose(data, new_data):
                print_log(f'Clip weights <{tensor_name}> to per-channel ranges.', logger='current', level=logging.WARNING)
            if transposed:
                new_data = new_data.transpose(1, 0, 2, 3)
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
            if not np.allclose(data, new_data):
                print_log(f'Clip weights <{tensor_name}> to range [{clip_range_min}, {clip_range_max}].', logger='current', level=logging.WARNING)
        new_data = numpy_helper.from_array(new_data)
        named_initializer[tensor_name].raw_data = new_data.raw_data
        
    def weight_preprocess(self, target_tensor, out2node, inp2node, named_initializer):
        def find_weight(tensor):
            if tensor not in named_initializer:
                _node = out2node[tensor]
                for inp in _node.input:
                    return find_weight(inp)
            return tensor
        weight = find_weight(target_tensor)

        # TODO need more general method, like onnxruntime infer
        data = numpy_helper.to_array(named_initializer[weight])
        data = np.tanh(data)
        data = data / (np.max(np.abs(data)) + 1e-5)
        data = numpy_helper.from_array(data)
        named_initializer[weight].raw_data = data.raw_data

        redundant_nodes = []

        def find_redundant_nodes(tensor):
            if tensor == target_tensor:
                return
            nodes = inp2node[tensor]
            for node, idx in nodes:
                if node not in redundant_nodes:
                    redundant_nodes.append(node)
                    redundant_nodes.extend(self._get_constant_inputs(node))
                find_redundant_nodes(node.output[0])
        find_redundant_nodes(weight)
        return weight, redundant_nodes
    
    def prepare_initializer(self,onnx_model):
        named_initializer = {}
        for init in onnx_model.graph.initializer:
            named_initializer[init.name] = init
        return named_initializer
    
    def deal_with_weight_fakequant(self, node, out2node, inp2node, named_initializer):
        next_nodes = inp2node[node.output[0]]
        assert len(next_nodes) == 1
        next_node, idx = next_nodes[0]
        assert next_node.op_type in ['Conv', 'Gemm', 'ConvTranspose']
        redundant_nodes = []
        if node.input[0] not in named_initializer:
            node.input[0], redundant_nodes = \
                self.weight_preprocess(node.input[0], out2node, inp2node, named_initializer)
        next_node.input[idx] = node.input[0]
        return redundant_nodes

    def deal_with_activation_fakequant(self, node, inp2node):
        next_nodes = inp2node[node.output[0]]
        for next_node, idx in next_nodes:
            next_node.input[idx] = node.input[0]
        return
            
    def clip_and_collect_params(self, symbolic_nodes: List):
        """gen clip range jsonfile."""
        named_initializer = self.prepare_initializer(onnx_model=self.onnx_model)
        nodes_to_be_removed=[]
        clip_ranges = {}
        for node in symbolic_nodes:
            if node.op_type in PERCHANNEL_FAKEQUANTIZER:
                redundant_nodes = self.deal_with_weight_fakequant(node, self.output2node, self.input2node, named_initializer)
                nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node, self.name2data, named_initializer)
                tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(
                    node)
                clip_ranges[tensor_name] = {'step': [float(x) for x in scale],
                                                'zero_point': [int(x) for x in zero_point],
                                                'min': [float(x) for x in scale * (qmin - zero_point)],
                                                'max': [float(x) for x in scale * (qmax - zero_point)],
                                                'bit': int(np.log2(qmax - qmin + 1)),
                                                'type': "biased",
                                                }
            elif node.op_type in PERTENSOR_FAKEQUANTIZER:
                if node.output[0] in [x.name for x in self.graph.output]:
                    self.input2node[node.output[0]] = []

                next_nodes = self.input2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, self.output2node, self.input2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, self.name2data, named_initializer)
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, self.input2node)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node)
                clip_ranges[tensor_name] = {'step': float(scale),
                                                'zero_point': int(zero_point),
                                                'min': float(scale * (qmin - zero_point)),
                                                'max': float(scale * (qmax - zero_point)),
                                                'bit': int(np.log2(qmax - qmin + 1)),
                                                'type': "biased",
                                                }
                prev_node = self.output2node[node.input[0]]
                # (shiguang): Uncomment these codes for adjusting scale accoding superacme backends.
                # Note that the pattern like `relu -> concat -> fakequant` is not handle properly.
                if prev_node.op_type in ['Relu', 'Clip']:
                    adjust_range = clip_ranges[tensor_name]
                    min1, max1, bit = 0, (adjust_range['max'] - adjust_range['min']), adjust_range['bit']
                    if prev_node.op_type == 'Clip':
                        max1 = min(6.0, max1)
                    adjust_range['min'], adjust_range['max'] = min1, max1
                    adjust_range['step'] = max1 / (2**bit-1)
                    adjust_range['zero_point'] = 0 - (2**(bit-1))
                    clip_ranges[tensor_name] = adjust_range
        return clip_ranges,nodes_to_be_removed
    
    
    def post_process_clip_ranges(self, clip_ranges, graph, inp2node, outp2node):
        def find_the_closest_clip_range(node):
            if node.input[0] in clip_ranges:
                return node.input[0]
            # look forward
            ret = None
            if node.op_type in ['Flatten', 'Resize', 'Relu', 'Clip','Concat', 'MaxPool'] and node.output[0] in inp2node:
                ret = find_the_closest_clip_range(inp2node[node.output[0]][0][0])
            # Temporal plan, may not correct.
            # if ret is not None:
            #     return ret
            # # look backward
            # if node.op_type in ['Flatten', 'Resize', 'Relu', 'Clip','Concat', 'MaxPool'] and node.input[0] in outp2node:
            #     ret = find_the_closest_clip_range(outp2node[node.input[0]])
            return ret

        for node in graph.node:
            if node.op_type in ['Flatten', 'Resize', 'Relu','Clip', 'Concat', 'MaxPool']:
                tensor_name = find_the_closest_clip_range(node)
                if tensor_name:
                    for i in range(len(node.input)):
                        clip_ranges[node.input[i]] = clip_ranges[tensor_name]
                        inputname = node.input[i]
                        print_log(f'Pass <{tensor_name}> clip range to <{node.name}> input <{inputname}>.', logger='current', level=logging.DEBUG)
        return clip_ranges


    def _collect_symbolic_constant_inputs(self, symbolic_nodes: List):
        """Collect these constant nodes which is the input of all the symbolic
        node."""

        collected_constant_names = set()
        constant_inputs_out = list()
        for node in symbolic_nodes:
            constant_inputs = self._get_constant_inputs(node)
            for constant in constant_inputs:
                if constant.name in collected_constant_names:
                    continue
                constant_inputs_out.append(constant)
                collected_constant_names.add(constant.name)
        return constant_inputs_out
    
    def _remove_symbolic_related(self):
        """removeing symbolic related nodes and initializers in the original
        onnx model ."""
        # import pdb; pdb.set_trace()
        symbolic_nodes = self.collect_symbolic_nodes(self.onnx_model)
        self.clip_ranges,nodes_to_be_removed = self.clip_and_collect_params(symbolic_nodes)

        symbolic_nodes.extend(nodes_to_be_removed)
        
        collect_func = self._collect_symbolic_constant_inputs
        # Usually different activation fakequants share the same constant
        # input, and different weight fakequants share the same constant input.
        symbolic_constant_inputs = collect_func(symbolic_nodes)

        self._remove_symbolic_related_from_onnx(symbolic_nodes,
                                                symbolic_constant_inputs)
        
        self.optimizer.optimize(self.onnx_model)
        self.clip_ranges = self.post_process_clip_ranges(
            self.clip_ranges, self.graph, self.input2node, self.output2node)
        
        self.context = {"ppl": self.clip_ranges}

    def export(self):
        """Export end to end onnx model."""
        self._remove_symbolic_related()
        onnx.save(self.onnx_model, self.export_path)
        context_filename = self.export_path.replace('.onnx','_superacme_clip_ranges.json')
        with open(context_filename, 'w') as f:
            json.dump(self.context, f, indent=4)
