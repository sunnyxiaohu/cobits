from typing import Dict

import torch
import numpy as np
from pyhessian import hessian, hessian_vector_product, group_product, normalization, get_params_grad


class hessian_per_layer(hessian):

    def __init__(self, model, data=None):
        """
        model: the model that needs Hessain information
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        self.model = model.eval()  # make model is in evaluation model

        self.device = next(self.model.parameters()).device

        # pre-processing for single batch case to simplify the computation.

        # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
        data = self.model.data_preprocessor(data, True)
   
        losses = self.model(**data, mode='loss')
        losses['loss'].backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

        self.first_order_grad_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                self.first_order_grad_dict[name] = mod.weight.grad + 0.

    def layer_eigenvalues(self, maxIter=100, tol=1e-3) -> Dict:
        """
        compute the max eigenvalues in one model by layer.
        """
        device = self.device
        max_eigenvalues_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                weight = mod.weight
                eigenvalue = None
                v = [torch.randn(weight.size()).to(device)]
                v = normalization(v)
                first_order_grad = self.first_order_grad_dict[name]

                for i in range(maxIter):
                    self.model.zero_grad()

                    Hv = hessian_vector_product(first_order_grad, weight, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                    v = normalization(Hv)

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                max_eigenvalues_dict[name] = eigenvalue

        return max_eigenvalues_dict

    def layer_trace(self, maxIter=100, tol=1e-3) -> Dict:
        """
        Compute the trace of hessian in one model by layer.
        """
        device = self.device
        trace_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                trace_vhv = []
                trace = 0.
                weight = mod.weight
                first_order_grad = self.first_order_grad_dict[name]
                # import pdb; pdb.set_trace()
                for i in range(maxIter):
                    self.model.zero_grad()
                    v = torch.randint_like(weight, high=2, device=device)
                    # generate Rademacher random variables
                    v[v == 0] = -1
                    v = [v]

                    Hv = hessian_vector_product(first_order_grad, weight, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                trace_dict[name] = trace

        return trace_dict

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

class hessian_per_layer_acti(hessian):
    def __init__(self, model, data=None):
        """
        model: the model that needs Hessain information
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        self.model = model.eval()  # make model is in evaluation model

        self.device = next(self.model.parameters()).device

        # pre-processing for single batch case to simplify the computation.

        # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
        data = self.model.data_preprocessor(data, True)
   
        losses = self.model(**data, mode='loss')
        losses['loss'].backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

        self.first_order_grad_dict = {}
        self.layer_prepare(data)

    def layer_prepare(self, data):
        self.grad_savers = {}
        self.acti_savers = {}
        self.model.zero_grad()
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                self.grad_savers[name] = BWDSaverHook()
                self.acti_savers[name] = FWDSaverHook()
                mod.register_forward_hook(self.acti_savers[name])
                mod.register_full_backward_hook(self.grad_savers[name])
        losses = self.model(**data, mode='loss')
        losses['loss'].backward(create_graph=True)
        self.grad_dict = {key: self.grad_savers[key].store_input[0] + 0. for key in self.grad_savers}
        self.acti_dict = {key: self.acti_savers[key].store_input for key in self.acti_savers}

    def layer_eigenvalues(self, maxIter=100, tol=1e-3) -> Dict:
        """
        compute the top_n eigenvalues in one model by layer.
        """
        device = self.device
        max_eigenvalues_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                acti = self.acti_dict[name]
                max_eigenvalues_dict[name] = []
                first_order_grad = self.grad_dict[name]
                v = [torch.randn(acti.size()).to(device)]
                v = normalization(v)
                eigenvectors = []
                eigenvalue = None
                for i in range(maxIter):
                    v = orthnormal(v, eigenvectors)
                    self.model.zero_grad()

                    actis = [self.acti_dict[name] for name in self.acti_dict] 
                    grads = [self.grad_dict[name] for name in self.grad_dict]
                    v = [torch.randn_like(a) for a in actis]
                    Hv = hessian_vector_product(grads, actis, v)
                    Hv = hessian_vector_product(first_order_grad, acti, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                    v = normalization(Hv)
                    for i in range(len(Hv)):
                        Hv[i] = None

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            max_eigenvalues_dict[name] = eigenvalue
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                        max_eigenvalues_dict[name] = eigenvalue

        return max_eigenvalues_dict

    def layer_trace(self, maxIter=100, tol=1e-3) -> Dict:
        """
        Compute the trace of hessian in one model by layer.
        """
        device = self.device
        trace_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                trace_vhv = []
                trace = 0.
                acti = self.acti_dict[name]
                first_order_grad = self.grad_dict[name]
                for i in range(maxIter):
                    self.model.zero_grad()
                    v = torch.randint_like(acti, high=2, device=device)
                    # generate Rademacher random variables
                    v[v == 0] = -1
                    v = [v]
                    Hv = hessian_vector_product(first_order_grad, acti, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                    torch.cuda.empty_cache()
                trace_dict[name] = trace
        return trace_dict
