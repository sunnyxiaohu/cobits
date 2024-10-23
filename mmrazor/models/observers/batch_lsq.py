# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.distributed as dist

from mmrazor.registry import MODELS

try:
    from torch.ao.quantization.observer import (MinMaxObserver,
                                                PerChannelMinMaxObserver)
except ImportError:
    from mmrazor.utils import get_placeholder
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')


@MODELS.register_module()
class BatchLSQObserver(MinMaxObserver):
    """LSQ observer.

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """

    def __init__(self, *args, extreme_estimator=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.extreme_estimator = extreme_estimator

    def forward(self, x_orig):
        """Records the running minimum, maximum and tensor_norm of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)

        if self.extreme_estimator == 0:
            min_val_cur, max_val_cur = torch.aminmax(x)
        elif self.extreme_estimator == 1:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[1] = 0
            new_axis_list[0] = 1
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val_cur, max_val_cur = min_val_cur.mean(), max_val_cur.mean()
        elif self.extreme_estimator == 2:
            mean, var = x.mean(), x.var()
            min_val_cur, max_val_cur = mean - 3 * var, mean + 3 * var
        else:
            raise ValueError(f'Unsupported extreme estimator type: {self.extreme_estimator}')

        self.min_val.copy_(min_val_cur)
        self.max_val.copy_(max_val_cur)
        return x_orig


@MODELS.register_module()
class PerChannelBatchLSQObserver(PerChannelMinMaxObserver):
    """LSQ observer.

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """

    def __init__(self, *args, extreme_estimator=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert extreme_estimator in [1, 2], f'Unsuported extreme_estimator: {extreme_estimator}'
        self.extreme_estimator = extreme_estimator

    def forward(self, x_orig):
        """Records the running minimum, maximum and tensor_norm of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()        
        x = x.to(self.min_val.dtype)
        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)

        if self.extreme_estimator == 1:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
        elif self.extreme_estimator == 2:
            mean, var = y.mean(dim=1), y.var(dim=1)
            min_val_cur, max_val_cur = mean - 3 * var, mean + 3 * var
        else:
            raise ValueError(f'Unsupported extreme estimator type: {self.extreme_estimator}')

        self.min_val.resize_(min_val_cur.shape)
        self.max_val.resize_(max_val_cur.shape)
        self.min_val.copy_(min_val_cur)
        self.max_val.copy_(max_val_cur)
        return x_orig
