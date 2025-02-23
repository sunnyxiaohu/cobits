# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union, Dict, List
import torch

try:
    from torch.ao.quantization.backend_config import (BackendConfig,
                                                      BackendPatternConfig,
                                                      DTypeConfig,
                                                      ObservationType)
except ImportError:
    from mmrazor.utils import get_placeholder
    BackendConfig = get_placeholder('torch>=1.13')
    BackendPatternConfig = get_placeholder('torch>=1.13')
    DTypeConfig = get_placeholder('torch>=1.13')
    ObservationType = get_placeholder('torch>=1.13')

from .common_operator_config_utils import (_get_conv_configs,
                                           _get_linear_configs,
                                           _get_dynamicconv_configs,
                                           _get_dynamiclinear_configs)

# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_weightonly_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for weightonly reseaching.

    Note:
        Learn more about BackendConfig, please refer to:
        https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config # noqa: E501
    """

    # ===================
    # |  DTYPE CONFIGS  |
    # ===================
    # weighted op int8 dtype config
    # this is config for ops that has quantized weights, like linear, conv
    weighted_op_int8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    conv_dtype_configs = [weighted_op_int8_dtype_config]
    linear_dtype_configs = [weighted_op_int8_dtype_config]

    return BackendConfig('weightonly') \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))  \
        .set_backend_pattern_configs(_get_dynamicconv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_dynamiclinear_configs(linear_dtype_configs))


def get_weightonly_backend_config_dict():
    """Return the `BackendConfig` for weightonly reseaching in dictionary
    form."""
    return get_weightonly_backend_config().to_dict()


__all__ = [
    'get_weightonly_backend_config',
    'get_weightonly_backend_config_dict',
]
