# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantization-Aware Training utilities for Megatron models."""

import logging
from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    MODELOPT_AVAILABLE = True
except ImportError:
    MODELOPT_AVAILABLE = False
    mtq = None
    SequentialQuantizer = None
    TensorQuantizer = None
    NVFP4QTensor = None
    logger.warning("ModelOpt not available. QAT will be disabled.")


@dataclass
class QATConfig:
    """Configuration for Quantization-Aware Training."""
    enabled: bool = False
    quant_method: str = "nvfp4_qat"
    

def get_nvfp4_qat_config():
    """Return the NVFP4 QAT configuration.
    
    This uses the default NVFP4 configuration from ModelOpt.
    """
    if not MODELOPT_AVAILABLE:
        raise ImportError("ModelOpt is required for QAT but not available.")
    
    # Use the default NVFP4 config
    # mtq_config = mtq.NVFP4_DEFAULT_CFG
    mtq_config = mtq.NVFP4_WEIGHT_ONLY_CFG
    

    mtq_config["quant_cfg"]["*mixer.*"] = {"enable": False}
    
    logger.info(f"NVFP4 QAT config: {mtq_config}")
    return mtq_config


def apply_qat(model: nn.Module, qat_config: QATConfig):
    """Apply Quantization-Aware Training to the model.
    
    Args:
        model: The Megatron model to apply QAT to
        qat_config: QAT configuration
        
    Returns:
        The quantized model
    """
    if not qat_config.enabled:
        logger.info("QAT is not enabled, skipping.")
        return model
    
    if not MODELOPT_AVAILABLE:
        logger.warning("ModelOpt not available, skipping QAT.")
        return model
    
    if qat_config.quant_method != "nvfp4_qat":
        raise ValueError(f"Only 'nvfp4_qat' is supported, got: {qat_config.quant_method}")
    
    logger.info(f"Applying QAT with method: {qat_config.quant_method}")
    
    # Get quantization config
    mtq_config = get_nvfp4_qat_config()
    
    # Apply quantization to the model
    # For QAT, we don't need a calibration forward loop
    mtq.quantize(model, mtq_config)
    
    logger.info("QAT applied successfully")
    
    return model


def is_qat_enabled(quantization: Optional[str]) -> bool:
    """Check if QAT is enabled based on quantization parameter.
    
    Args:
        quantization: The quantization parameter from config
        
    Returns:
        True if QAT should be enabled
    """
    return quantization == "nvfp4_qat"