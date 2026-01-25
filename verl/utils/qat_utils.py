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
from typing import Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import modelopt.torch.quantization as mtq

    MODELOPT_AVAILABLE = True
except ImportError:
    MODELOPT_AVAILABLE = False
    mtq = None
    logger.warning("ModelOpt not available. QAT will be disabled.")


NVFP4_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        "nn.BatchNorm1d": {"*": {"enable": False}},
        "nn.BatchNorm2d": {"*": {"enable": False}},
        "nn.BatchNorm3d": {"*": {"enable": False}},
        "nn.LeakyReLU": {"*": {"enable": False}},
        "*lm_head*": {"enable": False},
        "*proj_out.*": {"enable": False},  # In Whisper model, lm_head has key name proj_out
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*router*": {"enable": False},  # Skip the MOE router
        "*mlp.gate.*": {"enable": False},  # Skip the MOE router
        "*mlp.shared_expert_gate.*": {"enable": False},  # Skip the MOE router
        "*linear_attn.conv1d*": {"enable": False},
        "*mixer.conv1d*": {"enable": False},
        "*output_layer*": {"enable": False},
        "output.*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}


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

    # mtq_config = mtq.NVFP4_WEIGHT_ONLY_CFG
    mtq_config = NVFP4_WEIGHT_ONLY_CFG

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


def reset_quantizer_amax(model: nn.Module) -> int:
    """Reset all amax values in quantizers of a QAT model.

    This function traverses all modules in the model and resets the _amax
    attribute in weight_quantizer and input_quantizer. This forces the
    quantizers to recalibrate amax during the next forward pass.

    Args:
        model: The QAT model with quantizers

    Returns:
        Number of quantizers that were reset
    """
    reset_count = 0
    from modelopt.torch.quantization.model_calib import max_calibrate
    from modelopt.torch.quantization.nn import TensorQuantizer

    before_amax_dict = {}

    pattern = "*weight_quantizer"
    import fnmatch

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and fnmatch.fnmatch(name, pattern):
            before_amax = module.amax
            module.reset_amax()
            reset_count += 1
            before_amax_dict[name] = before_amax

    max_calibrate(model, forward_loop=None, distributed_sync=True)

    for name, module in model.named_modules():
        if "layers.0" in name:
            if name in before_amax_dict:
                after_amax = module.amax
                print(f"[lark]: reset amax for: {name} before: {before_amax_dict[name]} after: {after_amax}")

    logger.info(f"Reset {reset_count} quantizer amax values for recalibration")
    return reset_count
