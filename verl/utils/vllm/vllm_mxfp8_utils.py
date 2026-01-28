# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
MXFP8 (Microscaling FP8) utilities for vLLM rollout in verl.

MXFP8 differs from standard FP8 in several ways:
1. Uses E8M0 format for scales (8-bit exponent, no mantissa)
2. Typically uses smaller block sizes (32 elements vs 128x128)
3. Supports both 1D and 2D block quantization schemes

This module provides utilities for on-the-fly MXFP8 quantization during rollout.
"""

import logging
from dataclasses import dataclass, field
from unittest.mock import patch

import torch
import vllm
from packaging import version

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
except ImportError as e:
    raise ImportError("MXFP8 quantization not available") from e

logger = logging.getLogger(__name__)

# Default MXFP8 block quantization parameters
# MXFP8 uses E8M0 scale format with block-wise quantization
MXFP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",  # Data format is still E4M3
    "quant_method": "mxfp8",
    "weight_block_size": [1, 32],  # MXFP8 typically uses 1x32 blocks
}


@dataclass()
class MXFP8State:
    """State tracking for MXFP8 quantization parameters."""
    # A cache of mxfp8 parameter names, we can check this cache to see if a
    # param name corresponds to a mxfp8 weight
    seen_params: set = field(default_factory=lambda: set())
    mxfp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])


mxfp8_state: MXFP8State = MXFP8State()


def is_mxfp8_model(vllm_config):
    """Check if the model is configured for MXFP8 quantization."""
    try:
        from vllm.model_executor.layers.quantization.mxfp8 import MXFp8Config
        if hasattr(vllm_config, "quant_config") and isinstance(vllm_config.quant_config, MXFp8Config):
            return True
    except ImportError:
        # MXFP8 not available in this vLLM version
        pass
    return False


def get_module_from_param_name(model, name: str):
    """Get the module corresponding to a parameter name."""
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if isinstance(current_module, FusedMoE):
                return current_module
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


def is_mxfp8_weight(name, model):
    """Check if a weight should be quantized to MXFP8."""
    if name not in mxfp8_state.seen_params:
        mxfp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear layers
            if (isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn) or (
                isinstance(module, FusedMoE)
                and module.w13_weight.dtype == torch.float8_e4m3fn
                and module.w2_weight.dtype == torch.float8_e4m3fn
            ):
                mxfp8_state.mxfp8_param_names.add(name)
    return name in mxfp8_state.mxfp8_param_names


def scaled_mxfp8_blockwise(
    data_hp,
    weight_block_size,
):
    """
    Cast tensor from high precision to MXFP8 with blockwise quantization.
    
    MXFP8 uses E8M0 format for scales, which means:
    - Scale is a power of 2 (stored as 8-bit exponent)
    - More efficient hardware implementation
    - Smaller block sizes (typically 32 elements)
    
    Args:
        data_hp: High precision input tensor (2D)
        weight_block_size: Block size for quantization [block_row, block_col]
    
    Returns:
        Tuple of (quantized_data, scale_inverse)
    """
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size0 = weight_block_size[0]  # Row block size
    block_size1 = weight_block_size[1]  # Column block size

    # Save unpadded shape for later cropping
    unpadded_shape = data_hp.shape

    # Pad dimensions to be multiples of block size if needed
    pad_dim0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
    pad_dim1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1

    if pad_dim0 > 0 or pad_dim1 > 0:
        logger.debug(
            f"Padding weight from {data_hp.shape} to "
            f"({data_hp.shape[0] + pad_dim0}, {data_hp.shape[1] + pad_dim1}) "
            f"for blockwise MXFP8 quantization"
        )
        data_hp = torch.nn.functional.pad(data_hp, (0, pad_dim1, 0, pad_dim0), mode="constant", value=0)

    # FP8 E4M3 format max value
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    padded_shape = data_hp.shape
    blk_m = data_hp.shape[0] // block_size0
    blk_n = data_hp.shape[1] // block_size1

    # Reshape for block-wise processing
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)

    # For MXFP8, use E8M0 scale (power of 2)
    # Compute the exponent for E8M0 scale
    # E8M0 scale = 2^exponent, where exponent is an 8-bit integer
    # We compute scale as the nearest power of 2 that can represent max_abs / max_dtype
    
    # First compute the ideal scale
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    # For MXFP8, round scale to nearest power of 2 (E8M0 format)
    # This is done by: 2^round(log2(scale))
    log2_scale = torch.log2(scale_fp)
    log2_scale_rounded = torch.round(log2_scale)
    scale_e8m0 = torch.pow(2.0, log2_scale_rounded)
    
    descale_fp = torch.reciprocal(scale_e8m0)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_e8m0, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(padded_shape)

    # Remove padding to restore original shape
    fp_data = fp_data[: unpadded_shape[0], : unpadded_shape[1]]

    return fp_data, descale_fp


def quant_weights(weights, model, quant_config, dtype=torch.bfloat16):
    """Quantize weights to MXFP8 format."""
    weights_quantized = []
    for k, v in weights:
        if not is_mxfp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into mxfp8 and its scale factor
        if quant_config.weight_block_size is not None:
            logger.info(f"Using MXFP8 blockwise quantization for {k}")
            param_lp, param_scale = scaled_mxfp8_blockwise(
                v.to(dtype),
                weight_block_size=quant_config.weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            weights_quantized.append([k, param_lp])
            if version.parse(vllm.__version__) >= version.parse("0.11.0"):
                if "expert" in k:
                    weights_quantized.append([k + "_scale_inv", param_scale])
                else:
                    weights_quantized.append([k + "_scale", param_scale])
            else:
                weights_quantized.append([k + "_scale_inv", param_scale])

        else:
            raise ValueError(
                "Currently only support blockwise quantization, please set weight_block_size in quant_config"
            )

    return weights_quantized


def load_quanted_weights(weights, model_runner):
    """Load MXFP8 quantized weights into the model."""
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config
    vllm_dtype = model_runner.vllm_config.model_config.dtype

    weights_quantized = quant_weights(weights, model, quant_config, dtype=vllm_dtype)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weight loader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    loaded_params = model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type
    return loaded_params


def process_weights_after_loading_for_mxfp8(self, layer) -> None:
    """Quantize weights to MXFP8 format after loading.
    
    Supports both first-time loading and weight reloading.
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_quantize,
    )
    from vllm.model_executor.utils import replace_parameter
    
    # Skip if already processed (normal vLLM flow will call this)
    if getattr(layer, "_already_called_process_weights_after_loading", False):
        return
    
    # Check if this is a reload case (weight is already fp8)
    is_reload = layer.weight.dtype == torch.float8_e4m3fn
    
    if is_reload:
        # For reload: the weight should already be bf16 at this point
        # (restored by the weight loading mechanism)
        # If it's still fp8, we have a problem - the bf16 reload didn't happen
        raise RuntimeError(
            "Weight reload detected but layer.weight is still fp8. "
            "Ensure bf16 weights are properly restored before quantization."
        )
    
    weight = layer.weight.data
    
    # Ensure weight is contiguous and bf16 before quantization
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 weight, got {weight.dtype}")
    weight = weight.contiguous()
    
    # Quantize weight to MXFP8 format with swizzled scale layout
    weight_fp8, w_scale_blocked = mxfp8_quantize(weight)
    
    # Check if we have original quantized params to update (for cuDAGraph compat)
    original_weight = getattr(layer, '_original_quantized_weight', None)
    original_weight_scale = getattr(layer, '_original_quantized_weight_scale', None)
    
    if original_weight is not None and original_weight_scale is not None:
        # Reload case with cuDAGraph: copy data to original tensors
        with torch.no_grad():
            original_weight.copy_(weight_fp8)
            original_weight_scale.copy_(w_scale_blocked)
        layer.weight = original_weight
        layer.weight_scale = original_weight_scale
    else:
        # First load case: use replace_parameter to preserve weight_loader
        replace_parameter(layer, "weight", weight_fp8)
        
        if hasattr(layer, 'weight_scale') and layer.weight_scale is not None:
            replace_parameter(layer, "weight_scale", w_scale_blocked)
        else:
            layer.weight_scale = torch.nn.Parameter(w_scale_blocked, requires_grad=False)
        
        # Save references for cuDAGraph-compatible reload later
        layer._original_quantized_weight = layer.weight
        layer._original_quantized_weight_scale = layer.weight_scale
    
    layer.orig_dtype = torch.bfloat16



def apply_vllm_mxfp8_patches():
    """
    Apply vLLM patches for MXFP8 blockwise quantization.
    
    This patches vLLM's weight processing functions to support on-the-fly
    MXFP8 quantization during rollout.
    """
    logger.info("Applying vllm MXFP8 patches for blockwise quantization")
    
    # Try to patch MXFP8 specific methods if available
    try:
        func1_path = "vllm.model_executor.layers.quantization.mxfp8.Mxfp8LinearMethod.process_weights_after_loading"
        patcher1 = patch(func1_path, process_weights_after_loading_for_mxfp8)
        patcher1.start()
        mxfp8_state.vllm_patches.append(patcher1)
        logger.info("Patched MXFP8 LinearMethod.process_weights_after_loading")
    except Exception as e:
        logger.warning(f"Could not patch MXFP8 LinearMethod: {e}")