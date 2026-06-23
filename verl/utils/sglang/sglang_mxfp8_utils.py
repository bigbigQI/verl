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

import logging
import os
from typing import Any

import torch

from verl.utils.fp8_utils import FP8QuantizerHelper
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

TARGET_MXFP8_BLOCK_SIZE = [1, 32]
MXFP8_SCALE_KEY_SUFFIX = ".weight_scale_inv"
MXFP8_SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
)
MXFP8_MODULES_TO_NOT_CONVERT = (
    "lm_head",
    "embed_tokens",
    "router",
    "mlp.gate",
    "eh_proj",
    "weights_proj",
)
MXFP8_BLOCK_QUANT_KWARGS: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp8",
    "weight_block_size": TARGET_MXFP8_BLOCK_SIZE,
    "scale_fmt": "ue8m0",
    # Keep SGLang's module construction in sync with the online refit policy:
    # these layers receive BF16 weights and should not be wrapped as MXFP8
    # linear methods.
    "ignored_layers": list(MXFP8_MODULES_TO_NOT_CONVERT),
    "modules_to_not_convert": list(MXFP8_MODULES_TO_NOT_CONVERT),
}


def get_mxfp8_quant_config() -> dict[str, Any]:
    quant_config = dict(MXFP8_BLOCK_QUANT_KWARGS)
    quant_config["weight_block_size"] = list(TARGET_MXFP8_BLOCK_SIZE)
    quant_config["ignored_layers"] = list(MXFP8_MODULES_TO_NOT_CONVERT)
    quant_config["modules_to_not_convert"] = list(MXFP8_MODULES_TO_NOT_CONVERT)
    return quant_config


def _get_weight_block_size(quant_config: dict[str, Any] | Any) -> list[int] | None:
    if isinstance(quant_config, dict):
        return quant_config.get("weight_block_size")
    return getattr(quant_config, "weight_block_size", None)


def _get_config_tuple(quant_config: dict[str, Any] | Any, key: str) -> tuple[str, ...]:
    if isinstance(quant_config, dict):
        value = quant_config.get(key, ())
    else:
        value = getattr(quant_config, key, ())
    return tuple(value or ())


def _module_path_match(pattern: str, module_name: str) -> bool:
    pattern = pattern.removeprefix("model.")
    module_name = module_name.removeprefix("model.")
    if pattern == module_name:
        return True
    if module_name.startswith(pattern + "."):
        return True
    return ("." + pattern + ".") in ("." + module_name + ".")


def strip_weight_suffix(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected parameter name ending with '.weight', got: {weight_name}")
    return weight_name[: -len(".weight")]


def _quantize_with_flashinfer(tensor_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # NeMo-RL's MXFP8 refit path uses FlashInfer's swizzle-free layout for the
    # qweight/scale pair consumed by SGLang after post_process_weights.
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    return flashinfer_mxfp8_quantize(tensor_2d.contiguous(), is_sf_swizzled_layout=False)


def _quantize_with_sglang(tensor_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

    return mxfp8_group_quantize(tensor_2d)


def _quantize_mxfp8(tensor_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return _quantize_with_flashinfer(tensor_2d)
    except ImportError:
        logger.warning("flashinfer is not available; falling back to SGLang MXFP8 quantization helper")
        return _quantize_with_sglang(tensor_2d)


class SGLangMXFP8QuantizerHelper(FP8QuantizerHelper):
    def should_quantize_param(self, param_name: str, tensor: torch.Tensor | None = None):
        """Match SGLang/NeMo's HF-name MXFP8 policy for online weight refits."""
        if not param_name.endswith(".weight"):
            return False

        param_lower = param_name.lower()
        if any(pattern.lower() in param_lower for pattern in MXFP8_SKIP_WEIGHT_SUBSTRINGS):
            return False

        module_name = strip_weight_suffix(param_name)
        skip_module_patterns = (
            *_get_config_tuple(self.quant_config, "extra_high_precision_layers_hf"),
            *_get_config_tuple(self.quant_config, "modules_to_not_convert"),
        )
        if any(_module_path_match(pattern, module_name) for pattern in skip_module_patterns):
            return False

        if tensor is None:
            return True
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if tensor.dim() < 2:
            return False
        if tensor.shape[-1] % TARGET_MXFP8_BLOCK_SIZE[1] != 0:
            return False
        return True

    async def quant_weights_by_name(self, weights, dtype=torch.bfloat16):
        """Quantize SGLang rollout weights to MXFP8.

        SGLang's MXFP8 kernels use FP8 E4M3 weights with UE8M0 scales grouped
        along the input dimension in chunks of 32.
        """
        weight_block_size = _get_weight_block_size(self.quant_config)
        if weight_block_size is None:
            raise ValueError("weight_block_size not found in quant_config")
        if list(weight_block_size) != TARGET_MXFP8_BLOCK_SIZE:
            raise ValueError(
                f"SGLang MXFP8 requires weight_block_size={TARGET_MXFP8_BLOCK_SIZE}, got {weight_block_size}"
            )

        async for name, tensor in ensure_async_iterator(weights):
            if not self.should_quantize_param(name, tensor):
                yield (name, tensor)
                continue

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    logger.debug(f"Quantizing to MXFP8: {name}")

            tensor_2d = tensor.to(dtype).reshape(-1, tensor.shape[-1]).contiguous()
            param_lp, param_scale = _quantize_mxfp8(tensor_2d)

            yield (name, param_lp.view_as(tensor))
            yield (
                strip_weight_suffix(name) + MXFP8_SCALE_KEY_SUFFIX,
                param_scale.view(
                    *tensor.shape[:-1],
                    tensor.shape[-1] // TARGET_MXFP8_BLOCK_SIZE[1],
                ).contiguous(),
            )

            del tensor_2d, param_lp, param_scale
