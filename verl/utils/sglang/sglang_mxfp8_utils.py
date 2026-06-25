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

from typing import Any

import torch

from verl.utils.fp8_utils import FP8QuantizerHelper

TARGET_MXFP8_BLOCK_SIZE = [1, 32]
# Mirror the FP8 hardcode (activation_scheme / fmt / quant_method / weight_block_size),
# plus scale_fmt for the MXFP8 UE8M0 scales.
MXFP8_BLOCK_QUANT_KWARGS: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp8",
    "weight_block_size": TARGET_MXFP8_BLOCK_SIZE,
    "scale_fmt": "ue8m0",
}


def get_mxfp8_quant_config() -> dict[str, Any]:
    return dict(MXFP8_BLOCK_QUANT_KWARGS)


def _quantize_with_sglang(tensor_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # SGLang's triton MXFP8 helper: FP8 E4M3 weights + UE8M0 (uint8) scales grouped
    # along the input dim in chunks of 32, in SGLang's swizzle-free layout.
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

    return mxfp8_group_quantize(tensor_2d)


class SGLangMXFP8QuantizerHelper(FP8QuantizerHelper):
    """Quantize SGLang rollout weights to MXFP8.

    SGLang's MXFP8 kernels use FP8 E4M3 weights with UE8M0 scales grouped along
    the input dimension in chunks of 32. Only the per-weight quantization
    (``_quantize_param``) differs from :class:`FP8QuantizerHelper`; the selection
    policy, quant-config access and async iteration loop are reused from the base.
    """

    _quant_label = "MXFP8"
    # MXFP8 weights cannot fall back to bf16: a failure must surface, not silently
    # emit an unquantized weight the rollout engine can't consume.
    _reraise_quant_errors = True

    # NOTE: should_quantize_param, _get_quant_config_value and _resolve_weight_block_size
    # are intentionally NOT overridden; the base FP8 implementations are reused so MXFP8
    # stays consistent with the FP8 path. Only the per-weight quantization differs below.

    def _quantize_param(self, param_name, tensor, dtype, weight_block_size):
        tensor_2d = tensor.to(dtype).reshape(-1, tensor.shape[-1]).contiguous()
        param_lp, param_scale = _quantize_with_sglang(tensor_2d)
        scale = param_scale.view(
            *tensor.shape[:-1],
            tensor.shape[-1] // TARGET_MXFP8_BLOCK_SIZE[1],
        ).contiguous()
        return [
            (param_name, param_lp.view_as(tensor)),
            (param_name + "_scale_inv", scale),
        ]
