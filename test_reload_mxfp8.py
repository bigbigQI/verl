"""
Standalone test for MXFP8 quantized weight sync with SGLang server.

Tests the same flow used in the project's sglang rollout (sglang_rollout.py):
  1. Launch SGLang HTTP server with dummy weights + mxfp8 quantization config
  2. Load pre-quantized MXFP8 weights from safetensors on disk
  3. Sync weights to server via sgl_update_weights (same as training loop)
  4. Verify inference produces meaningful results after sync

Usage:
    # Single GPU (TP=1)
    python tests/test_sglang_mxfp8_weight_sync.py

    # Custom settings
    python tests/test_sglang_mxfp8_weight_sync.py --tp-size 1 --port 30000

    # Multi-GPU (TP=4), must init with torchrun
    torchrun --nproc_per_node=4 tests/test_sglang_mxfp8_weight_sync.py --tp-size 4
"""

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os

import torch
import torch.distributed as dist
from safetensors import safe_open
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import init_device_mesh

from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_mxfp8")

# ─── Default Configuration ─────────────────────────────────────────
MXFP8_MODEL_PATH = "/apps/quant_models/qwen3_30b_mxfp8"
#MXFP8_MODEL_PATH = "/apps/quant_models/qwen3_8b_mxfp8"

# Same mxfp8 quantization config as in sglang_rollout.py & async_sglang_server.py
MXFP8_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp8",
    "weight_block_size": [1, 32],
    "scale_fmt": "ue8m0",
}

BUCKET_BYTES = 256 << 20  # 256 MB per bucket (same scale as project default)


def parse_args():
    p = argparse.ArgumentParser(description="Test MXFP8 weight sync with SGLang")
    p.add_argument("--model-path", type=str, default=MXFP8_MODEL_PATH)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--mem-fraction", type=float, default=0.7)
    p.add_argument("--bucket-mb", type=int, default=256, help="Weight sync bucket size in MB")
    return p.parse_args()


# ─── Weight Loading ────────────────────────────────────────────────
def load_safetensors_weights(model_dir: str, device: str = "cpu"):
    """Load all weight tensors from safetensors files per model.safetensors.index.json."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Group parameter names by file
    file_to_names = {}
    for name, fname in weight_map.items():
        file_to_names.setdefault(fname, []).append(name)

    weights = []
    for fname in sorted(file_to_names):
        fpath = os.path.join(model_dir, fname)
        logger.info(f"  Loading {fname}")
        with safe_open(fpath, framework="pt", device=device) as f:
            for name in file_to_names[fname]:
                weights.append((name, f.get_tensor(name)))

    logger.info(f"Loaded {len(weights)} tensors from {len(file_to_names)} safetensors files")
    return weights


def bucket_tensors(tensors, max_bytes):
    """Group (name, tensor) pairs into byte-sized buckets.

    Same logic as verl.workers.rollout.sglang_rollout.utils.get_named_tensor_buckets.
    """
    bucket, cur = [], 0
    for name, t in tensors:
        nbytes = t.element_size() * t.numel()
        if cur + nbytes > max_bytes and bucket:
            yield bucket
            bucket, cur = [], 0
        bucket.append((name, t))
        cur += nbytes
    if bucket:
        yield bucket


# ─── Main Test ─────────────────────────────────────────────────────
async def main():
    args = parse_args()
    bucket_bytes = args.bucket_mb << 20
    mp.set_start_method("spawn", force=True)

    # ── Step 1: Init torch.distributed (required by sgl_update_weights for device_mesh)
    logger.info("=" * 60)
    logger.info("Step 1: Initializing torch.distributed")
    logger.info("=" * 60)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    device_mesh = init_device_mesh("cuda", (args.tp_size,), mesh_dim_names=("infer_tp",))
    logger.info(f"Device mesh initialized: {device_mesh}")

    # ── Step 2: Launch SGLang HTTP server (dummy weights + mxfp8 quantization)
    #    This mirrors async_sglang_server.py: SGLangHttpServer.launch_server()
    logger.info("=" * 60)
    logger.info("Step 2: Launching SGLang HTTP server (dummy weights + mxfp8)")
    logger.info("=" * 60)
    engine = AsyncHttpServerAdapter(
        model_path=args.model_path,
        tp_size=args.tp_size,
        quantization="mxfp8",
        json_model_override_args=json.dumps({"quantization_config": MXFP8_QUANT_CONFIG}),
        load_format="dummy",
        mem_fraction_static=args.mem_fraction,
        trust_remote_code=True,
        enable_memory_saver=True,
        log_level="info",
        host="127.0.0.1",
        port=args.port,
        launch_server=True,
        first_rank_in_node=True,
        fp8_gemm_runner_backend="triton",
        moe_runner_backend="cutlass",
    )
    logger.info(f"SGLang server running at http://127.0.0.1:{args.port}")

    # ── Step 3: Load pre-quantized MXFP8 weights from safetensors
    logger.info("=" * 60)
    logger.info("Step 3: Loading MXFP8 weights from safetensors (to CPU)")
    logger.info("=" * 60)
    weights = load_safetensors_weights(args.model_path, device="cpu")

    # ── Step 4: Sync weights via sgl_update_weights
    #    Same call path as sglang_rollout.py: ServerAdapter.update_weights()
    logger.info("=" * 60)
    logger.info("Step 4: Syncing MXFP8 weights to SGLang server")
    logger.info("=" * 60)

    # Use iterator with look-ahead to detect the last bucket without loading all into memory
    bucket_iter = bucket_tensors(weights, bucket_bytes)
    
    try:
        current_batch = next(bucket_iter)
        bucket_idx = 0
        
        while True:
            try:
                # Try to peek the next batch
                next_batch = next(bucket_iter)
                # If we got next_batch, current_batch is not the last one
                is_last_bucket = False
                
                # Move batch to CUDA (sgl_update_weights expects GPU tensors, matching training flow)
                cuda_batch = [(name, t.cuda()) for name, t in current_batch]
                
                logger.info(
                    f"  Bucket {bucket_idx}: {len(cuda_batch)} tensors, "
                    f"first={cuda_batch[0][0]}, last={cuda_batch[-1][0]}"
                )
                
                await sgl_update_weights(
                    engine=engine,
                    params_batch=cuda_batch,
                    device_mesh_key="infer_tp",
                    device_mesh=device_mesh,
                    run_post_process=is_last_bucket,
                )
                
                del cuda_batch
                torch.cuda.empty_cache()
                
                # Move to next batch
                current_batch = next_batch
                bucket_idx += 1
                
            except StopIteration:
                # No more batches after current_batch, so it's the last one
                is_last_bucket = True
                
                # Move batch to CUDA
                cuda_batch = [(name, t.cuda()) for name, t in current_batch]
                
                logger.info(
                    f"  Bucket {bucket_idx}: {len(cuda_batch)} tensors, "
                    f"first={cuda_batch[0][0]}, last={cuda_batch[-1][0]} (last bucket)"
                )
                
                await sgl_update_weights(
                    engine=engine,
                    params_batch=cuda_batch,
                    device_mesh_key="infer_tp",
                    device_mesh=device_mesh,
                    # Trigger post-processing (e.g. MxFP8 MoE scale swizzle) on the last bucket
                    run_post_process=is_last_bucket,
                )
                
                del cuda_batch
                torch.cuda.empty_cache()
                break
                
    except StopIteration:
        # Empty iterator, no buckets to process
        logger.warning("No weight buckets to sync")

    del weights
    logger.info("Weight sync complete!")

    # Flush KV cache after weight update (same as sglang_rollout.py line 233)
    await engine.flush_cache()
    logger.info("Cache flushed")

    # ── Step 5: Test inference
    logger.info("=" * 60)
    logger.info("Step 5: Testing inference with synced MXFP8 weights")
    logger.info("=" * 60)
    test_prompts = [
        "Hello, my name is",
        "What is 2 + 3? Answer:",
    ]
    sampling_params = {"temperature": 0.7, "max_new_tokens": 128, "top_p": 0.9}

    for prompt in test_prompts:
        result = await engine.generate(prompt=prompt, sampling_params=sampling_params)
        text = result.get("text", "<empty>")
        logger.info(f"\n  Prompt : {prompt}\n  Output : {text}")

    # ── Cleanup
    logger.info("=" * 60)
    logger.info("Cleanup")
    logger.info("=" * 60)
    engine.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info("Test PASSED!")


if __name__ == "__main__":
    asyncio.run(main())