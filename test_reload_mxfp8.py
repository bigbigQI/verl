"""
Test script for MXFP8 reload functionality.

This script demonstrates:
1. First load bf16 parameters and online quantize to mxfp8
2. Perform inference (first time)
3. Reload bf16 parameters using vLLMColocateWorkerExtension pattern
4. Online quantize to mxfp8 again
5. Perform inference (second time)

Key changes:
- Uses the vLLMColocateWorkerExtension pattern for weight updates
- Implements bucket-based weight transfer similar to IPC mode
- Supports MXFP8 online quantization through vLLM's load_weights
"""
import gc
import os
from typing import Generator, TypedDict

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams


class TensorMetadata(TypedDict):
    """Metadata for a tensor in a weight bucket."""
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class MXFP8WeightUpdater:
    """
    Weight updater class that mimics vLLMColocateWorkerExtension pattern.
    
    This class implements the bucket-based weight update mechanism used in verl
    for synchronizing weights between training and inference processes.
    
    Key features:
    1. Bucket-based weight batching for memory efficiency
    2. Support for MXFP8 online quantization via vLLM's load_weights
    3. Proper memory management with explicit cleanup
    """
    
    def __init__(self, llm: LLM, bucket_size_mb: int = 512):
        """
        Initialize the weight updater.
        
        Args:
            llm: The vLLM LLM instance
            bucket_size_mb: Size of weight bucket in megabytes (default 512MB)
        """
        self.llm = llm
        self.bucket_size = bucket_size_mb << 20  # Convert MB to bytes
        self._init_model_runner()
        
    def _init_model_runner(self):
        """Initialize model runner reference from LLM engine."""
        try:
            self.model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        except AttributeError:
            self.model_runner = self.llm.llm_engine.engine_core.model_executor.driver_worker.model_runner
        self.model = self.model_runner.model
        self.device = next(self.model.parameters()).device
        
    def _get_named_tensor_buckets(
        self, 
        weights: list[tuple[str, torch.Tensor]], 
        target_dtype: torch.dtype = torch.bfloat16
    ) -> Generator[list[tuple[str, torch.Tensor]], None, None]:
        """
        Yield buckets of weights that fit within the bucket size limit.
        
        This mimics the bucket-based transfer used in vLLM IPC mode.
        
        Args:
            weights: List of (name, tensor) pairs
            target_dtype: Target dtype for weight conversion
            
        Yields:
            List of (name, tensor) pairs that fit in one bucket
        """
        current_bucket: list[tuple[str, torch.Tensor]] = []
        current_size = 0
        
        for name, weight in weights:
            # Convert weight to target dtype
            weight = weight.to(target_dtype)
            weight_size = weight.nbytes
            
            # Check if single weight exceeds bucket size
            if weight_size > self.bucket_size:
                # Yield current bucket first if not empty
                if current_bucket:
                    yield current_bucket
                    current_bucket = []
                    current_size = 0
                # Yield the large weight as its own bucket
                print(f"Warning: Weight {name} ({weight_size / 1e6:.2f}MB) exceeds bucket size")
                yield [(name, weight)]
                continue
                
            # Check if adding this weight would exceed bucket size
            if current_size + weight_size > self.bucket_size:
                yield current_bucket
                current_bucket = []
                current_size = 0
                
            current_bucket.append((name, weight))
            current_size += weight_size
            
        # Yield remaining weights
        if current_bucket:
            yield current_bucket
    
    def _update_weights_bucket(self, weights: list[tuple[str, torch.Tensor]]):
        """
        Update model weights from a single bucket.
        
        This is the core update logic from vLLMColocateWorkerExtension._update_weights,
        adapted for MXFP8 models.
        
        For MXFP8:
        - vLLM's load_weights handles the online quantization internally
        - The process_weights_after_loading hook in vLLM converts bf16 -> mxfp8
        
        Args:
            weights: List of (name, tensor) pairs to load
        """
        # Move weights to GPU if needed
        gpu_weights = []
        for name, weight in weights:
            if weight.device != self.device:
                weight = weight.to(self.device, non_blocking=True)
            gpu_weights.append((name, weight))
        
        # Synchronize to ensure all transfers complete
        torch.cuda.synchronize()
        
        # Load weights using vLLM's load_weights method
        # For MXFP8, vLLM internally handles the quantization
        self.model.load_weights(gpu_weights)
        
    def update_weights(
        self, 
        weights: list[tuple[str, torch.Tensor]],
        target_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Update model weights using bucket-based transfer pattern.
        
        This method implements the full update_weights_from_ipc flow from
        vLLMColocateWorkerExtension, but without the IPC/ZMQ communication
        since we're in the same process.
        
        Args:
            weights: List of (name, tensor) pairs
            target_dtype: Target dtype for weight conversion (default bf16)
        """
        print(f"Updating weights with bucket size: {self.bucket_size / 1e6:.2f}MB")
        
        bucket_count = 0
        total_weights = 0
        
        for bucket in self._get_named_tensor_buckets(weights, target_dtype):
            bucket_count += 1
            total_weights += len(bucket)
            print(f"  Processing bucket {bucket_count}: {len(bucket)} weights")
            
            self._update_weights_bucket(bucket)
            
            # Clean up bucket memory
            del bucket
            
        # Final cleanup similar to vLLMColocateWorkerExtension
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        print(f"Weight update complete: {total_weights} weights in {bucket_count} buckets")
        
    def print_sample_weights(self, prefix: str = ""):
        """Print sample weights from first layer for debugging."""
        for name, param in self.model.named_parameters():
            if "layers.0" in name and "weight" in name:
                print(f"{prefix}Name: {name}, dtype: {param.dtype}, data: {param.data.flatten()[:4]}")


def get_bf16_weights_from_hf(model_name: str, trust_remote_code: bool = True):
    """
    Load bf16 weights from HuggingFace model.
    Returns a list of (name, tensor) pairs.
    """
    print(f"Loading bf16 weights from HuggingFace model: {model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="cpu",  # Load to CPU first to avoid GPU memory issues
    )
    
    # Collect all weights
    weights = []
    for name, param in hf_model.named_parameters():
        weights.append((name, param.data.clone()))
    
    # Clean up HuggingFace model to free memory
    del hf_model
    torch.cuda.empty_cache()
    
    print(f"Loaded {len(weights)} weights from HuggingFace model")
    return weights


def reload_weights_with_mxfp8_quant(llm: LLM, weights: list[tuple[str, torch.Tensor]]):
    """
    Reload bf16 weights into vLLM model with MXFP8 online quantization.
    
    Uses the vLLMColocateWorkerExtension pattern:
    1. Creates MXFP8WeightUpdater (similar to extension class)
    2. Updates weights using bucket-based transfer
    3. vLLM internally handles MXFP8 quantization
    
    Args:
        llm: The vLLM LLM instance
        weights: List of (name, tensor) pairs in bf16 format
    """
    # Create weight updater (similar to vLLMColocateWorkerExtension)
    updater = MXFP8WeightUpdater(llm, bucket_size_mb=512)
    
    print("Before reload:")
    updater.print_sample_weights(prefix="  ")
    
    # Update weights using bucket-based pattern
    updater.update_weights(weights, target_dtype=torch.bfloat16)
    
    print("After reload:")
    updater.print_sample_weights(prefix="  ")


def is_mxfp8_model(vllm_config) -> bool:
    """
    Check if the model is using MXFP8 quantization.
    
    Args:
        vllm_config: vLLM configuration object
        
    Returns:
        True if the model uses MXFP8 quantization
    """
    try:
        from vllm.model_executor.layers.quantization.mxfp8 import MXFP8Config
        if hasattr(vllm_config, "quant_config") and isinstance(vllm_config.quant_config, MXFP8Config):
            return True
    except ImportError:
        pass
    return False


def main():
    model_name = "Qwen/Qwen3-8B-Base"
    
    print("=" * 80)
    print("Step 1: Initialize vLLM with MXFP8 quantization")
    print("=" * 80)
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        quantization="mxfp8",
        enable_sleep_mode=True,  # Enable sleep mode for testing
        load_format="dummy",
    )
    
    # Print model info and verify MXFP8 quantization
    print("\n" + "=" * 80)
    print("Model Info:")
    print("=" * 80)
    try:
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    except AttributeError:
        model_runner = llm.llm_engine.engine_core.model_executor.driver_worker.model_runner
    
    print(f"Is MXFP8 model: {is_mxfp8_model(model_runner.vllm_config)}")
    print(f"Quant config: {model_runner.vllm_config.quant_config}")
    
    # Print sample weights (first layer)
    print("\nModel Weights (First Layer Sample):")
    model = model_runner.model
    for name, param in model.named_parameters():
        if "layers.0" in name and "weight" in name:
            print(f"  Name: {name}, dtype: {param.dtype}, data: {param.data.flatten()[:4]}")
    
    # Setup inference
    prompt = "Hello, my name is"
    prompts = [prompt]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    
    # First inference (with dummy weights)
    print("\n" + "=" * 80)
    print("Step 2: First Inference (With Dummy Weights)")
    print("=" * 80)
    outputs = llm.generate(prompts, sampling_params)
    
    first_output = outputs[0].outputs[0].text
    print(f"Input: {prompt}")
    print(f"Output: {first_output}")
    
    # Load bf16 weights from HuggingFace
    print("\n" + "=" * 80)
    print("Step 3: Load bf16 Weights from HuggingFace")
    print("=" * 80)
    bf16_weights = get_bf16_weights_from_hf(model_name)
    
    # Reload weights with MXFP8 quantization using extension pattern
    print("\n" + "=" * 80)
    print("Step 4: Reload bf16 Weights with MXFP8 Online Quantization")
    print("       (Using vLLMColocateWorkerExtension pattern)")
    print("=" * 80)
    reload_weights_with_mxfp8_quant(llm, bf16_weights)
    
    # Clean up bf16 weights to free memory
    del bf16_weights
    gc.collect()
    torch.cuda.empty_cache()
    
    # Second inference after reload
    print("\n" + "=" * 80)
    print("Step 5: Second Inference (After Weight Reload)")
    print("=" * 80)
    outputs = llm.generate(prompts, sampling_params)
    
    second_output = outputs[0].outputs[0].text
    print(f"Input: {prompt}")
    print(f"Output: {second_output}")
    
    # Compare outputs
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"First output (dummy weights):  {first_output[:50]}...")
    print(f"Second output (real weights):  {second_output[:50]}...")
    outputs_match = first_output == second_output
    print(f"Outputs match: {outputs_match}")
    
    if not outputs_match:
        print("SUCCESS: Outputs differ, indicating weights were successfully reloaded!")
    else:
        print("WARNING: Outputs are identical - weights may not have been properly updated")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
