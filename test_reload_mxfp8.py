"""
Test script for MXFP8 reload functionality.

This script demonstrates:
1. First load bf16 parameters and online quantize to mxfp8
2. Perform inference (first time)
3. Reload bf16 parameters using model.load_weights
4. Online quantize to mxfp8 again
5. Perform inference (second time)
"""
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from verl.utils.vllm.vllm_mxfp8_utils import apply_vllm_mxfp8_patches
apply_vllm_mxfp8_patches()

import torch
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams


def get_bf16_weights_from_hf(model_name: str, trust_remote_code: bool = True):
    """
    Load bf16 weights from HuggingFace model.
    Returns an iterator of (name, tensor) pairs.
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


def reload_weights_with_mxfp8_quant(llm, weights):
    """
    Reload bf16 weights into vLLM model with MXFP8 online quantization.
    
    This function:
    1. Gets the vLLM model and model_runner
    2. Quantizes bf16 weights to MXFP8 format
    3. Loads the quantized weights into the model
    """

    try:
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    except AttributeError:
        model_runner = llm.llm_engine.engine_core.model_executor.driver_worker.model_runner
    model = model_runner.model
    model.load_weights(weights)
    for name, param in model.named_parameters():
        if "layers.0" in name and "weight" in name:
            print(f"After reload: Name: {name}, dtype: {param.dtype}, data: {param[:4]}")
    
    return weights


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
    )
    
    # Print some model info
    print("\n" + "=" * 80)
    print("Model Weights (First Layer Sample):")
    print("=" * 80)
    try:
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    except AttributeError:
        model_runner = llm.llm_engine.engine_core.model_executor.driver_worker.model_runner
    
    model = model_runner.model
    for name, param in model.named_parameters():
        if "layers.0" in name and "weight" in name:
            print(f"Name: {name}, dtype: {param.dtype}, data: {param[:4]}")
    
    # Setup inference
    prompt = "Hello, my name is"
    prompts = [prompt]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    
    # First inference
    print("\n" + "=" * 80)
    print("Step 2: First Inference (After Initial MXFP8 Quantization)")
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
    
    # Reload weights with MXFP8 quantization
    print("\n" + "=" * 80)
    print("Step 4: Reload bf16 Weights with MXFP8 Online Quantization")
    print("=" * 80)
    reload_weights_with_mxfp8_quant(llm, bf16_weights)
    
    # Second inference after reload
    print("\n" + "=" * 80)
    print("Step 5: Second Inference (After Weight Reload)")
    print("=" * 80)
    outputs = llm.generate(prompts, sampling_params)
    
    second_output = outputs[0].outputs[0].text
    print(f"Input: {prompt}")
    print(f"Output: {second_output}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
