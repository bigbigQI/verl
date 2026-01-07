from vllm import LLM, SamplingParams

# 初始化LLM，加载ModelOpt量化的模型
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# 应用 ModelOpt patch
from verl.utils.modelopt_utils import apply_vllm_modelopt_patches
apply_vllm_modelopt_patches()
print("=" * 80)
print("Applied vLLM ModelOpt patches")
print("=" * 80)

quant_config = {
        "config_groups": {
            "group_0": {
                "weights": {
                    "dynamic": "false",
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16
                },
                "targets": [
                    "Linear"
                ]
            }
        },
        "ignore": [
            "lm_head"
        ],
        "quant_algo": "NVFP4",
        "producer": {
            "name": "modelopt",
            "version": "0.4"
        },
        "quant_method": "modelopt"
    }


llm = LLM(
    # model="/apps/quant_models/Qwen3-8B-Base-nvfp4-default-blockwise-qat—mlp-woq-32/hf_checkpoint",  # ModelOpt量化后的模型
    # model="/apps/TensorRT-Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-30B-A3B-Base_nvfp4",
    model="/apps/quant_models/qwen3_30b_nvfp4",
    # model="/apps/models/Qwen3-30B-A3B-NVFP4",
    # model="/apps/models/Qwen3-8B-Base-scaled", 
    quantization="modelopt",  # 指定使用modelopt量化
    trust_remote_code=True,
    load_format="pt",
    # load_format="dummy",
    # hf_overrides={"quantization_config": quant_config},
    enable_sleep_mode=True,  # 启用 sleep mode 用于测试
)

# 打印每个weight的name和shape
# print("=" * 80)
# print("Model Weights:")
# print("=" * 80)
# model = llm.llm_engine.model_executor.driver_worker.model_runner.model
# for name, param in model.named_parameters():
#     if "layers.0" in name:
#         print(f"Name: {name}, Shape: {param.shape}, dtype: {param.dtype}")
# print("=" * 80)


# prompt = """
# user
# Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.
# \( f \) is a function whose domain is the set of nonnegative integers and whose range is contained in the set of nonnegative integers. \( f \) satisfies the condition that \( f(f(n)) + f(n) = 2n + 3 \) for all nonnegative integers \( n \). Find \( f(2014) \).

# Remember to put your answer on its own line after "Answer:".
# assistant
# """

prompt = "Hello, my name is"

prompts = [prompt]
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

# 第一次推理
print("=" * 80)
print("第一次推理 (Before Sleep)")
print("=" * 80)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output)

# 测试 sleep mode - 卸载权重
print("\n" + "=" * 80)
print("进入 Sleep Mode - 卸载权重")
print("=" * 80)
import asyncio
asyncio.run(llm.llm_engine.sleep(level=1))
print("Sleep mode 完成")

# 测试 wake_up mode - 重新加载权重
print("\n" + "=" * 80)
print("进入 Wake Up Mode - 重新加载权重")
print("=" * 80)
asyncio.run(llm.llm_engine.wake_up(tags=["weights", "kv_cache"]))
print("Wake up 完成")

# 第二次推理 - 验证 patch 是否仍然生效
print("\n" + "=" * 80)
print("第二次推理 (After Wake Up) - 验证 patch 是否生效")
print("=" * 80)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output)

print("\n" + "=" * 80)
print("测试完成！如果两次推理都成功，说明 patch 正确生效")
print("=" * 80)