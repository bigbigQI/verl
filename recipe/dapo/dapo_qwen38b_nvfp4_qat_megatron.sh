#!/usr/bin/env bash
set -xeuo pipefail

# Example: DAPO training with NVFP4 QAT using Megatron backend
# This script demonstrates how to enable Quantization-Aware Training

ID=${1:-"dapo-qwen3-8b-nvfp4-qat-megatron-bs32"}
HOME_DIR=/apps

project_name='DAPO-QAT-Test'
exp_name=$ID

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# Rollout Importance Sampling parameters
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
max_response_length=$((1024 * 16))
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=1
train_prompt_bsz=32
gen_prompt_bsz=96
n_resp_per_prompt=16
train_prompt_mini_bsz=32

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env_fp8.yaml"}
echo "RUNTIME_ENV: ${RUNTIME_ENV}"
NNODES=${NNODES:-2}
echo "NNODES: ${NNODES}"

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}"}
MODEL_PATH="/apps/models/Qwen3-8B-Base"
# MODEL_PATH="Qwen/Qwen3-0.6B"
# MODEL_PATH="/apps/quant_models/Qwen3-8B-Base-nvfp4-default-blockwise-qat/checkpoint-32"
# QUANT_MODEL_PATH="/apps/quant_models/Qwen3-8B-Base-nvfp4-default-blockwise-qat/hf_checkpoint/"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k-one.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

export VERL_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export WANDB_API_KEY=149737fd3c4537b349a37aab90b6fff96f385ebc


# export WANDB_API_KEY=149737fd3c4537b349a37aab90b6fff96f385ebc
# export ROCR_VISIBLE_DEVICES=
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

    # actor_rollout_ref.rollout.quantization=nvfp4_qat \
    # +actor_rollout_ref.rollout.prequant_model_path="${QUANT_MODEL_PATH}" \
python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.transformer_impl=local \
    actor_rollout_ref.actor.megatron.quantization=nvfp4_qat \
    actor_rollout_ref.rollout.quantization=nvfp4_qat \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.use_mbridge=False \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 16 )) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=5 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
