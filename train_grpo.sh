#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="/mnt/data/zxj/trl"

MODEL_NAME_OR_PATH="/mnt/data/zxj/checkpoints/Qwen2.5-Math-7B"
OUTPUT_DIR="/mnt/data/zxj/result/RL-model/Qwen2.5-Math-7B-DSR-14"

DATASET_NAME="./train_data/deepscaler_hard-1k.json"

DATASET_TRAIN_SPLIT="train"
DATASET_TEST_SPLIT="test"

mkdir -p "$OUTPUT_DIR"
REWARD_FUNCS="box_format_reward, box_accuracy_reward"

# --- Run the Python Script ---
accelerate launch --config_file "./trl_edge/accelerate_configs/zero3.yaml" ./trl_edge/scripts/grpo.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --dataset_train_split "$DATASET_TRAIN_SPLIT" \
    --dataset_test_split "$DATASET_TEST_SPLIT" \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --report_to "tensorboard" \
    --reward_funcs "$REWARD_FUNCS" \
    --beta 0.0 \
    --force_reflection True \
    --GEC True \
    --EDA True \
