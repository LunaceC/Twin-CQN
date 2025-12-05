#!/bin/bash
set -e

# GPU selection
export CUDA_VISIBLE_DEVICES=0
# Kept for completeness; TwinCQNAgent may ignore it, but it does no harm.
export TARGET_POLICY_SMOOTHING=0

# Root directory where your RLBench demos are stored.
# CHANGE THIS to your actual path before running.
DATA_ROOT=/home/your_name/Twin-CQN/rlbench_demos

# Optional: Weights & Biases configuration.
# Uncomment and set these if you want to enable wandb logging.
# export WANDB_PROJECT=Twin-CQN_RLBench
# export WANDB_ENTITY=your_wandb_entity_name

echo "======================================================"
echo " Example 1: Baseline CQNAgent on take_lid_off_saucepan (3-step)  "
echo "======================================================"

python train_rlbench.py \
  --config-name config_rlbench \
  rlbench_task=take_lid_off_saucepan \
  dataset_root="${DATA_ROOT}" \
  num_demos=120 \
  agent._target_=cqn.CQNAgent \
  seed=0
  # If you enable wandb in the config, you can also pass:
  # wandb.project=${WANDB_PROJECT} \
  # wandb.entity=${WANDB_ENTITY} \
  # wandb.name=take_lid_off_saucepan_cqn_3step_seed0

  # num_demos is greater than 100 here as some demos may be filtered out in environment parsing. 
  # the num of demos eventually used is 100 as hardcoded in rlbench_env.py, the rest will be truncated.


echo "======================================================"
echo " Example 2: TwinCQNAgent on take_lid_off_saucepan (3-step)       "
echo "======================================================"

python train_rlbench.py \
  --config-name config_rlbench \
  rlbench_task=take_lid_off_saucepan \
  dataset_root="${DATA_ROOT}" \
  num_demos=120 \
  agent._target_=cqn.TwinCQNAgent \
  seed=0
  # Optional wandb overrides (if enabled in config):
  # wandb.project=${WANDB_PROJECT} \
  # wandb.entity=${WANDB_ENTITY} \
  # wandb.name=take_lid_off_saucepan_twin_cqn_3step_seed0

  # num_demos is greater than 100 here as some demos may be filtered out in environment parsing. 
  # the num of demos eventually used is 100 as hardcoded in rlbench_env.py, the rest will be truncated.