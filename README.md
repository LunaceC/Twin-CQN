# Continuous Control with Coarse-to-fine Reinforcement Learning (CQN) and Twin-CQN

This is a codebase for HKUST-gz course IOTA5201 final project (2025 fall).

This repository is a fork of the original **Coarse-to-fine Q-Network (CQN)** implementation for continuous control, extended with a **Twin-CQN** variant for RLBench experiments.

The original algorithm is introduced in:

[**Continuous Control with Coarse-to-fine Reinforcement Learning**](https://younggyo.me/cqn/)

[Younggyo Seo](https://younggyo.me/), [Jafar Uruç](https://github.com/JafarAbdi), [Stephen James](https://stepjam.github.io/)

The core idea is to learn RL agents that zoom into continuous action space in a coarse-to-fine manner, allowing value-based RL with a small discrete branching factor at each refinement level.

This fork was used for a course project to:

- **Reproduce RLBench results** with the original CQN agent.
- **Introduce Twin-CQN**, a TD3-style variant with two critics and a pessimistic `min(Q1, Q2)` target, aiming to reduce late-stage value overestimation and instability in RLBench tasks.
- **Add a demo filter** in the RLBench environment wrapper to automatically skip obviously bad demonstrations with large action jumps.
- **Provide a small test script** that runs both the baseline CQN agent and the Twin-CQN agent on an RLBench task as a quick smoke test.

The original project webpage: <https://younggyo.me/cqn/>

![gif1](media/cqn_gif1.gif)
![gif2](media/cqn_gif2.gif)

---

## 1. Repository structure (fork overview)

At a high level:

- `train_rlbench.py`  
  Main entry point for RLBench experiments (kept as close as possible to the original; extended only to support Twin-CQN and demo filtering hooks where needed).

- `cqn.py`  
  Defines the agents used in RLBench:
  - `CQNAgent`: original coarse-to-fine Q-network agent.
  - `TwinCQNAgent`: Twin-CQN agent with two critics and a `min` backup.

- `config_rlbench.yaml`  
  Hydra configuration file for RLBench experiments. Defaults correspond to the original CQN setup, with additional options for:
  - Selecting the agent implementation (`CQNAgent` vs `TwinCQNAgent`).
  - Controlling RLBench-related options, including the demo filtering behavior.

- `rlbench_env.py` (or RLBench environment wrapper used by this repo)  
  Contains a **demo filtering hook** that checks action deltas in demonstrations and skips trajectories with excessively large changes. During data loading, you will see log messages such as:
  `Skipping demo 17 for large delta action at step 116 ...`.  
  This is intended to remove corrupted or unstable demos before training.

- `scripts/run_examples.sh`  
  Example runtime script (added in this fork) that demonstrates one baseline CQN run and one Twin-CQN run on an RLBench task. Paths, seeds, and tasks can be edited as needed.

- `train_rlbench_drqv2plus.py`, `train_dmc.py`, etc.  
  Other training scripts from the original repository are retained for completeness. They are not modified for Twin-CQN and are not required for reproducing the Twin-CQN results.

---

## 2. Environment setup

### 2.1 Conda environment

Create and activate the conda environment:


```bash
conda env create -f conda_env.yml
conda activate cqn
```

    # IMPORTANT: Install torch/torchvision AFTER creating the env,
    # using the official command from https://pytorch.org/get-started/locally/
    # so that you get a wheel that supports your specific GPU (e.g. RTX 50-series).
    #
    # Example (you must adapt cuXXX according to PyTorch's website):
    #   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX


### 2.2 RLBench and PyRep installation

Install RLBench and PyRep. The original implementation uses specific commits; we keep the same versions here. Follow the installation instructions from the RLBench and PyRep repositories, including headless mode if needed.

```bash
git clone https://github.com/stepjam/RLBench
git clone https://github.com/stepjam/PyRep

# Install PyRep
cd PyRep
git checkout 8f420be8064b1970aae18a9cfbc978dfb15747ef
pip install .

# Install RLBench
cd ../RLBench
git checkout b80e51feb3694d9959cb8c0408cd385001b01382
pip install .

# Go back to Twin-CQN repo root
cd ..
```



Refer to the RLBench and Robobase READMEs for additional details:

- RLBench: <https://github.com/stepjam/RLBench>  
- Robobase RLBench docs: <https://github.com/robobase-org/robobase>

---

## 3. RLBench experiments

### 3.1 Pre-collect demonstrations

Before training, collect demonstrations for the target tasks using RLBench’s `dataset_generator.py`. Example (for one task):

```bash
# From Twin-CQN repository root
mkdir -p rlbench_demos
export DEMO_ROOT="$(pwd)/rlbench_demos"

cd RLBench/rlbench

#collect extra demos in case filtered by velocity limits. The real number of demos used is 100 as hardcoded in rlbench_env.py
CUDA_VISIBLE_DEVICES=0 python dataset_generator.py   --save_path="${DEMO_ROOT}"   --image_size 84 84   --renderer opengl3   --episodes_per_task 120   --variations 1   --processes 1   --tasks take_lid_off_saucepan   --arm_max_velocity=2.0   --arm_max_acceleration=8.0

# Go back to Twin-CQN repo root
cd ../..
```

In this fork, the same demo dataset can be used for both the baseline CQN agent and the Twin-CQN agent.

---

### 3.2 Baseline CQN agent (original)

To run RLBench experiments with the original CQN agent:

```bash
CUDA_VISIBLE_DEVICES=0 python train_rlbench.py   rlbench_task=take_lid_off_saucepan   num_demos=120   dataset_root="${DEMO_ROOT}"   agent._target_=cqn.CQNAgent
```

Notes:

- `rlbench_task` should match the RLBench task name used when generating demos.
- `agent._target_` selects the agent class used by Hydra. `cqn.CQNAgent` is the original baseline from the CQN paper.

---

### 3.3 Twin-CQN agent (this fork)

The Twin-CQN variant uses two critics and a pessimistic target.

All other training details (encoder architecture, auxiliary BC loss, demonstrations, and training budget) are kept consistent with the baseline CQN setup for fair comparison.

To run RLBench experiments with the Twin-CQN agent:

```bash
CUDA_VISIBLE_DEVICES=0  python train_rlbench.py   rlbench_task=take_lid_off_saucepan   num_demos=120   dataset_root="${DEMO_ROOT}"    agent._target_=cqn.TwinCQNAgent
```

You can substitute any supported RLBench task (e.g., `open_door`, `turn_on_lightbulb`, `open_oven`, `take_lid_off_saucepan`, `press_switch`, `turn_tap`) as long as you have corresponding demonstrations under `dataset_root`.

---

### 3.4 Demo filtering in RLBench environment

This fork introduces a **demo filter** in the RLBench environment wrapper used by `train_rlbench.py`. The goal is to remove demonstrations that contain unusually large action jumps, which often correspond to unstable trajectories or collisions in the original demos.

When demos are loaded, you may see log lines such as:

```text
Skipping demo 17 for large delta action at step 116 (max=..., min=...)
[RLBench] get_demos: requested 120, received 120 from task, kept 117, skipped 3 demos with indices [...]
```

Key points:

- Filtering is applied at load time and only affects which demonstrations are added to the replay buffer.
- The logic is local to this fork and does not require changes to the upstream RLBench repository.
- If needed, you can adjust thresholds or disable the filter by editing the RLBench environment wrapper used by this repo (see the `rlbench`-related module in this repository and `config_rlbench.yaml` for any filtering parameters).

---


## 4. DMC experiments

The original repository also includes support for DeepMind Control Suite (DMC) tasks. That code path is retained but **not modified** for Twin-CQN in this fork.

To run DMC experiments with the original CQN agent:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dmc.py dmc_task=cartpole_swingup
```

Warning: as in the original README, CQN is not extensively tested in DMC.

---

## 5. Weights & Biases logging (optional)

If desired, you can enable Weights & Biases logging:

1. Set `wandb` configuration in `config_rlbench.yaml` (project name, entity, etc.), or
2. Pass overrides on the command line, for example:

   ```bash
   python train_rlbench.py      rlbench_task=open_oven      num_demos=120      dataset_root="${DEMO_ROOT}"      agent._target_=cqn.TwinCQNAgent      wandb.project=Twin-CQN_RLBench      wandb.entity=your_wandb_entity
   ```

By default, the repo is configured so that logging can be turned on or off without changing the core training code.

There is pre-conditioned tensorboard integration that could be turned on in config_rlbench.yaml
---

## 6. Acknowledgements

- The original CQN implementation and RLBench integration are by **Seo et al.**; this fork only adds a Twin-CQN agent, demo filtering, and minor configuration/runtime changes.
- The codebase builds upon the public implementation of [DrQ-v2](https://github.com/facebookresearch/drqv2), as in the original repository.

---

## 7. Citation

If you use this repository for research, please cite the original CQN paper:

```bibtex
@article{seo2024continuous,
  title={Continuous Control with Coarse-to-fine Reinforcement Learning},
  author={Seo, Younggyo and Uru{\c{c}}, Jafar and James, Stephen},
  journal={arXiv preprint arXiv:2407.07787},
  year={2024}
}
```

If you build on the Twin-CQN variant introduced in this fork, please additionally reference the corresponding project report or documentation describing the Twin-CQN design and evaluation.
