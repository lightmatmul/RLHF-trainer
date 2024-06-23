# RLHF Training for LLMs

This repository contains implementations for Reinforcement Learning with Human Feedback (RLHF) training of Large Language Models (LLMs) using Supervised Fine-Tuning (SFT), Reward Modeling, and Proximal Policy Optimization (PPO). The goal is to create a modular and maintainable codebase for replicating RLHF training on LLMs like LLaMA. The following codebase is specific to LLaMa 2, so while the components can work universally, data related components (such as special token formatting) need to be modified to fit other models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Supervised Fine-Tuning](#supervised-fine-tuning)
  - [Reward Modeling](#reward-modeling)
  - [Proximal Policy Optimization](#proximal-policy-optimization)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lightmatmul/rlhf_training.git
   cd rlhf_training```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt```

## Usage

### Supervised Fine-Tuning

To train a model using Supervised Fine-Tuning (SFT), run the following script:

  ```bash
  python scripts/train_sft.py
```

### Reward Modeling

To train a reward model, run the following script:
```bash
python scripts/train_reward.py
```

### Proximal Policy Optimization

To train a model using Proximal Policy Optimization (PPO), run the following script:
```bash
python scripts/train_ppo.py
```

### Configuration

The configuration files are located in the configs/ directory. Here’s a brief description of each:

  1. lora_config.py: Contains the configuration for LoRA (Low-Rank Adaptation).
  2. reward_config.py: Contains the constants and configurations specific to Reward Modeling.
  3. ppo_config.py: Contains the constants and configurations specific to PPO.

### Evaluation

GPT is used as AI evaluator to determine evaluate the impact of the alignment tuning compared to the original supervised finetuned model:
```bash
python eval/gpt_evaluator.py
python eval/count_wins.py
```

### Inference

To interact with the trained models, run the following scriptL:
```bash
python scripts/inference.py
```





