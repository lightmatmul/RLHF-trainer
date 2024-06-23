# RLHF Training for LLMs

This repository contains implementations for Reinforcement Learning with Human Feedback (RLHF) training of Large Language Models (LLMs) using Supervised Fine-Tuning (SFT), Reward Modeling, and Proximal Policy Optimization (PPO). The goal is to create a modular and maintainable codebase for replicating RLHF training on LLMs like LLaMA.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Supervised Fine-Tuning](#supervised-fine-tuning)
  - [Reward Modeling](#reward-modeling)
  - [Proximal Policy Optimization](#proximal-policy-optimization)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

## Project Structure

rlhf_training/
│
├── configs/
│ ├── init.py
│ ├── lora_config.py
│ ├── reward_config.py
│ └── ppo_config.py
│
├── data/
│ ├── init.py
│ ├── data_loader.py
│ ├── reward_data_loader.py
│ └── ppo_data_loader.py
│
├── models/
│ ├── init.py
│ ├── model_utils.py
│ ├── sft_trainer.py
│ ├── reward_trainer.py
│ └── ppo_trainer.py
│
├── utils/
│ ├── init.py
│ └── training_utils.py
│
├── scripts/
│ ├── train_sft.py
│ ├── train_reward.py
│ └── train_ppo.py
│
├── requirements.txt
└── README.md


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rlhf_training.git
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
python scripts/train_sft.py```

### Reward Modeling

To train a reward model, run the following script:
```bash
python scripts/train_reward.py```






