from configs.reward_config import lora_config, RLHF_DATASET
from data.data_loader import *
from models.reward_trainer import train_reward_model

if __name__ == "__main__":
    dataset = get_reward_data(RLHF_DATASET)
    train_reward_model(
        lora_config,
        dataset
    )