# Reward training constants
PER_DEVICE_BATCH_SIZE_REWARD = 4
GRADIENT_ACCUMULATION_STEPS = 2 # gradient_accumulation_steps = desired_batch_size / PER_DEVICE_BATCH_SIZE
MAX_LENGTH_REWARD = 1024 
NUM_EPOCHS_REWARD = 1
SAVE_CKPT_STEPS = 46
LR_REWARD = 5e-5
CHECKPOINT_PATH_REWARD = './reward_ckpt/'
RLHF_DATASET = "HumanDynamics/reward_modeling_dataset"
