from configs.lora_config import config as lora_config
from transformers import BitsAndBytesConfig
import torch 

# PPO training constants
PER_DEVICE_BATCH_SIZE_POLICY = 2
GRADIENT_ACCUMULATION_STEPS = 2
MAX_LENGTH_POLICY = 1024
NUM_EPOCHS_POLICY = 3
LR_POLICY = 5e-4
LR_VALUE = 5e-4
PROMPT_DATASET = "HumanDynamics/ppo_dataset"
CHECKPOINT_PATH_PPO = './ppo_ckpt/'
SAVE_CKPT_STEPS = 1000
BETA = 0.001  # LR-penalty factor in rewards
EPSILON = 0.2  # PPO clip factor
GAMMA = 0.99
EVAL_STEPS = 100  # Print eval every k steps
EVAL_SENTENCES = [
    "<s>[INST] <<SYS>>You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.<</SYS>> ### Instruction: Premise: The little girl holds up a yellow balloon animal. Hypothesis: A small child has a balloon. .Given the premise, can we conclude the hypothesis? [/INST] ### Response: ",
    "<s>[INST] <<SYS>>You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer. <</SYS>>### Instruction: Piye Which language is this? [/INST] ### Response: ",
    "<s>[INST] <<SYS>>You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<</SYS>> ### Instruction: What are some methods you can use to combat post-conference fatigue?  [/INST] ### Response: "
]

# BitsAndBytes config
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
