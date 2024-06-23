import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig, AutoTokenizer
from data.data_loader import *
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.training_utils import save_ckpt, print_eval_results
from configs.ppo_config import (
    PER_DEVICE_BATCH_SIZE_POLICY,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_LENGTH_POLICY,
    NUM_EPOCHS_POLICY,
    LR_POLICY,
    LR_VALUE,
    PROMPT_DATASET,
    CHECKPOINT_PATH_PPO,
    SAVE_CKPT_STEPS,
    BETA,
    EPSILON,
    GAMMA,
    EVAL_STEPS,
    EVAL_SENTENCES,
    nf4_config
)

def get_log_y(model, inputs, maxlen):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=maxlen,
    )
    # Get log P (y|x) from scores = log (P(y_1|x)) + log(P(y_2|y_1,x))
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    log_p_y = torch.sum(transition_scores, dim=1)
    # Returns logprobs and outputs
    return log_p_y, outputs

def get_rewards(reward_model, inputs, out_policy):
    # Concatenate inputs x and outputs y and pass to reward model
    concat_inputs = torch.cat((inputs["input_ids"], out_policy.sequences), dim=1)
    reward_logits = reward_model(input_ids=concat_inputs).logits
    rewards = reward_logits.sum(dim=1)
    rewards = torch.sigmoid(rewards)  # scale between [0,1]
    return rewards

def get_values(mode, value_model, inputs, out_policy):
    # Concatenate inputs x and outputs y, pass to value model
    values = []
    if mode == "final":
        concat_inputs = torch.cat(
            (inputs["input_ids"], out_policy.sequences), dim=1
        )
        values = (
            value_model(input_ids=concat_inputs)
            .logits[:, -1]
            .flatten()
        )
    # if mode == initial , then value estimate only on sequence x is computed
    if mode == "initial":
        values = (
            value_model(input_ids=inputs["input_ids"])
            .logits[:, -1]
            .flatten()
        )

    values = torch.sigmoid(values)  # scale between [0,1]
    return values

def generate_output(model, inputs):
    # Generate output tokens 
    return model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        repetition_penalty=1.176,
        temperature=0.7,
        top_p=0.1,
        top_k=40,
        do_sample=True,
    )


def run_eval(policy_model, sft_model, tokenizer, eval_sentences, epoch, step):
    # Side-by-side eval of sft and policy on eval_sentences 
    policy_model.eval()
    sft_model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            eval_sentences,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH_POLICY,
            return_tensors="pt",
        )
        outputs_policy = generate_output(policy_model, inputs)
        outputs_sft = generate_output(sft_model, inputs)

        sentences_policy = tokenizer.batch_decode(outputs_policy, skip_special_tokens=True)
        sentences_baseline = tokenizer.batch_decode(outputs_sft, skip_special_tokens=True)
        print_eval_results(eval_sentences, sentences_policy, sentences_baseline, epoch, step)
    
    policy_model.train()
    sft_model.train()

def train_ppo_model():
    accelerator = Accelerator()
    torch.cuda.empty_cache()

    # Initialize reward model from Reward model.
    reward_model = LlamaForCausalLM.from_pretrained(
        "HumanDynamics/reward_model",
        quantization_config=nf4_config,
        device_map="auto"
    )

    # SFT model used as a baseline
    sft_model = LlamaForCausalLM.from_pretrained(
        "HumanDynamics/sft_model",
        quantization_config=nf4_config,
        device_map="auto"
    )

    # Initialize policy model from sft_model
    policy_model = copy.deepcopy(sft_model)

    # Initialize Value model from Reward model.
    value_model = copy.deepcopy(reward_model)

    reward_model, sft_model, policy_model, value_model = accelerator.prepare(
        reward_model, sft_model, policy_model, value_model
    )

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt_dataset = get_prompt_data(PROMPT_DATASET)
    raw_dataloader = DataLoader(prompt_dataset["train"], batch_size=PER_DEVICE_BATCH_SIZE_POLICY)
    raw_dataloader = accelerator.prepare(raw_dataloader)

    value_model = value_model.train()
    for param in value_model.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = True

    policy_model = policy_model.train()
    for param in policy_model.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = True

    reward_model.train()
    for param in reward_model.parameters():
        param.requires_grad = False
    
    sft_model.train()
    for param in sft_model.parameters():
        param.requires_grad = False

    prev_policy = copy.deepcopy(policy_model)
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=LR_POLICY)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=LR_VALUE)
    optimizer_policy, optimizer_value = accelerator.prepare(optimizer_policy, optimizer_value)
    scheduler_policy = CosineAnnealingLR(optimizer_policy, T_max=NUM_EPOCHS_POLICY)
    scheduler_value = CosineAnnealingLR(optimizer_value, T_max=NUM_EPOCHS_POLICY)

    start_time = time.time()
    total_steps = len(raw_dataloader) * NUM_EPOCHS_POLICY // GRADIENT_ACCUMULATION_STEPS
    grad_accumulation_counter = 0
    current_step = 0

    for epoch in range(NUM_EPOCHS_POLICY):
        for i, batch in enumerate(raw_dataloader):
            if i % EVAL_STEPS == 0:
                run_eval(policy_model, sft_model, tokenizer, EVAL_SENTENCES, epoch, i)

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()

            inputs = tokenizer(
                batch["prompt"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH_POLICY,
                return_tensors="pt",
            )

            log_y_policy, out_policy = get_log_y(policy_model, inputs, MAX_LENGTH_POLICY)
            log_y_sft, out_sft = get_log_y(sft_model, inputs, MAX_LENGTH_POLICY)
            log_y_prev_policy, out_prev_policy = get_log_y(prev_policy, inputs, MAX_LENGTH_POLICY)

            rewards_raw = get_rewards(reward_model, inputs, out_policy)
            reward_penalty = BETA * (log_y_policy - log_y_sft)
            rewards = rewards_raw - reward_penalty

            values = get_values("final", value_model, inputs, out_policy)
            initial_values = get_values("initial", value_model, inputs, out_policy)

            advantages = rewards - initial_values

            prev_policy = copy.deepcopy(policy_model)

            policy_diff = log_y_policy - log_y_prev_policy
            policy_ratio = torch.exp(policy_diff).clamp_(0, 1)

            surr1 = policy_ratio * advantages
            surr2 = torch.clamp(policy_ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages

            ppo_loss = torch.mean(-1.0 * torch.min(surr1, surr2))
            ppo_loss.backward(retain_graph=True)
            optimizer_policy.step()
            scheduler_policy.step()

            value_loss = torch.mean(nn.MSELoss()(values, rewards))
            value_loss.backward()
            optimizer_value.step()
            scheduler_value.step()
            
            total_steps = epoch * len(raw_dataloader) + i
            total_batches = len(raw_dataloader) * NUM_EPOCHS_POLICY
            percentage_completion = (total_steps / total_batches) * 100
            elapsed_time = time.time() - start_time
            
            if total_steps > 0:
                remaining_time = elapsed_time * (total_batches - total_steps) / total_steps

            print(f"------ Step: {total_steps}/{total_batches}, Completion: {percentage_completion:.2f}%, Time Elapsed: {elapsed_time // 60:.2f} min, Estimated Remaining Time: {remaining_time // 60:.2f} min, PPO Loss: {ppo_loss:.2f}, Value Loss: {value_loss:.2f}, Mean Reward: {torch.mean(rewards).item():.2f} ------")
            
            if total_steps % SAVE_CKPT_STEPS == 0:
                save_ckpt(CHECKPOINT_PATH_PPO, policy_model, 'policy_', epoch, i)
                save_ckpt(CHECKPOINT_PATH_PPO, value_model, 'value_', epoch, i)

        save_ckpt(CHECKPOINT_PATH_PPO, policy_model, 'policy_', epoch, i)
        save_ckpt(CHECKPOINT_PATH_PPO, value_model, 'value_', epoch, i)
