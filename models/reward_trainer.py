import torch
import time
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model
from utils.training_utils import save_ckpt, smoothed_plot_reward_loss, smoothed_plot_avg_scores
from configs.reward_config import *

def preference_loss(chosen, rejected):
    # Preference Loss function based on the derivation of the Bradley-Terry model.
    return torch.mean(-1.0 * (torch.log(torch.sigmoid(chosen - rejected))))

def train_reward_model(config, dataset):
    reward_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_8bit=True,
        device_map="auto"
    )
    reward_model = prepare_model_for_kbit_training(reward_model)
    reward_model = get_peft_model(reward_model, config)
    reward_model.print_trainable_parameters()

    reward_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.padding_side = "right"

    raw_dataloader = DataLoader(dataset['train'], batch_size=PER_DEVICE_BATCH_SIZE_REWARD)

    optimizer = torch.optim.Adam(reward_model.parameters(), lr=LR_REWARD)
    reward_loss_hist = []
    avg_score_chosen_hist = []
    avg_score_rejected_hist = []

    total_steps = len(raw_dataloader) * NUM_EPOCHS_REWARD // GRADIENT_ACCUMULATION_STEPS
    start_time = time.time()
    grad_accumulation_counter = 0
    current_step = 0

    # Train reward model
    reward_model.train()
    optimizer.zero_grad()
    for epoch in range(NUM_EPOCHS_REWARD):
        # Reset gradients accumulated in the optimizer 
        for i, batch in enumerate(raw_dataloader):
            grad_accumulation_counter += 1
            # Tokenize the chosen & rejected sequences in the batch
            inputs_chosen = reward_tokenizer(
                batch['chosen'], padding='max_length', truncation=True,
                max_length=MAX_LENGTH_REWARD, return_tensors="pt"
            )
            inputs_rejected = reward_tokenizer(
                batch['rejected'], padding='max_length', truncation=True,
                max_length=MAX_LENGTH_REWARD, return_tensors="pt"
            )

            # Extract input ids and attention masks for the chosen and rejected sequences
            input_ids_chosen = inputs_chosen['input_ids']
            attention_mask_chosen = inputs_chosen['attention_mask']
            input_ids_rejected = inputs_rejected['input_ids']
            attention_mask_rejected = inputs_rejected['attention_mask']

            # Forward pass through the reward model for both chosen and rejected sequences
            # Extract the last logits as the score of each sequence
            outputs_chosen = reward_model(
                input_ids=input_ids_chosen,
                attention_mask=attention_mask_chosen,
            ).logits[:, -1]
            outputs_rejected = reward_model(
                input_ids=input_ids_rejected,
                attention_mask=attention_mask_rejected,
            ).logits[:, -1]

            loss = preference_loss(outputs_chosen, outputs_rejected)
            avg_score_chosen = outputs_chosen.mean().item()
            avg_score_rejected = outputs_rejected.mean().item()
            loss.backward()

            # Update weights after GRADIENT_ACCUMULATION_STEPS
            if grad_accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                elapsed_time = (time.time() - start_time) / 60
                total_steps_remaining = total_steps - current_step
                time_per_step = elapsed_time / (current_step + 1)
                remaining_time = time_per_step * total_steps_remaining
                completion_percentage = (current_step / total_steps) * 100

                print(f'------ step: {current_step} / {total_steps}, '
                      f'percentage completion: {completion_percentage:.2f}%, '
                      f'loss: {loss.item():.2f}, avg score (chosen): {avg_score_chosen:.2f}, '
                      f'avg score (rejected): {avg_score_rejected:.2f}, '
                      f'elapsed time: {elapsed_time:.2f} min, '
                      f'remaining time: {remaining_time:.2f} min ------')

                current_step += 1

                # Save model every SAVE_CKPT_STEPS
                if grad_accumulation_counter % SAVE_CKPT_STEPS == 0:
                    save_ckpt(CHECKPOINT_PATH_REWARD, reward_model, 'reward_', epoch, grad_accumulation_counter)

                reward_loss_hist.append(loss.item())
                avg_score_chosen_hist.append(avg_score_chosen)
                avg_score_rejected_hist.append(avg_score_rejected)
        
        # Make sure to update weights after each epoch
        if grad_accumulation_counter % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
            # increment step after weight update if the last batch wasn't a full gradient accumulation step
            current_step += 1

    # Save the latest checkpoint after the training ends
    save_ckpt(CHECKPOINT_PATH_REWARD, reward_model, 'reward_', epoch, grad_accumulation_counter)
    smoothed_plot_reward_loss(reward_loss_hist, 'Reward model loss', 5)
    smoothed_plot_avg_scores(avg_score_chosen_hist, avg_score_rejected_hist, 'Average Scores', 5)