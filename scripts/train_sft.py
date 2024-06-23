import os
import transformers
from configs.lora_config import config
from configs.training_args import training_arguments
from data.data_loader import load_and_prepare_dataset
from models.model_utils import load_model_and_tokenizer
from models.sft_trainer import train_sft_model

model_name = "meta-llama/Llama-2-13b-hf"
dataset_name = "HumanDynamics/sft_dataset"
sft_model_path = "sft_model"

if __name__ == "__main__":
    training_data, eval_data = load_and_prepare_dataset(dataset_name)
    model, tokenizer = load_model_and_tokenizer(model_name, config)
    trainer = train_sft_model(model, tokenizer, training_data, eval_data, training_arguments, config)
    trainer.model.save_pretrained(sft_model_path)
