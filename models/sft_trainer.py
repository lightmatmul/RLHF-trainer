from trl import SFTTrainer
from model_utils import load_model_and_tokenizer

def train_sft_model(model_name, training_data, eval_data, training_arguments, config):
    model, tokenizer = load_model_and_tokenizer(model_name, config)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=training_data,
        eval_dataset=eval_data,
        peft_config=config,
        dataset_text_field="text",
        # set to max context len of llama 2 (can be extended using RoPE training)
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )
    trainer.train()
    return trainer
