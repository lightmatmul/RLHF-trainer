from transformers import TrainingArguments

# training hyperparameters for SFT fine-tuning
training_arguments = TrainingArguments(
    output_dir="./sft_ckpt",
    num_train_epochs=12,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    fp16=False,
    bf16=False,
    save_steps=250,
    logging_steps=5,
    learning_rate=5e-5,
    group_by_length=True, # Group sequences by their length for efficient padding
    lr_scheduler_type="cosine", # Group sequences by their length for efficient padding
    report_to="wandb", # MLOps
    evaluation_strategy="steps",
    eval_steps=10
)
