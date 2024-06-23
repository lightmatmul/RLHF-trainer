import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model

def load_model_and_tokenizer(model_name, config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # load in 8 bit quantization to save memory
        device_map="auto"  # Automatic device allocating for loading
    )
    # Wrap with LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

