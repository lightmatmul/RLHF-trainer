from datasets import load_dataset
from torch.utils.data import DataLoader

def get_prompt_data(dataset_name):
    prompt_dataset = load_dataset(dataset_name)
    return prompt_dataset

def load_and_prepare_dataset(dataset_name, split_ratio=0.01):
    # Load and split dataset
    data = load_dataset(dataset_name, split='train').train_test_split(test_size=split_ratio)

    # Add special tokens to dataset
    def map_function(examples):
        return {
            # Specific to llama 2
            'text': [f'<s>[INST] <<SYS>>\n{system.strip()}\n<</SYS>>\n\n' + instruction + ' [/INST] ' + output + '</s>' for system, instruction, output in zip(examples['system'], examples['instruction'], examples['output'])]
        }

    training_data = data['train'].map(map_function, batched=True)
    eval_data = data['test'].map(map_function, batched=True)

    return training_data, eval_data

def get_reward_data(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

def create_reward_dataloader(dataset, batch_size):
    return DataLoader(
        dataset['train'],
        batch_size=batch_size
    )
