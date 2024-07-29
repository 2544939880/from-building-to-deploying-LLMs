import tiktoken
import torch
import torch.nn as nn
from preproccess import InstructionDataset, download_and_load_file
from torch.utils.data import DataLoader

torch.manual_seed(123)

file_path = "instructions_finetuning/instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

# Hyper-parameters configuration
HYPER_CONFIG = {
    "batch_size" : 8,   
    "num_workers" : 0,
    "num_epochs" : 5,
    "lr" : 1e-4,
    "weight_decay" : 1e-5,
    "num_classes" : 2,
} 

tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

data = download_and_load_file(file_path, url)

# Split the data to train, validation, and test set
train_rate = int(len(data) * 0.85)   # 85% train rate
valid_rate = int(len(data) * 0.05)   # 5% validation rate
train_data = data[: train_rate]
valid_data = data[train_rate : train_rate + valid_rate]
test_data = data[train_rate + valid_rate: ]   

# Create the dataset of train, validation and test
train_set = InstructionDataset(train_data, tokenizer)
valid_set = InstructionDataset(valid_data, tokenizer)
test_set = InstructionDataset(test_data, tokenizer)

# Desiged the collate function
def collate_fn(batch, pad_token_id=50256, ignore_id=-100, allowed_length=None):        
    batch_max_length = max(len(data) for data in batch)
    inputs, targets = [], []

    for item in batch:
        # Padding the last tokens to max_length+1
        # Adds at least 1 additional padding tokens to easily split target_batch
        new_item = item.copy()
        new_item = new_item + [pad_token_id] * (batch_max_length - len(item) + 1)

        # Create the input and target of each batch
        input = torch.tensor(new_item[: -1], dtype=torch.int32)
        target = torch.tensor(new_item[1: ], dtype=torch.int32)

        # The ingnore_index value to replace all padding token IDs of the target, 
        # that can ingnore padding value in the loss function calculation
        mask = target == pad_token_id
        mask = [idx for idx, item in enumerate(mask) if item]
        target[mask[1:]] = ignore_id

        # Optionally truncate to maximum length
        if allowed_length is not None:
            input = input[:allowed_length]
            target = target[:allowed_length]

        inputs.append(input)
        targets.append(target)

    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)
    return inputs_tensor, targets_tensor


# Create the dataloader
train_loader = DataLoader(
    dataset=train_set,
    batch_size=HYPER_CONFIG["batch_size"],
    num_workers=HYPER_CONFIG["num_workers"],
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)
valid_loader = DataLoader(
    dataset=valid_set,
    batch_size=HYPER_CONFIG["batch_size"],
    num_workers=HYPER_CONFIG["num_workers"],
    collate_fn=collate_fn,
    shuffle=False,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=HYPER_CONFIG["batch_size"],
    num_workers=HYPER_CONFIG["num_workers"],
    collate_fn=collate_fn,
    shuffle=False,
    drop_last=False,
)



