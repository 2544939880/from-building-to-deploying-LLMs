import os
import json
import urllib.error
import urllib.request
import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split


# Define the file path and URL for the dataset
file_path = "practice-B/instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

class InstructionDataset(Dataset):
    def __init__(
            self, 
            tokenizer: tiktoken.Encoding=tiktoken.get_encoding("gpt2"),
            data_path: str=file_path,
            url: str=url,
        ):
        self.data_path = data_path
        self.url = url
        # Download and load the dataset
        self.data = self.__download_and_load_file__()

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in self.data:
            full_text = self.__format_prompt__(entry) # Format the prompt
            self.encoded_texts.append(
                # Tokenize the formatted prompt
                tokenizer.encode(full_text, allowed_special={"<|endoftext|>"}) 
            )
    
    def __download_and_load_file__(self):
        """
        Downloads the dataset file if it doesn't exist, and then loads it.
        """
        if os.path.exists(self.data_path):
            print(f"{self.data_path} already exists. Skipping download and extraction.")
        else:
            # Downloading the file
            try:
                with urllib.request.urlopen(self.url) as response:
                    with open(self.data_path, 'w', encoding='utf-8') as file:
                        file.write(response.read().decode('utf-8'))  
                print(f'File successfully downloaded to {self.data_path}')
            except urllib.error.HTTPError as e:
                print(f'HTTP Error: {e.code}')
            except urllib.error.URLError as e:
                print(f'URL Error: {e.reason}')
            except Exception as e:
                print(f'Unexpected Error: {str(e)}')

        # load file
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data

    def __format_prompt__(self, entry):
        """
        Formats the data entry into the required prompt structure.
        """
        input_text = self.__format_input__(entry)
        response_text = f"\n\n### Response:\n{entry['output']}"

        return input_text + response_text
    
    def __format_input__(self, entry):
        """
        Formats the data entry into the required input structure.
        """
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text
    

    def __getitem__(self, index):
        """
        Returns the tokenized text at the specified index.
        """
        return self.encoded_texts[index]

    def __len__(self):
        """
        Returns the number of data entries.
        """
        return len(self.data)


# Desiged the collate function
def __custom_collate_fn(batch, pad_token_id=50256, ignore_id=-100, allowed_length=None):        
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


def create_dataloader(
        batch_size,
        num_workers,
        dataset: Dataset,
        split_rate_list: list[float]=[0.85, 0.05, 0.1],
        collate_fn=__custom_collate_fn,
):
    assert (sum(split_rate_list) == 1.0), \
    "The sum of the split rates should be equal to 1"

    train_portion = int(len(dataset) * split_rate_list[0])  # 85% for training
    test_portion = int(len(dataset) * split_rate_list[-1])    # 10% for testing
    val_portion = len(dataset) - train_portion - test_portion  # Remaining 5% for validation

    train_set, val_set, test_set = random_split(
            dataset=dataset,
            lengths=[train_portion, val_portion, test_portion],
            generator=torch.Generator().manual_seed(123),
    )
    
    # Create the dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = InstructionDataset()
    data = dataset.__download_and_load_file__()
    print(len(data), data[50])
    format_data = dataset.__format_prompt__(data[50])
    print(format_data)

    train_loader, val_loader, test_loader = create_dataloader(
        num_workers=0,
        batch_size=8,
        dataset=dataset,
    )

    print("Train loader:")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"{[i]} -> ", inputs.shape, targets.shape)
        print(inputs[0], "\n", targets[0])
        break
