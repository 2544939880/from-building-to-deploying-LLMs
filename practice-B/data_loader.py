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
def __custom_collate_fn(batch, device="cpu", pad_token_id=50256, ignore_id=-100, allowed_length=None):        
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

    inputs_tensor = torch.stack(inputs).to(dtype=torch.long, device=device)
    targets_tensor = torch.stack(targets).to(dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor

def __format_input__(entry):
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

def create_dataloader(
        batch_size,
        num_workers,
        dataset: Dataset=None,
        split_rate_list: list[float]=[0.85, 0.05, 0.1],
        collate_fn=__custom_collate_fn,
):
    assert (sum(split_rate_list) == 1.0), \
    "The sum of the split rates should be equal to 1"

    data = download_and_load_file()

    train_portion = int(len(data) * split_rate_list[0])  # 85% for training
    test_portion = int(len(data) * split_rate_list[-1])    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation
    
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Train portion: ", train_portion)
    print("validation portion: ", val_portion)
    print("test portion: ", test_portion)
    
    test_input = __format_input__(val_data[0])

    train_set = InstructionDataset_V2(train_data)
    val_set = InstructionDataset_V2(val_data)
    test_set = InstructionDataset_V2(test_data)

    torch.manual_seed(123)
    
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

    return train_loader, val_loader, test_loader, test_input


def download_and_load_file(data_path=file_path, url=url):
    """
    Downloads the dataset file if it doesn't exist, and then loads it.
    """
    if os.path.exists(data_path):
        print(f"{data_path} already exists. Skipping download and extraction.")
    else:
        # Downloading the file
        try:
            with urllib.request.urlopen(url) as response:
                with open(data_path, 'w', encoding='utf-8') as file:
                    file.write(response.read().decode('utf-8'))  
            print(f'File successfully downloaded to {data_path}')
        except urllib.error.HTTPError as e:
            print(f'HTTP Error: {e.code}')
        except urllib.error.URLError as e:
            print(f'URL Error: {e.reason}')
        except Exception as e:
            print(f'Unexpected Error: {str(e)}')

    # load file
    with open(data_path, 'r') as file:
        data = json.load(file)
    return data


class InstructionDataset_V2(Dataset):
    def __init__(self, data, tokenizer: tiktoken.Encoding=tiktoken.get_encoding("gpt2")):
        self.data = data
        self.tokenizer = tokenizer
        
        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in self.data:
            full_text = self.__format_prompt__(entry) # Format the prompt
            self.encoded_texts.append(
                # Tokenize the formatted prompt
                tokenizer.encode(full_text, allowed_special={"<|endoftext|>"}) 
            )
    
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


if __name__ == "__main__":
    dataset = InstructionDataset()
    data = dataset.__download_and_load_file__()
    print(len(data), data[50])
    format_data = dataset.__format_prompt__(data[50])
    print(format_data)

    train_loader, val_loader, test_loader, _ = create_dataloader(
        num_workers=0,
        batch_size=8,
        dataset=dataset,
    )

    print("Train loader:")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"{[i]} -> ", inputs.shape, targets.shape)
        print(inputs[0], "\n", targets[0])
        break
