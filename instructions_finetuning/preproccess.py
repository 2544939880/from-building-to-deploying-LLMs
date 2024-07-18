import os
import urllib.error
import urllib.request
import json
from torch.utils.data import DataLoader, Dataset
import torch
import tiktoken

file_path = "instructions_finetuning/instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

def download_and_load_file(file_path, url):
    if os.path.exists(file_path):
        print(f"{file_path} already exists. Skipping download and extraction.")
    else:
        # Downloading the file
        try:
            with urllib.request.urlopen(url) as response:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(response.read().decode('utf-8'))  
            print(f'File successfully downloaded to {file_path}')
        except urllib.error.HTTPError as e:
            print(f'HTTP Error: {e.code}')
        except urllib.error.URLError as e:
            print(f'URL Error: {e.reason}')
        except Exception as e:
            print(f'Unexpected Error: {str(e)}')

    # load file
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer: tiktoken.Encoding):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            full_text = self.__format_prompt__(entry)
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
        
    def __format_prompt__(self, entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        response_text = f"\n\n### Response:\n{entry['output']}"

        return instruction_text + input_text + response_text
    
    
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

