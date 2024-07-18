from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer: tiktoken.Encoding, sample_length, stride) -> None:
        super().__init__()
        self.input_idx = []
        self.target_idx = []

        token_idx = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in  range(0, len(token_idx) - sample_length + 1, stride):
            input_chunk = token_idx[i: i + sample_length]
            target_chunk = token_idx[i + 1: i + sample_length + 1]
            self.input_idx.append(torch.tensor(input_chunk))
            self.target_idx.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.target_idx)
    
    def __getitem__(self, index):
        return self.input_idx[index], self.target_idx[index]
    
def create_dataloader_V1(txt, batch_size=4, sample_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers = 7):
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    dataset = GPTDatasetV1(txt, tokenizer, sample_length, stride)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            drop_last=drop_last, 
                            num_workers=num_workers, 
                            persistent_workers=True)
    
    return dataloader

