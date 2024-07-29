from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import pandas as pd



class SpamDataset(Dataset):
    def __init__(self, file_csv, tokenizer, max_length=None, padding_token_id=50256):
        super().__init__()
        self.data = pd.read_csv(file_csv)

        # Pre-tokenize texts
        self.tokened_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        self.labels = [label for label in self.data["Label"]]
        
        if max_length is None:
            self.max_length = self.__longest_encoded_length()
        else:
            self.max_length = max_length
            # Cut off the sequence if the sequence length over the max_length
            self.tokened_texts = [
                tokened_text[:self.max_length] for tokened_text in self.tokened_texts
            ]

        # Padding the last tokens to max_length
        self.tokened_texts = [
            tokened_text + [padding_token_id] * (self.max_length - len(tokened_text))
            for tokened_text in self.tokened_texts
        ]


    def __longest_encoded_length(self,):
        max_length = 0
        for tokened_text in self.tokened_texts:
            text_length = len(tokened_text)
            if text_length > max_length:
                max_length = text_length
        return max_length

    def __len__(self,):
        return len(self.tokened_texts)

    def __getitem__(self, index):
        text = self.tokened_texts[index]
        label = self.labels[index]

        return (
            torch.tensor(text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

def create_dataloader(tokenizer, batch_size, num_workers):
    train_set = SpamDataset(f"./practice-A/sms_spam_collection/train.csv", tokenizer)
    valid_set = SpamDataset(f"./practice-A/sms_spam_collection/validation.csv", tokenizer)
    test_set = SpamDataset(f"./practice-A/sms_spam_collection/test.csv", tokenizer)

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        # persistent_workers=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        # persistent_workers=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        # persistent_workers=True
    )
    
    print("="*100)
    print(f"{len(train_loader)} training batches")
    print(f"{len(valid_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    return train_loader, valid_loader, test_loader


