from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class SpamDataset(Dataset):
    def __init__(self, file_csv, tokenizer, max_length=None, padding_token_id=50256):
        """
        Initializes the SpamDataset object.

        Args:
        - file_csv: Path to the CSV file containing the dataset.
        - tokenizer: Tokenizer for encoding the text data.
        - max_length: Maximum length for tokenized sequences (default: None).
        - padding_token_id: Token ID used for padding (default: 50256).
        """
        super().__init__()
        # Load the data from CSV file
        self.data = pd.read_csv(file_csv)

        # Pre-tokenize texts
        self.tokened_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        # Extract labels
        self.labels = [label for label in self.data["Label"]]
        
        # Determine the maximum length for sequences
        if max_length is None:
            self.max_length = self.__longest_encoded_length()
        else:
            self.max_length = max_length
            # Cut off the sequence if the sequence length over the max_length
            self.tokened_texts = [
                tokened_text[:self.max_length] for tokened_text in self.tokened_texts
            ]

        # Pad sequences to the maximum length
        self.tokened_texts = [
            tokened_text + [padding_token_id] * (self.max_length - len(tokened_text))
            for tokened_text in self.tokened_texts
        ]

    def __longest_encoded_length(self,):
        """
        Determines the length of the longest encoded text sequence.
        
        Returns:
        - max_length: The length of the longest sequence.
        """
        max_length = 0
        for tokened_text in self.tokened_texts:
            text_length = len(tokened_text)
            if text_length > max_length:
                max_length = text_length
        return max_length

    def __len__(self,):
        """
        Returns the number of samples in the dataset.
        
        Returns:
        - Length of the dataset.
        """
        return len(self.tokened_texts)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.
        
        Args:
        - index: The index of the sample to retrieve.
        
        Returns:
        - A tuple containing the tokenized text and the label, both as tensors.
        """
        text = self.tokened_texts[index]
        label = self.labels[index]

        return (
            torch.tensor(text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

def create_dataloader(tokenizer, batch_size, num_workers):
    """
    Creates DataLoader objects for the training, validation, and test sets.

    Args:
    - tokenizer: Tokenizer for encoding the text data.
    - batch_size: Number of samples per batch.
    - num_workers: Number of subprocesses to use for data loading.
    
    Returns:
    - train_loader: DataLoader for the training set.
    - valid_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    """
    # Create dataset objects for training, validation, and test sets
    train_set = SpamDataset(f"./practice-A/sms_spam_collection/train.csv", tokenizer)
    valid_set = SpamDataset(f"./practice-A/sms_spam_collection/validation.csv", tokenizer)
    test_set = SpamDataset(f"./practice-A/sms_spam_collection/test.csv", tokenizer)

    # Set manual seed for reproducibility
    torch.manual_seed(123)

    # Create DataLoader for the training set
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        # persistent_workers=True,
    )

    # Create DataLoader for the validation set
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        # persistent_workers=True,
    )

    # Create DataLoader for the test set
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        # persistent_workers=True
    )
    
    # Print the number of batches for each DataLoader
    print("="*100)
    print(f"{len(train_loader)} training batches")
    print(f"{len(valid_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    return train_loader, valid_loader, test_loader


