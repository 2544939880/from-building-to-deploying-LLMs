import sys
import os
# Get the absolute path of the parent directory of the current file's directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(parent_dir)

import time
import torch
import tiktoken
from data_loader import InstructionDataset, create_dataloader
from architecture import GPTModel
from load_weigths import load_weigths_gpt2_hf
from trainer import Trainer
from transformers import GPT2Model
# Set a manual seed for reproducibility
torch.manual_seed(123)

# Select accelerator device (cuda, mps, or cpu)
if torch.backends.mps.is_available() or torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
else:
    device = torch.device("cpu")


# The gpt2-medium (355M) parameter configuration
GPT_CONFIG_355M = {
        "vocab_size" : 50257,    # Vocabulary size
        "context_length" : 1024, # Context length         
        "emb_dim" : 1024,        # Embedding dimension
        "n_heads" : 16,          # Number of attntion heads
        "n_layers" : 24,         # Number of layers
        "dropout" : 0.0,         # Dropout rate
        "qkv_bias" : True        # Query, Key, and Value bias
}    

# Hyper-parameters configuration
HYPER_CONFIG = {
    "batch_size" : 4,   
    "num_workers" : 0,
    "num_epochs" : 2,
    "lr" : 5e-5,
    "weight_decay" : 0.1,
} 

def freeze_model(model: GPTModel):
    """
    Freezes the model parameters except the output layer and the last transformer block.
    
    Args:
    - model: The GPTModel object.

    Returns:
    - The modified GPTModel object with certain parameters frozen.
    """
    torch.manual_seed(123)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last transformer block and final layer norm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    for param in model.out_proj.parameters():
        param.requires_grad = True
    return model

def main():
    # Create a GPT model and load pre-trained weights
    model = GPTModel(GPT_CONFIG_355M)
    model = load_weigths_gpt2_hf(model, CHOOSE_MODEL = "gpt2-medium (355M)")
    # model = freeze_model(model)
    # model= GPT2Model.from_pretrained("openai-community/gpt2-medium", cache_dir="checkpoints")
    model.eval()

    dataset = InstructionDataset()
    
    train_loader, val_loader, test_loader = create_dataloader(
        num_workers=HYPER_CONFIG["num_workers"],
        batch_size=HYPER_CONFIG["batch_size"],
        dataset=dataset,
        split_rate_list=[0.85, 0.05, 0.1],    # [train, validation, test]
    )

    # The GPT-2 encoder tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create a AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=HYPER_CONFIG["lr"],
        weight_decay=HYPER_CONFIG["weight_decay"],
    )
    
    # Create an SGD optimizer
    # optimizer = torch.optim.SGD(model.parameters(),
    #     lr=HYPER_CONFIG["lr"],
    #     weight_decay=HYPER_CONFIG["weight_decay"],
    #     momentum=0.99,
    # )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        device=device,
        num_epochs=HYPER_CONFIG["num_epochs"],
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=val_loader,
        eval_freq=5,
        eval_iter=5,
        tokenizer=tokenizer,
        checkpoint_path="./practice-B/checkpoints/"
    )

    
    # Print the model information
    trainer.model_information()

    # Generate the test text before fine-tuning
    data = dataset.__download_and_load_file__()
    format_input = dataset.__format_input__(data[1])

    # before_out = trainer.text_generator(format_input, max_new_tokens=50)
    # print('-'*100)
    # print(before_out)

    # Train the model and record running time
    start_time = time.time()
    trainer.training()
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {total_time_minutes:.2f} minutes.")

    after_out = trainer.text_generator(format_input, max_new_tokens=50)
    print('-'*100)
    print(after_out)

if __name__ == "__main__":
    main()

