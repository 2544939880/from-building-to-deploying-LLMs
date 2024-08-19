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
    "lr" : 0.00005,
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

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def main():
    # Create a GPT model and load pre-trained weights
    model = GPTModel(GPT_CONFIG_355M)
    model = load_weigths_gpt2_hf(model, CHOOSE_MODEL = "gpt2-medium (355M)")
    # model = freeze_model(model)
    # model= GPT2Model.from_pretrained("openai-community/gpt2-medium", cache_dir="checkpoints")
    model.eval()

    dataset = InstructionDataset()
    
    train_loader, val_loader, test_loader, test_input = create_dataloader(
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
    # model.to(device)
    # train_losses, val_losses, tokens_seen = train_model_simple(
    #     model, train_loader, val_loader, optimizer, device,
    #     num_epochs=2, eval_freq=5, eval_iter=5,
    #     start_context=test_input, tokenizer=tokenizer
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

    before_out = trainer.text_generator(test_input, max_new_tokens=50)
    print('-'*100)
    print(before_out)

    # Train the model and record running time
    start_time = time.time()
    trainer.training()
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {total_time_minutes:.2f} minutes.")

    after_out = trainer.text_generator(test_input, max_new_tokens=50)
    print('-'*100)
    print(after_out)

if __name__ == "__main__":
    main()

