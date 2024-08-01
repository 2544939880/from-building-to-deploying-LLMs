import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from data_loader import create_dataloader
from load_weigths import load_weigths_gpt2_hf

from trainer import Trainer

from architecture import GPTModel
import torch
import tiktoken

torch.manual_seed(123)

# Select accelerator device
if torch.backends.mps.is_available() or torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
else:
    device = torch.device("cpu")

# The gpt2-small (124M) parameter configuration
GPT_CONFIG_124M = {
        "vocab_size" : 50257,    # Vocabulary size
        "context_length" : 1024, # Context length         
        "emb_dim" : 768,         # Embedding dimension
        "n_heads" : 12,          # Number of attntion heads
        "n_layers" : 12,         # Number of layers
        "dropout" : 0.0,         # Dropout rate
        "qkv_bias" : True       # Query, Key, and Value bias
    }    

# Hyper-parameters configuration
HYPER_PARAMS_CONFIG = {
    "batch_size" : 8,   
    "num_workers" : 0,
    "num_epochs" : 5,
    "lr" : 1e-4,
    "weight_decay" : 1e-5,
    "num_classes" : 2,
}

def freeze_model(model: GPTModel, out_dim):
    torch.manual_seed(123)
    for param in model.parameters():
                param.requires_grad = False

    model.out_proj = torch.nn.Linear(
        in_features=model.out_proj.in_features, 
        out_features=out_dim, 
    )

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    # Init the out head weights and bias
    # torch.nn.init.xavier_uniform_(model.out_proj.weight)
    # if model.out_proj.bias is not None:
    #     torch.nn.init.zeros_(model.out_proj.bias)

    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")

    return model

def main():
    # Create a GPT model and load pre-trained weights
    model = GPTModel(GPT_CONFIG_124M)
    model = load_weigths_gpt2_hf(model)
    
    # Freeze model parameter and configuer the out dimention
    model = freeze_model(model, HYPER_PARAMS_CONFIG['num_classes'])
    model.to(device)

    # The gpt2 encoder of tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the train, validation, and test dataloader
    train_loader, valid_loader, test_loader = create_dataloader(
        tokenizer, HYPER_PARAMS_CONFIG["batch_size"], HYPER_PARAMS_CONFIG["num_workers"]
    )

    # Create a AdamW optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    # Create a SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=HYPER_PARAMS_CONFIG["lr"],
                                weight_decay=HYPER_PARAMS_CONFIG["weight_decay"],
                                momentum=0.99,
                                )


    trainer = Trainer(
            model=model, 
            device=device, 
            train_loader=train_loader,
            valid_loader=valid_loader,
            is_classification=True,
            num_epochs=HYPER_PARAMS_CONFIG["num_epochs"], 
            optimizer=optimizer,
            eval_freq=100,
            eval_iter=10,
            warmup=True,
            cos_dec=True,
            checkpoint_path="./practice-A/checkpoints/"
        )
    
    # Print the model information
    trainer.model_information()
    
    # train model
    start_time = time.time()
    trainer.training()
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {total_time_minutes:.2f} minutes.")

    # evaluate model
    trainer.evaluate(test_loader, checkpoint_path="./practice-A/checkpoints/")

    # # Plot the losses and accuracy of the train and validation sets
    # trainer.plot_values(
    #     HYPER_PARAMS_CONFIG["num_epochs"], total_examples, train_losses, val_losses, "loss")
    # trainer.plot_values(
    #     HYPER_PARAMS_CONFIG["num_epochs"], total_examples, train_accs, val_accs, "accuracy")
    # # plt.show()




if __name__ == "__main__":
    main()



