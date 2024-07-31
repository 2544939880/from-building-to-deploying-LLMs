import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

from thop import profile
from torchsummary import summary

class Trainer():
    '''
    A class to encapsulate the training, evaluation, and testing procedures for a PyTorch model.

    This class supports various features including learning rate warmup, cosine decay, gradient 
    clipping, and periodic evaluation. It can handle both classification and regression tasks.
    '''
    
    def __init__(
            self, 
            model: nn.Module, 
            device: torch.device, 
            num_epochs: int,
            optimizer: torch.optim.Optimizer,
            train_loader: DataLoader = None, 
            valid_loader: DataLoader = None, 
            eval_freq: int = None, 
            eval_iter: int = None,
            is_classification: bool = False, 
            warmup: bool = False,
            cos_dec: bool = False,
            grd_clip: bool = False, 
            inital_lr: float = 3e-5, 
            min_lr: float = 1e-6,
            checkpoint_path: str = None,
            tokenizer: tiktoken.Encoding = None,
    ):
        """
        Initializes the Trainer object with the provided model, training parameters, and options.

        Args:
        - model: The PyTorch model.
        - device: The device to run the model on ('cpu', 'mps', or 'cuda').
        - num_epochs: The number of epochs for training.
        - optimizer: The optimizer for training the model.
        - train_loader: The DataLoader for training data (default: None).
        - valid_loader: The DataLoader for validation data (default: None).
        - eval_freq: Evaluation frequency to control how often (in epochs) the model is evaluated 
                     during training (default: None).
        - eval_iter: Evaluation iterations to control the number of batches processed (default: None).
        - is_classification: Whether the task is classification, if is_classification is True, only 
                             the last time step of the model's output is used for loss calculation
                             (default: False).
        - warmup: Whether to use learning rate warmup (default: False).
        - cos_dec: Whether to use cosine decay for learning rate (default: False).
        - grd_clip: Whether to use gradient clipping (default: False).
        - initial_lr: Initial learning rate with warmup (default: 3e-5).
        - min_lr: Minimum learning rate with cosine decay (default: 1e-6).
        - checkpoint_path: The path to the directory containing the model checkpoints (default: None).
        """
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.is_classification = is_classification
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.warmup = warmup
        self.cos_dec = cos_dec
        self.grd_clip = grd_clip
        self.inital_lr = inital_lr
        self.min_lr = min_lr
        self.checkpoint_path = checkpoint_path
        self.tokenizer = tokenizer

        # Initializes the logs to keep track of training progress
        self.log = {
            "train_losses": [],      # List to store training losses
            "valid_losses": [],      # List to store validation losses
            "train_accs": [],        # List to store training accuracies
            "valid_accs": [],        # List to store validation accuracies
            "track_tokens_seen": [], # List to track tokens seen (if applicable)
            "track_lrs": [],         # List to track learning rates
            "global_step": -1,       # Counter for the global step in training
        }


    def __log__(self):
        """
        Get the training logs

        Returns:
        - log (dict): A dictionary containing the training logs.
        """
        return self.log
    
    def __calculate_loss_batch(
            self, 
            input_batch: torch.Tensor, 
            target_batch: torch.Tensor,
    ):
        """
        Calculates the loss for a single batch of data.
        
        Args:
        - input_batch (torch.Tensor): Input data batch.
        - target_batch (torch.Tensor): Target labels batch.
    
        Returns:
        - loss (torch.Tensor): The computed loss for the batch.
        """
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        # Forward pass
        logits = self.model(input_batch)

        if self.is_classification:
            # For classification tasks, use only the last time step's output
            logits = logits[:, -1, :]  # [batch_size, num_classes]
            loss = F.cross_entropy(logits, target_batch)
        else:
            # For prediction tasks, flatten logits and target
            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        
        return loss

    def __calculate_loss(self, data_loader:DataLoader):
        """
        Calculates the average loss over all batches (or self.eval_iter batches) in the data loader.
        
        Args:
        - data_loader (DataLoader): The data loader providing batches of data.
        
        Returns:
        - average_loss (float): The average loss over all processed batches.
        """
        total_loss = 0.0

        if self.eval_iter is None:
            num_batches = len(data_loader)
        else:
            # Control the number of batches in the dataloader with "num_batches"
            num_batches = min(self.eval_iter, len(data_loader))
        
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                # Calculate loss for the current batch
                batch_loss = self.__calculate_loss_batch(input_batch, target_batch)
                total_loss += batch_loss.item()
            else:
                break

        # Compute the average loss over processed batches
        return total_loss / num_batches

    def __calculate_accuracy(self, data_loader: DataLoader):
        """
        Calculates the accuracy over all batches (or self.eval_iter batches) for 
        classification tasks using the provided data loader.

        Args:
        - data_loader (DataLoader): The data loader providing batches of data.

        Returns:
        - accuracy (float): The accuracy over all processed batches.
        """
        self.model.eval()   # Set the model to evaluation mode
        correct_predictions, total_predictions = 0, 0

        if self.eval_iter is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(self.eval_iter, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                # Move input and target batches to the device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # Disable gradient calculation for inference
                with torch.no_grad():
                    logits = self.model(input_batch)[:, -1, :]   # [batch_size, num_classes]
                
                # Get the predicted labels
                predicted_labels = torch.argmax(logits, dim=-1)
                
                # Update total and correct predictions count
                total_predictions += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
        
        return correct_predictions / total_predictions

    def __text_to_token_ids(self, text):
        token_ids = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0) # add batch dimension
        return token_ids_tensor

    def __token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0) # remove batch dimension
        return self.tokenizer.decode(flat.tolist())

    def model_information(self):
        """
        Prints a summary of the model including the number of parameters and GFLOPs.

        This method takes a batch of input data from the training loader. It then prints 
        the model summary, the number of floating point operations per second (FLOPs), 
        and the number of parameters in the model.
        """
        # Get a batch of input data from the training loader
        input_batch, _ = next(iter(self.train_loader))
        input_batch = input_batch.to(self.device)

        # Print model summary
        summary(self.model, input_data=input_batch, device=self.device)

        # Calculate FLOPs and parameters
        flops, params = profile(self.model, (input_batch,))

        # Print FLOPs and parameters in human-readable format
        print(f'GFLOPs: {flops / 1000**3 : .2f}')
        print(f'MParams: {params / 1000**2 : .2f}')
        print("="*100)

        # Print training environment information
        print("Accelerate device: ", self.device)
        print("The number of epochs: ", self.num_epochs)
        print("Evaluation frequency: ", self.eval_freq)
        print("Evaluation iterations: ", self.eval_iter)
        print("Classifiter: ", self.is_classification)
        print("Learning rate warmup: ", self.warmup)
        print("Learning rate cosine decay: ", self.cos_dec)
        print("Gradient clipping: ", self.grd_clip)
        print("="*100)

    def training(self):
        """
        Trains the model over multiple epochs with options for warmup and cosine decay
        learning rate scheduling, gradient clipping, and optional evaluation steps.
        """
        warmup_step = 0
        # Get the peak learning rate of optimizer
        peak_lr = self.optimizer.param_groups[0]["lr"]
        
        # Calculate the total trainging steps and wramup step
        total_train_step = len(self.train_loader) * self.num_epochs

        if self.warmup:
            # 20% warmup
            warmup_step = int(0.20 * total_train_step)  
            # Calculate the learning increment of the warmup stage
            lr_increment = (peak_lr - self.inital_lr) / warmup_step

        # Main training loop
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode

            for i, (input_batch, target_batch) in enumerate(self.train_loader):
                # print(f"{i}: {input_batch.shape}, {target_batch.shape}")
                self.optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                self.log["global_step"] += 1

                # Adjust the learning rate (warmup or cosine decay stage)
                if self.log["global_step"] < warmup_step:
                    # Liner warmup
                    lr = self.inital_lr + lr_increment * self.log["global_step"]
                elif self.cos_dec:
                    # Cosine decay after warmup
                    progress = ((self.log["global_step"] - warmup_step) /
                                (total_train_step - warmup_step))
                    lr = self.min_lr + ((peak_lr - self.min_lr) * 0.5 * 
                                (1 + math.cos(math.pi * progress)))
                else:
                    lr = peak_lr
                
                # load learning rate to the optimizer
                for param_group in  self.optimizer.param_groups:
                    param_group["lr"] = lr
                self.log["track_lrs"].append(lr)    

                # Calculate and backpropagate the loss
                loss = self.__calculate_loss_batch(input_batch, target_batch)
                loss.backward() 

                # Gradient clipping after the warmup stage
                if self.grd_clip and (self.log["global_step"] > warmup_step):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()  # Update model weights using loss gradients
            
                # Optional evaluation step
                if self.log["global_step"] % self.eval_freq == 0:
                    self.model.eval()
                    with torch.no_grad():
                        train_loss = self.__calculate_loss(self.train_loader)
                        if self.valid_loader is not None: 
                            valid_loss = self.__calculate_loss(self.valid_loader)
                        else:
                            valid_loss = 0.0
                    self.model.train()
            
                    self.log["train_losses"].append(train_loss)
                    self.log["valid_losses"].append(valid_loss)
                    print(f"Ep {epoch+1} (Step {self.log["global_step"]:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {valid_loss:.3f}")

            # For classifitation tasks, calculate the train and validation accuracy
            if self.is_classification:
                train_acc = self.__calculate_accuracy(self.train_loader)
                if self.valid_loader is not None: 
                    valid_acc = self.__calculate_accuracy(self.valid_loader)
                else:
                    valid_acc = 0.0
                self.log["train_accs"].append(train_acc)
                self.log["valid_accs"].append(valid_acc)  

                print(f"Ep {epoch+1} (Step {self.log["global_step"]:06d}): "
                      f"Training accuracy: {train_acc*100:.2f}% | "
                      f"Validation accuracy: {valid_acc*100:.2f}%")
                
            # Save the trained model when the setting condition is met
            if (self.checkpoint_path is not None) and (valid_acc > 0.94):
                # Create the checkpoint directory if it doesn't exist
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                file_dir = os.path.join(self.checkpoint_path, 
                                        f"gpt2_small_{self.log["global_step"]+1}.pth"
                                        )
                
                torch.save(self.model.state_dict(), file_dir)
                print(f"Saved path: {file_dir}")

    def evaluate(self, eval_loader: DataLoader, checkpoint_path):
        """
        Evaluates the classification model using checkpoints found in the specified path.

        Args:
        - eval_loader (DataLoader): The data loader providing batches of evaluation data.
        - checkpoint_path (str): The path to the directory containing the model checkpoints.

        Prints:
        - Evaluation loss and accuracy for each checkpoint.
        """
        self.model.eval() # Set model to evaluation mode

        # Temporarily store the current value of eval_iter and set it to None
        original_eval_iter = self.eval_iter
        self.eval_iter = None

        file_dir = os.listdir(checkpoint_path)
        for filename in file_dir:
            if ".pth" in filename:
                checkpoint = os.path.join(checkpoint_path, filename)
            else:
                continue
            self.model.load_state_dict(torch.load(checkpoint), strict=False)
            with torch.no_grad():
                eval_loss = self.__calculate_loss(eval_loader)
                eval_acc = self.__calculate_accuracy(eval_loader)
            print(f"{filename} -> Test loss: {eval_loss: .3f}, accuracy: {eval_acc*100: .2f}%")
        
        # Restore the original value of eval_iter
        self.eval_iter = original_eval_iter

    def text_generator(self, prompt, max_new_tokens, temperature=0.0, top_k=None):
        self.model.eval()
        context_length = self.model.position_emb.weight.shape[0]
        token_ids = self.__text_to_token_ids(prompt).to(self.device)

        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the supported context length
            # E.g., if LLM supports only 5 tokens, and the context length is 10
            # then only the last 5 tokens are used as context
            ids_cond = token_ids[:, -context_length:]
            
            # Get the predictions
            with torch.no_grad():
                logits = self.model(ids_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # The logits with the top k sampled
            if top_k is not None:
                top_logits, top_positions = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            if temperature > 0.0:
                logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)   # [batch_size, context_length]
                id_next = torch.multinomial(probs, num_samples=1)  # [batch_length, 1]
            
            else:
                # Get the idx of the vocab entry with the highest logits value
                id_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            if id_next == self.tokenizer.encode('<|endoftext|>', allowed_special={"<|endoftext|>"}):
                break

            # Append sampled token id to the running sequence
            token_ids = torch.cat((token_ids, id_next), dim=1)  # (batch, n_tokens+1)

        return self.__token_ids_to_text(token_ids)


def plot_values(num_epochs, total_predictions, train_values, valid_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3), dpi=300)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_values))
    total_preds_tensor  = torch.linspace(0, total_predictions, len(train_values))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_tensor, train_values, label=f"Training {label}")
    ax1.plot(epochs_tensor, valid_values, linestyle ="-.",label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create the second x-axis for total predictions
    ax2 = ax1.twiny()   # Create a second x-axis that shares the same y-axis
    ax2.plot(total_preds_tensor, train_values, alpha=0)
    ax2.set_xlabel("Total predictions")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")


# def __plot_lrs(num_epochs, total_predictions, track_lrs):
#     fig, ax1 = plt.subplots(figsize=(5, 3), dpi=300)

#     epochs_tensor = torch.linspace(0, num_epochs, len(track_lrs))
#     total_preds_tensor  = torch.linspace(0, total_predictions, len(track_lrs))
    
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel()

#     plt.ylabel("Learning rate")
#     plt.xlabel("Step")

#     total_training_steps = len(train_loader) * n_epochs
#     plt.plot(range(total_training_steps), track_lrs)
#     plt.tight_layout(); 
#     plt.savefig("1.pdf")
