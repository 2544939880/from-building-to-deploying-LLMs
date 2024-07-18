import torch
import tiktoken
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

from thop import profile
from torchsummary import summary

def text_to_token_ids(text, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
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

        # 对logits进行top_k采样 (可选)
        if top_k is not None:
            top_logits, top_positions = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 温度系数缩放（可选）
        if temperature > 0.0:
            logits = logits / temperature

            # 根据分布概率采样
            probs = torch.softmax(logits, dim=-1)   # [batch_size, context_size]
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
        
        else:
            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 出现eos token 终止生成
        if idx_next == eos_id: break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded, max_new_tokens=23, 
            context_size=context_size, temperature=1, top_k=1, eos_id=50256,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # print("Output text: \n", decoded_text)
        print(decoded_text.replace("\n", " "))  # Compact print format


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_cls_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]   # [batch_size, num_classes]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_epoch(data_loader: DataLoader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        print("nan")
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Control the number of batches in the dataloader with "num_batches"
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            batch_loss = calc_loss_cls_batch(input_batch, target_batch, model, device)
            total_loss += batch_loss.item()
        else:
            break

    return total_loss / num_batches

def calc_accuracy_loader(data_loader: DataLoader, model, device, num_batches=None):
    model.eval()
    correct_predictions, total_predictions = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]   # [batch_size, num_classes]
            predicted_labels = torch.argmax(logits, dim=-1)
            
            total_predictions += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / total_predictions


def evaluate_model(model, train_loader, valid_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_epoch(train_loader, model, device, eval_iter)
        valid_loss = calc_loss_epoch(valid_loader, model, device, eval_iter)
    model.train()
    return train_loss, valid_loss

def train_model_simple(model, train_loader, valid_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

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
                    model, train_loader, valid_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def train_model(model, train_loader, valid_loader, optimizer, num_epochs, 
                device, eval_freq, eval_iter, start_context, tokenizer, 
                warmup_rate=0.20, inital_lr=3e-5, min_lr=1e-6):
    
    train_losses, valid_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # 获取 optimizer 中最大学习率
    peak_lr = optimizer.param_groups[0]["lr"]
    
    # 计算训练阶段迭代总次数和 warmup_step
    total_training_steps = len(train_loader) * num_epochs
    warmup_step = int(warmup_rate * total_training_steps)

    # 计算 Warmup 阶段学习率增量
    lr_increment = (peak_lr - inital_lr) / warmup_step

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()   # 初始化梯度值
            global_step += 1

            # 根据当前阶段（Warmup 或 Cosine annealing）调整学习率
            if global_step < warmup_step:
                # Liner warmup
                lr = inital_lr + lr_increment * global_step
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_step) /
                            (total_training_steps - warmup_step))
                lr = min_lr + ((peak_lr - min_lr) * 0.5 * 
                               (1 + math.cos(math.pi * progress)))
            
            # load learning rate to the optimizer
            for param_group in  optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)    

            batch_loss = calc_loss_batch(input_batch, target_batch, model, device)
            batch_loss.backward()   # 计算损失梯度  

            # 在 Warmup 阶段后应用 Gradient clipping 以避免梯度爆炸
            if global_step > warmup_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()        # 根据梯度更新模型权重
            tokens_seen += input_batch.numel()
            
            # 评估阶段（可选）
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(
                    model, train_loader, valid_loader, device, eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {valid_loss:.3f}")

        # Print a text after each epoch
        generate_and_print(model, tokenizer, device, start_context)

    return train_losses, valid_losses, track_tokens_seen

def model_infomation_print(model, input_batch, device) -> None:
    input_batch = input_batch.to(device)
    summary(model, input_data=input_batch, device=device)
    flops, params = profile(model, (input_batch,))

    print(f'GFLOPs: {flops / 1000**3 : .2f}')
    print(f'MParams: {params / 1000**2 : .2f}')
    print("="*100, '\n')

def tarin_classifier_model(model, train_loader, valid_loader, optimizer, device, 
                            num_epochs, eval_freq, eval_iter, warmup=False, cos_dec=False,
                            grd_clip=False, inital_lr=3e-5, min_lr=1e-6):
    # Initialize lists to track losses and accuracy
    train_losses, val_losses, train_accs, val_accs, track_lrs= [], [], [], [], []
    warmup_step, total_predictions, global_step = 0, 0, -1

    # Get the peak learning rate of optimizer
    peak_lr = optimizer.param_groups[0]["lr"]
    
    # Calculate the total trainging steps and wramup step
    total_training_steps = len(train_loader) * num_epochs

    if warmup:
        # 20% warmup
        warmup_step = int(0.20 * total_training_steps)  
        # Calculate the learning increment of the warmup stage
        lr_increment = (peak_lr - inital_lr) / warmup_step

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for i, (input_batch, target_batch) in enumerate(train_loader):
            # print(f"{i}: {input_batch.shape}, {target_batch.shape}")
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            global_step += 1

            # Adjust the learning rate (warmup or cosine decay stage)
            if global_step < warmup_step:
                # Liner warmup
                lr = inital_lr + lr_increment * global_step
            elif cos_dec:
                # Cosine decay after warmup
                progress = ((global_step - warmup_step) /
                            (total_training_steps - warmup_step))
                lr = min_lr + ((peak_lr - min_lr) * 0.5 * 
                            (1 + math.cos(math.pi * progress)))
            else:
                lr = peak_lr
            
            # load learning rate to the optimizer
            for param_group in  optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)    

            loss = calc_loss_cls_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients

            # Gradient clipping after the warmup stage
            if grd_clip and (global_step > warmup_step):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update model weights using loss gradients
            total_predictions += input_batch.shape[0]

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, valid_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        

        train_acc = calc_accuracy_loader(train_loader, model, device, eval_iter)
        val_acc = calc_accuracy_loader(valid_loader, model, device, eval_iter)
        print(f"Training accuracy: {train_acc*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_acc*100:.2f}%")
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save the trained model
        if val_acc > 0.90:
            torch.save(model.state_dict(), f"spam_gpt2_{int(val_acc*100)}_{global_step+1}.pth")
        
    return train_losses, val_losses, train_accs, val_accs, total_predictions, track_lrs


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

@torch.no_grad()    # Disabled the gradient
def test_classifier_model(test_loader, model, device):
    test_loss = calc_loss_epoch(test_loader, model, device)
    test_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"Test loss: {test_loss: .3f}, accuracy: {test_acc*100: .2f}%")

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
