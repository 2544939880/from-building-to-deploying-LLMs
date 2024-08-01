# ⚡️ From Building to Deploying LLMs

This repository aims at the complete lifecycle of developing large language models (LLMs), including all stages from model building and pre-training to fine-tuning and deployment.

# ⚙️ Setup

This repository running environment `python=3.12`. If you already have a Python installation on your machine, the quickest way to get started is to install the package requirements from the [requirements.txt](./requirements.txt) file by executing the following pip installation command from the root directory of this code repository: 
    
    pip install -r requirements.txt

>[!TIP]
>I am using computers running macOS (Macmini M2 16GB), but this workflow is similar for Linux machines and may work for other operating systems as well.


# 🧑‍💻 Code Specification

- [`architecture.py`](./architecture.py): This code describes the architecture of a GPT model implemented using PyTorch.

- [`load_weigths.py`](./load_weigths.py): This code defines some function, which loads pre-trained weights from a Hugging Face GPT-2 model into a custom GPT model architecture defined in the architecture.py.

- [`trainer.py`](./trainer.py): A class to encapsulate the training, evaluation, and testing procedures for a PyTorch model. This class supports various features including *learning rate warmup, cosine decay, gradient clipping, and periodic evaluation*. It can handle both classification and regression tasks.

# 🚂 Pre-training

The [demo.ipynb](./demo.ipynb) provides a comprehensive description of the pre-training process and its specifics.


# ⏩ Fine-tuning

The [practice-A](./practice-A/) and [practice-B](./practice-B/) involves fine-tuning pre-trained models like GPT-2 for various downstream tasks. This process focuses on understanding the specifics of fine-tuning for both classification and autoregressive tasks. Low requirements on hardware performance.
