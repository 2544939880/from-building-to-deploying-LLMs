# Spam Classification with Pretrained GPT-2

This practice implements a spam classification model using the pre-trained GPT-2 (small) architecture. The model is trained and evaluated on the SMS Spam Collection dataset.

## Project Structure

```css
├── sms_spam_collection/
├── data_loader.py
├── main.py
├── preproccess.py
└── README.md
```

## Code Description

[**`preproccess.py`**](./preproccess.py):

This script is designed to download the SMS Spam Collection dataset, extract it, and process it into a balanced and randomly split format for training, validation, and testing purposes.

**File Functionality**

- `download_and_unzip_spam_data()`: The function first checks if the dataset has already been downloaded and extracted. If not, it downloads the zip file and extracts its contents.

- `create_balanced_dataset()`: Balances the dataset by ensuring that the number of spam and ham messages is equal.

- `random_split()`: The dataset is shuffled and split into training, validation, and test sets according to specified proportions.

- `create_files_csv()`: Saves the split datasets into separate CSV files.

---

[**`data_loader.py`**](./data_loader.py):

This script defines a custom dataset class (`SpamDataset`) for loading and processing the SMS Spam Collection dataset, and a function (`create_dataloader`) to create data loaders for training, validation, and testing. The data is tokenized, padded, and prepared for model training using PyTorch.

---

[**`main.py`**](./main.py)

This script sets up and **fine-tuning** a GPT-2 model on a **classification task** using the SMS Spam Collection dataset. It initializes the model, loads pre-trained weights, configures the model for classification, creates data loaders, and trains the model using an optimizer. The script also evaluates the model and prints training information.

---

## How to Run

1. **Set Up Environment**: Ensure you have Python installed and set up a virtual environment. Install the required packages:
   
        pip install -r requirements.txt

2. **Download and Prepare Data**: Run the `preproccess.py` script to download and prepare the SMS Spam Collection dataset.

        python preproccess.py

3. **Run Training**: Execute the `main.py` script to start training the model.

        python main.py


>[!TIP]
> - Ensure that your environment has the necessary compute resources (preferably a GPU) for training the model.  
>
> - Adjust the hyper-parameters in `HYPER_PARAMS_CONFIG` and `GPT_CONFIG_124M` as needed based on your hardware and dataset size.

