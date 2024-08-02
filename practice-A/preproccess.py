import urllib
import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

# URL for the dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Paths for saving the downloaded zip file and the extracted data
zip_path = "./practice-A/sms_spam_collection.zip"
extracted_path = "./practice-A/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    # Check if the data file already exists to skip download and extraction if so
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    
    # Downloading the file
    try:
        with urllib.request.urlopen(url) as response:
            with open(zip_path, 'wb') as file:
                file.write(response.read())  
        print(f'File successfully downloaded to {zip_path}')
    except urllib.error.HTTPError as e:
        print(f'HTTP Error: {e.code}')
    except urllib.error.URLError as e:
        print(f'URL Error: {e.reason}')
    except Exception as e:
        print(f'Unexpected Error: {str(e)}')

    # Extract the downloaded zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    
    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File successfully extracted and saved as {data_file_path}")

def create_balanced_dataset(df: pd.DataFrame):
    # Find the number of spam messages
    num_spam = df[df['Label'] == 'spam'].shape[0]
    # Sample the same number of ham messages
    ham_subset = df[df['Label'] == 'ham'].sample(num_spam, random_state=123)
    
    # Combine the spam messages and the sampled ham messages
    balanced_df = pd.concat([ham_subset, df[df["Label"] == 'spam']])

    # Map the string labels to integer labels: 'ham' -> 0, 'spam' -> 1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    return balanced_df

def random_split(df: pd.DataFrame, train_frac, valid_frac, test_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Calculate the indices for splitting the DataFrame
    train_end_idx = int(len(df) * train_frac)
    valid_end_idx = train_end_idx + int(len(df) * valid_frac)

    # Split the DataFrame into training, validation, and test sets
    train_df = df[: train_end_idx]
    valid_df = df[train_end_idx: valid_end_idx]
    test_df = df[valid_end_idx: ]

    # Print the sizes of the splits
    print("="*100)
    print("Total dataset: ", len(df))
    print("Train set: ", len(train_df))
    print("Validation set: ", len(valid_df))
    print("Test set: ", len(test_df))

    # Save the splits to CSV files
    train_df.to_csv(f"{Path(extracted_path)}/train.csv", index=None)
    valid_df.to_csv(f"{Path(extracted_path)}/validation.csv", index=None)
    test_df.to_csv(f"{Path(extracted_path)}/test.csv", index=None)

def create_files_csv(data_file_path, split_rate: list, balanced=False):
    # Read the dataset from the file
    df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])
    print(df)
    print(df['Label'].value_counts())
    
    # Create a balanced dataset if specified
    if balanced:
        df = create_balanced_dataset(df)
        print(df['Label'].value_counts())

    # Split the dataset and save the splits to CSV files
    random_split(df, split_rate[0], split_rate[1], split_rate[2])
    print(f"{split_rate} random splited and saved to train.csv, validation.csv, and test.csv")


if __name__ == "__main__":
    # Download, extract, and process the dataset
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    create_files_csv(data_file_path, [0.7, 0.1, 0.2], balanced=True)
