import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from transformers import BertTokenizerFast
from transformers import AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='5k', choices=['5k', 'full'])
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'legal_bert', 'neo_bert'])
    args = parser.parse_args()
    return args

class LegalDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.process_data(dataframe, self.tokenizer)
    
    def process_data(self, data, tokenizer):
        """
        Tokenizes the full text of each row in the data and parses issue_data
        """
        tokenized_texts = []

        for text in data['full_text']:
            tokenized = tokenizer(
                text,
                max_length=100000,
                padding='max_length',
                return_tensors='pt',
                truncation=True
            )
            tokenized_texts.append(tokenized)

        data['tokenized_full_text'] = tokenized_texts
        data['issue_area'] = data['issue_area'].astype(float).astype(int)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'caseDisposition': torch.tensor(row['caseDisposition'], dtype=torch.float32),
            'tokenized_full_text': {
                'input_ids': row['tokenized_full_text']['input_ids'].squeeze(0),
                'attention_mask': row['tokenized_full_text']['attention_mask'].squeeze(0)
            },
            'issue_area': torch.tensor(row['issue_area'], dtype=torch.long)
        }

def load_data(args):
    """
    Reads the data from files and returns train, dev, and test sets as DataLoaders

    Args:
        args args object

    Returns:
        train_loader, dev_loader, test_loader: DataLoaders for train, dev, and test sets
    """
    train_path = os.path.join(args.data_path, f'train_{args.dataset}.parquet')
    dev_path = os.path.join(args.data_path, f'dev_{args.dataset}.parquet')
    test_path = os.path.join(args.data_path, f'test_{args.dataset}.parquet')

    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)
    test_df = pd.read_parquet(test_path)

    if args.model == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif args.model == 'legal_bert':
        tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained('chandar-lab/NeoBERT')

    train_dataset = LegalDataset(train_df.iloc[:100, :], tokenizer)
    dev_dataset = LegalDataset(dev_df, tokenizer)
    test_dataset = LegalDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Simple testing code to print tokenized text
    # for batch in train_loader:
    #     tokenized_text = batch['tokenized_full_text']
    #     print("Input IDs:", tokenized_text['input_ids'][0][:100])
    #     print("Attention Mask:", tokenized_text['attention_mask'][0][:10])

    #     decoded_text = tokenizer.decode(tokenized_text['input_ids'][0], skip_special_tokens=True)
    #     print("Decoded Text:", decoded_text[:100])
    #     quit()

    return train_loader, dev_loader, test_loader

def split_data(dataset):
    """
    Splits a data file into train dev test sets using 60 20 20 split
    dataset, '5k' or 'full': Which dataset to split"
    """
    csv_path = os.path.join('data', f'petitions_{dataset}.csv')
    parquet_path = os.path.join('data', f'petitions_{dataset}.parquet')
    train_path = os.path.join('data', f'train_{dataset}.parquet')
    dev_path = os.path.join('data', f'dev_{dataset}.parquet')
    test_path = os.path.join('data', f'test_{dataset}.parquet')

    df = pd.read_parquet(parquet_path)

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=1)

    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1)

    train_df.to_parquet(train_path)
    dev_df.to_parquet(dev_path)
    test_df.to_parquet(test_path)

def main():
    args = get_args()
    batch_size = args.batch_size
    data_path = args.data_path
    dataset = args.dataset

    #split_data(dataset)

    train_loader, dev_loader, test_loader = load_data(args)


    
if __name__ == "__main__":
    main()