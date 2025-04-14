import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from transformers import BertTokenizerFast
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

SPLIT_DATA = True
GET_LOADERS = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='5k', choices=['5k', 'full'])
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'legal_bert', 'neo_bert'])
    args = parser.parse_args()
    return args


# From csv: docket_number,year,issue_area,caseDisposition,full_text
class LegalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, compressor_model_name='bert-base-uncased', 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 embedding_cache_path=None, recalculate=False):
        self.tokenizer = tokenizer
        self.device = device
        self.data = dataframe.copy()
        
        # Try to load precomputed embeddings if they exist
        if embedding_cache_path and os.path.exists(embedding_cache_path) and not recalculate:
            print(f"Loading precomputed embeddings from {embedding_cache_path}")
            self.precomputed_embeddings = torch.load(embedding_cache_path)
            if len(self.precomputed_embeddings) != len(self.data):
                print(f"Warning: Number of cached embeddings ({len(self.precomputed_embeddings)}) " 
                      f"doesn't match dataframe length ({len(self.data)}). Recomputing...")
                recalculate = True
            else:
                print("Successfully loaded precomputed embeddings")
                return
                
        # Load the encoder model
        print(f"Loading encoder model: {compressor_model_name}")
        self.encoder = AutoModel.from_pretrained(compressor_model_name).to(device)
        self.encoder.eval()
        
        # Precompute embeddings in batches
        self.precomputed_embeddings = self.batch_compute_embeddings()
        
        # Save embeddings for future use if path is provided
        if embedding_cache_path:
            print(f"Saving embeddings to {embedding_cache_path}")
            torch.save(self.precomputed_embeddings, embedding_cache_path)
            
    def batch_compute_embeddings(self, batch_size=32):
        print("Precomputing embeddings in batches...")
        all_embeddings = []
        
        # Create a temporary dataloader for batch processing
        # Note: We can't use self as the dataset here since __getitem__ would call this function
        temp_dataset = [(i, row['full_text']) for i, row in self.data.iterrows()]
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]))
        
        for _, texts in tqdm(temp_loader, desc="Computing embeddings"):
            batch_embeddings = self.batch_encode_and_compress(texts)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    def batch_encode_and_compress(self, texts):
        batch_embeddings = []
        max_len = self.tokenizer.model_max_length
        
        # Process each text in the batch
        for text in texts:
            # Tokenize text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunk_size = max_len - 2  # Reserve tokens for [CLS] and [SEP]
            chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
            
            # Handle empty text
            if not chunks:
                batch_embeddings.append(torch.zeros(self.encoder.config.hidden_size))
                continue
                
            # Process chunks in mini-batches for longer texts
            text_embeddings = []
            for i in range(0, len(chunks), 8):  # Process 8 chunks at a time
                mini_batch = chunks[i:i+8]
                mini_batch_inputs = []
                
                for chunk in mini_batch:
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(chunk)
                    if len(input_ids) > max_len:
                        input_ids = input_ids[:max_len]
                    mini_batch_inputs.append(input_ids)
                
                # Pad sequences to same length
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(ids) for ids in mini_batch_inputs], 
                    batch_first=True, 
                    padding_value=self.tokenizer.pad_token_id
                ).to(self.device)
                
                # Create attention mask
                attention_mask = (padded_inputs != self.tokenizer.pad_token_id).float().to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.encoder(input_ids=padded_inputs, attention_mask=attention_mask)
                    chunk_embeddings = outputs.pooler_output
                
                text_embeddings.append(chunk_embeddings)
            
            # Flatten mini-batch results and average across chunks
            all_chunk_embeddings = torch.cat(text_embeddings) if len(text_embeddings) > 1 else text_embeddings[0]
            avg_embedding = all_chunk_embeddings.mean(dim=0)
            batch_embeddings.append(avg_embedding.cpu())
            
        return batch_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Retrieve the precomputed embedding
        compressed_embedding = self.precomputed_embeddings[idx]
        return {
            'compressed_embedding': compressed_embedding,
            'caseDisposition': torch.tensor(row['caseDisposition'], dtype=torch.long),
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
    train_path = os.path.join(args.data_path, f'train_{args.dataset}2.parquet')
    dev_path = os.path.join(args.data_path, f'dev_{args.dataset}2.parquet')
    test_path = os.path.join(args.data_path, f'test_{args.dataset}2.parquet')

    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)
    test_df = pd.read_parquet(test_path)

    if args.model == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model_name = 'bert-base-uncased'
    elif args.model == 'legal_bert':
        tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        model_name = 'nlpaueb/legal-bert-base-uncased'
    else:
        tokenizer = AutoTokenizer.from_pretrained('chandar-lab/NeoBERT')
        model_name = 'chandar-lab/NeoBERT'

    # Define embedding cache paths
    train_cache = f"embeddings_{args.dataset}_{args.model}_train.pt"
    dev_cache = f"embeddings_{args.dataset}_{args.model}_dev.pt"
    test_cache = f"embeddings_{args.dataset}_{args.model}_test.pt"

    # Create datasets with caching
    train_dataset = LegalDataset(train_df, tokenizer, model_name, embedding_cache_path=train_cache)
    dev_dataset = LegalDataset(dev_df, tokenizer, model_name, embedding_cache_path=dev_cache)
    test_dataset = LegalDataset(test_df, tokenizer, model_name, embedding_cache_path=test_cache)

    # Save the datasets
    torch.save(train_dataset, f"train_dataset_{args.dataset}_{args.model}.pt")
    torch.save(dev_dataset, f"dev_dataset_{args.dataset}_{args.model}.pt")
    torch.save(test_dataset, f"test_dataset_{args.dataset}_{args.model}.pt")

def return_loaders(args):
    """
    Returns train, dev, and test DataLoaders
    """
    train_dataset = torch.load(f"train_dataset_{args.dataset}_{args.model}.pt")
    dev_dataset = torch.load(f"dev_dataset_{args.dataset}_{args.model}.pt")
    test_dataset = torch.load(f"test_dataset_{args.dataset}_{args.model}.pt")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader

def split_data(dataset):
    """
    Splits a data file into train dev test sets using 60 20 20 split
    dataset, '5k' or 'full': Which dataset to split"
    """
    csv_path = os.path.join('data', f'preliminary_petitions.csv')
    train_path = os.path.join('data', f'train_{dataset}2.parquet')
    dev_path = os.path.join('data', f'dev_{dataset}2.parquet')
    test_path = os.path.join('data', f'test_{dataset}2.parquet')

    df = pd.read_csv(csv_path)

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=1)

    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1)

    train_df.to_parquet(train_path)
    dev_df.to_parquet(dev_path)
    test_df.to_parquet(test_path)

def main():
    args = get_args()
    dataset = args.dataset

    if SPLIT_DATA:
        split_data(dataset)

    load_data(args)

    if GET_LOADERS:
        train_loader, dev_loader, test_loader = return_loaders(args)
    
if __name__ == "__main__":
    main()
