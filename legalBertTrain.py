import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModel, AdamW, AutoConfig
import argparse
from legalBertLoad import LegalDataset

# ---------------------------
# Collate function
# ---------------------------
def collate_fn(batch):
    compressed = torch.stack([item['compressed_embedding'] for item in batch])
    caseDisposition = torch.stack([(item['caseDisposition'] - 1).long() for item in batch])
    issue_area = torch.stack([(item['issue_area'] - 1).long() for item in batch])
    return {
        'compressed_embedding': compressed,
        'caseDisposition': caseDisposition,
        'issue_area': issue_area
    }

# ---------------------------
# Custom Model for Finetuning LegalBERT
# ---------------------------
class LegalBertFinetuner(nn.Module):
    def __init__(self, embedding_dim, num_issue_labels, num_disp_labels, dropout_prob=0.2):
        super(LegalBertFinetuner, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.issue_classifier = nn.Linear(embedding_dim, num_issue_labels)
        self.disposition_classifier = nn.Linear(embedding_dim, num_disp_labels)

    def forward(self, compressed_embedding):
        x = self.dropout(compressed_embedding)
        issue_logits = self.issue_classifier(x)
        disp_logits = self.disposition_classifier(x)
        return issue_logits, disp_logits

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_epoch(model, dataloader, optimizer, device, ce_loss_fn):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        compressed = batch['compressed_embedding'].to(device)
        labels_issue = batch['issue_area'].to(device)
        labels_disp = batch['caseDisposition'].to(device)

        optimizer.zero_grad()
        issue_logits, disp_logits = model(compressed)
        
        # Loss for issue_area (classification)
        loss_issue = ce_loss_fn(issue_logits, labels_issue)
        # Loss for caseDisposition (binary classification, use BCEWithLogitsLoss)
        loss_disp = ce_loss_fn(disp_logits, labels_disp)
        
        loss = loss_issue + loss_disp
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, ce_loss_fn):
    model.eval()
    total_loss = 0.0
    correct_issue = 0
    total_issue = 0
    correct_disp = 0
    total_disp = 0
    
    with torch.no_grad():
        for batch in dataloader:
            compressed = batch['compressed_embedding'].to(device)
            labels_issue = batch['issue_area'].to(device)
            labels_disp = batch['caseDisposition'].to(device)

            issue_logits, disp_logits = model(compressed)
            loss_issue = ce_loss_fn(issue_logits, labels_issue)
            loss_disp = ce_loss_fn(disp_logits, labels_disp)
            loss = loss_issue + loss_disp
            total_loss += loss.item()
            
            # Accuracy for issue_area classification
            preds_issue = torch.argmax(issue_logits, dim=1)
            correct_issue += (preds_issue == labels_issue).sum().item()
            total_issue += labels_issue.size(0)
            
            # Accuracy for binary caseDisposition (threshold at 0.5)
            preds_disp = torch.argmax(disp_logits, dim=1)
            correct_disp += (preds_disp == labels_disp).sum().item()
            total_disp += labels_disp.size(0)
            
    avg_loss = total_loss / len(dataloader)
    issue_acc = correct_issue / total_issue if total_issue > 0 else 0
    disp_acc = correct_disp / total_disp if total_disp > 0 else 0
    return avg_loss, issue_acc, disp_acc

# ---------------------------
# Main Finetuning Loop
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='train_dataset_5k_legal_bert.pt')
    parser.add_argument('--dev_dataset', type=str, default='dev_dataset_5k_legal_bert.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Set the number of classes for issue_area classification.
    parser.add_argument('--num_issue_labels', type=int, default=14)
    parser.add_argument('--num_disp_labels', type=int, default=10)
    args = parser.parse_args()
    
    # Load pre-saved datasets
    train_dataset = torch.load(args.train_dataset)
    dev_dataset = torch.load(args.dev_dataset)
    
    # Create DataLoaders with our collate function
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize our LegalBERT finetuning model.
    config = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
    embedding_dim = config.hidden_size
    model = LegalBertFinetuner(embedding_dim, args.num_issue_labels, args.num_disp_labels)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Define loss functions:
    # CrossEntropyLoss for multi-class classification of issue_area
    ce_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, ce_loss_fn)
        dev_loss, issue_acc, disp_acc = evaluate(model, dev_loader, device, ce_loss_fn)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Issue Acc: {issue_acc:.4f} | Disposition Acc: {disp_acc:.4f}")
    
    # Save the finetuned model state
    torch.save(model.state_dict(), "legal_bert_finetuned_with_embeddings.pt")
    
if __name__ == "__main__":
    main()
