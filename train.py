import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
from model import BertClassifier  # Il faut que model.py est dans le même dossier

# --- Config ---
config = {
    "model_name": "bert-base-multilingual-cased",
    "max_length": 259, # c'est le max_length du dataset utilisé (train_6.csv)
    "batch_size": 16,
    "lr": 2e-5,
    "epochs": 10,
    "num_labels": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- Dataset ---
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["premise"],
            row["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }

# --- Fonctions d'entraînement et validation ---
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    losses = []
    preds = []
    labels = []
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    accuracy = accuracy_score(labels, preds)
    return avg_loss, accuracy

def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)

            losses.append(loss.item())
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    accuracy = accuracy_score(labels, preds)
    return avg_loss, accuracy

# --- Fonction principale ---
def main():
    # Initialisation wandb
    wandb.init(project="bert-multilingual-entailment", config=config)
    cfg = wandb.config

    # Chargement des données
    df = pd.read_csv("C:/Users/MSI/Desktop/Deep Learning_2/bert-entailment-classifier/train_6.csv")
    df = df[["premise", "hypothesis", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_df, tokenizer, cfg.max_length)
    val_dataset = CustomDataset(val_df, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    # Modèle, optimizer, loss
    model = BertClassifier(cfg.model_name, cfg.num_labels).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.80
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, cfg.device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, cfg.device)

        # Log wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Sauvegarde du meilleur modèle
        torch.save(model.state_dict(), "C:/Users/MSI/Desktop/Deep Learning_2/bert-entailment-classifier/saved_model.pt") 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Le modèle sauvegardé a dépassé le seuil de performance souhaité \n"
            f"et présente une val_accuracy de : {best_val_acc:.4f}")

        else:
          print(f"Le modèle sauvegardé n'a pas atteint le seuil de performance de val_accuracy souhaité qui vaut {best_val_acc:.4f}. \n"
            f"Il présente une val_accuracy de : {val_acc:.4f}")
          
    wandb.finish()

if __name__ == "__main__":
    main()

