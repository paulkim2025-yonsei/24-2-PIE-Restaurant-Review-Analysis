# src/train.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .model import LSTMAttentionModel

class AdvDataset(Dataset):

    def __init__(self, emb, label):
        self.emb = torch.tensor(np.array(emb)).float()
        self.y = torch.tensor(np.array(label)).float()

    def __getitem__(self, idx):
        return self.emb[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

def O2Label(logit):

    prob = torch.sigmoid(logit)
    return (prob > 0.5).squeeze().int()

def train_model(emb, label, save_weights_path, save_model_path, epochs=1000):

    X_train, X_test, y_train, y_test = train_test_split(
        emb, label, test_size=0.2, stratify=label, random_state=0
    )

    train_ds = AdvDataset(X_train, y_train)
    test_ds = AdvDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAttentionModel(i=300, h=32, sl=800).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float('inf')
    patience = 0
    best_model = None
    best_epoch = 0

    for e in tqdm(range(epochs)):
        # train
        model.train()
        train_loss_val = 0.0
        train_cr = 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss_val += loss.item()
            preds = O2Label(output)
            train_cr += (preds == y_batch).sum().item()

        train_loss_val /= len(train_ds)
        train_acc = train_cr / len(train_ds)

        # eval
        model.eval()
        test_loss_val = 0.0
        test_cr = 0
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output, _ = model(x_batch)
                loss = criterion(output.squeeze(), y_batch)
                test_loss_val += loss.item()
                preds = O2Label(output)
                test_cr += (preds == y_batch).sum().item()

        test_loss_val /= len(test_ds)
        test_acc = test_cr / len(test_ds)

        # Early stopping or best model tracking
        if test_loss_val < best_test_loss:
            best_test_loss = test_loss_val
            best_model = model
            best_epoch = e
            patience = 0
        else:
            patience += 1

        if patience >= 100:
            lr *= 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if e % 50 == 0:
            print(f"[Epoch {e}/{epochs}] "
                  f"Train Loss={train_loss_val:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss_val:.4f}, Test Acc={test_acc:.4f}")

        if test_loss_val <= 0.0001:
            break

    torch.save(best_model.state_dict(), save_weights_path)
    torch.save(best_model, save_model_path)
    print(f"Best model saved at epoch {best_epoch} with test loss {best_test_loss:.4f}.")
    return best_model
