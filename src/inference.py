# src/inference.py

import torch
import numpy as np
from .model import LSTMAttentionModel

def load_model(model_path, device='cpu'):

    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def predict_label(model, embs, threshold=0.5, device='cpu'):

    with torch.no_grad():
        x = torch.tensor(embs).float().to(device)
        output, _ = model(x)
        prob = torch.sigmoid(output)
        label = int((prob > threshold).item())
    return label
