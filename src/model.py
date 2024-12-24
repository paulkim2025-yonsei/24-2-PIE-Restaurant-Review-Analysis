# src/model.py

import torch
import torch.nn as nn

class LSTMAttentionModel(nn.Module):

    def __init__(self, i=300, h=32, sl=800):
        super(LSTMAttentionModel, self).__init__()
        self.sl = sl
        self.h = h
        self.lstm = nn.LSTM(i, h, batch_first=True)
        self.weight = nn.Linear(sl, sl)
        self.fc = nn.Linear(h * sl, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x: (batch_size, seq_len=800, emb_dim=300)
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size=32)
        # Attention-ish:
        p = out.permute(0, 2, 1)  # (batch_size, 32, 800)
        q = self.weight(p)        # (batch_size, 32, 800)
        q = q.permute(0, 2, 1)    # (batch_size, 800, 32)

        qk = out * q  # element-wise multiplication
        out_fc = self.fc(self.flatten(qk))
        return out_fc, qk
