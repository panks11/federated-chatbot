# models/lstm.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)             # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)        # [batch_size, seq_len, hidden_dim]
        last_hidden = lstm_out[:, -1, :]         # [batch_size, hidden_dim]
        out = self.dropout(last_hidden)
        return self.fc(out)                      # [batch_size, output_dim]
