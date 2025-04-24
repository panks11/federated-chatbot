# models/ann.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, output_dim, dropout=0.5):
        super(ANNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        layers = []
        input_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)             # [batch_size, seq_len, embedding_dim]
        pooled = embedded.mean(dim=1)            # [batch_size, embedding_dim]
        hidden = self.hidden_layers(pooled)      # [batch_size, hidden_dims[-1]]
        return self.output_layer(hidden)         # [batch_size, output_dim]
