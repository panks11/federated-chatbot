# experiments/run_centralized.py

import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.preprocess import load_data, Vocabulary, TextDataset
from models.ann import ANNModel
from models.lstm import LSTMModel
from utils.metrics import accuracy
from utils.logger import get_logger

def load_config(path="config/base_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return accuracy(all_labels, all_preds)

def main():
    config = load_config()
    logger = get_logger("centralized", config["logging"]["log_dir"])
    device = torch.device(config["device"])

    # Load and prepare dataset
    train_data, train_labels, label2id = load_data(os.path.join(config["data_path"], "train.json"))
    test_data, test_labels, _ = load_data(os.path.join(config["data_path"], "test.json"))

    vocab = Vocabulary(max_size=config["vocab_size"])
    vocab.build_vocab(train_data)

    train_dataset = TextDataset(train_data, train_labels, vocab, config["max_seq_length"])
    test_dataset = TextDataset(test_data, test_labels, vocab, config["max_seq_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Select model
    num_classes = len(label2id)
    if config["model_type"] == "lstm":
        model = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dim=config["lstm"]["hidden_dim"],
            output_dim=num_classes,
            num_layers=config["lstm"]["num_layers"],
            dropout=config["lstm"]["dropout"]
        )
    else:
        model = ANNModel(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dims=config["ann"]["hidden_dims"],
            output_dim=num_classes,
            dropout=config["ann"]["dropout"]
        )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        logger.info(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
