# experiments/run_federated_lstm.py

import os
import yaml
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy

from data.preprocess import load_data, Vocabulary, TextDataset
from models.lstm import LSTMModel
from utils.metrics import accuracy
from utils.logger import get_logger

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_client_config_paths(base_path, num_clients):
    return [os.path.join(base_path, f"client_{i}_config.yaml") for i in range(num_clients)]

def train_local(model, dataloader, criterion, optimizer, device):
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

def average_weights(state_dicts):
    avg_state_dict = deepcopy(state_dicts[0])
    for key in avg_state_dict:
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict

def main():
    base_config = load_yaml("config/base_config.yaml")
    logger = get_logger("fed_lstm", base_config["logging"]["log_dir"])
    device = torch.device(base_config["device"])

    client_config_paths = get_client_config_paths("config", base_config["federated"]["num_clients"])

    client_models = []
    client_data_loaders = []

    for path in client_config_paths:
        config = load_yaml(path)

        train_data, train_labels, label2id = load_data(config["train_data_path"])
        test_data, test_labels, _ = load_data(config["test_data_path"])

        vocab = Vocabulary(max_size=config["vocab_size"])
        vocab.build_vocab(train_data)

        train_dataset = TextDataset(train_data, train_labels, vocab, config["max_seq_length"])
        test_dataset = TextDataset(test_data, test_labels, vocab, config["max_seq_length"])

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

        model = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dim=config["lstm"]["hidden_dim"],
            output_dim=len(label2id),
            num_layers=config["lstm"]["num_layers"],
            dropout=config["lstm"]["dropout"]
        ).to(device)

        client_models.append((model, train_loader, test_loader, vocab))

    criterion = torch.nn.CrossEntropyLoss()

    global_model = deepcopy(client_models[0][0])
    global_weights = global_model.state_dict()

    for rnd in range(base_config["federated"]["rounds"]):
        logger.info(f"--- Federated Round {rnd+1} ---")
        local_weights = []

        for i, (model, train_loader, _, _) in enumerate(client_models):
            model.load_state_dict(global_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=base_config["learning_rate"])

            for _ in range(base_config["federated"]["local_epochs"]):
                train_local(model, train_loader, criterion, optimizer, device)

            local_weights.append(deepcopy(model.state_dict()))

        # Aggregate
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Evaluate
        total_acc = 0
        for i, (_, _, test_loader, _) in enumerate(client_models):
            global_model.eval()
            acc = evaluate(global_model, test_loader, device)
            total_acc += acc_
