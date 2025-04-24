# trainer/federated_client.py

import torch
from torch.utils.data import DataLoader
from utils.metrics import accuracy
from copy import deepcopy

class FederatedClient:
    def __init__(self, client_id, model, train_dataset, test_dataset, config, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

        self.local_epochs = config["epochs"]
        self.lr = config["learning_rate"]
        self.weight_decay = config["weight_decay"]

    def get_model_weights(self):
        return deepcopy(self.model.state_dict())

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train_local(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.local_epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_model_weights()

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        return accuracy(all_labels, all_preds)
