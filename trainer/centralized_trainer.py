# trainer/centralized_trainer.py

import torch
from torch.utils.data import DataLoader
from utils.metrics import accuracy


class CentralizedTrainer:
    def __init__(self, model, train_dataset, test_dataset, config, device):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"])
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.epochs = config["epochs"]
        self.log_interval = config["logging"]["log_interval"]

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for step, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

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

    def run(self, logger=None):
        for epoch in range(self.epochs):
            loss = self.train_one_epoch()
            acc = self.evaluate()
            msg = f"[Epoch {epoch+1}/{self.epochs}] Loss: {loss:.4f}, Accuracy: {acc:.4f}"
            print(msg)
            if logger:
                logger.info(msg)
