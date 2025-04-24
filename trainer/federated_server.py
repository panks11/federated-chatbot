# trainer/federated_server.py

import torch
from copy import deepcopy
from utils.metrics import accuracy

class FederatedServer:
    def __init__(self, global_model, clients, config, device, logger=None):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.logger = logger

        self.rounds = config["federated"]["rounds"]
        self.clients_per_round = config["federated"]["clients_per_round"]

    def average_weights(self, weights_list):
        avg_weights = deepcopy(weights_list[0])
        for key in avg_weights:
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = avg_weights[key] / len(weights_list)
        return avg_weights

    def evaluate_global_model(self):
        accs = []
        for client in self.clients:
            client.set_model_weights(self.global_model.state_dict())
            acc = client.evaluate()
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)
        return avg_acc, accs

    def run(self):
        for rnd in range(1, self.rounds + 1):
            if self.logger:
                self.logger.info(f"--- Federated Round {rnd} ---")

            selected_clients = self.clients[:self.clients_per_round]  # Simple selection
            local_weights = []

            for client in selected_clients:
                client.set_model_weights(self.global_model.state_dict())
                weights = client.train_local()
                local_weights.append(weights)

            # Aggregate
            aggregated_weights = self.average_weights(local_weights)
            self.global_model.load_state_dict(aggregated_weights)

            # Evaluation
            avg_acc, client_accs = self.evaluate_global_model()
            log_msg = f"[Round {rnd}] Avg Accuracy: {avg_acc:.4f}"
            if self.logger:
                self.logger.info(log_msg)
                for idx, acc in enumerate(client_accs):
                    self.logger.info(f"Client {idx} Accuracy: {acc:.4f}")
            else:
                print(log_msg)
