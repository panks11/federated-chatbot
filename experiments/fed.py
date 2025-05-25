import random
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from data.preprocess import TextDataset, load_data
from models.lstm import LSTM
from utils.create_non_iid_clients import split_noniid_text, load_config




# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


# Utility operations for training and evaluation
def train_model(model, dataloader, optimizer, epochs=1):
    model.train()
    for _ in range(epochs):
        total_loss, total_samples = 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    return total_loss / total_samples

def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def copy_weights(target, source):
    for name in target:
        target[name].data.copy_(source[name].data)

def subtract_weights(target, a, b):
    for name in target:
        target[name].data.copy_(a[name].data - b[name].data)

def average_weights(targets, sources):
    for name in targets[0]:
        avg = torch.mean(torch.stack([s[name].data for s in sources]), dim=0)
        for target in targets:
            target[name].data.add_(avg)


def flatten_weights(params):
    return torch.cat([v.flatten() for v in params.values()])


# Federated client
def split_dataset(dataset, frac=0.8):
    n = len(dataset)
    n_train = int(n * frac)
    return torch.utils.data.random_split(dataset, [n_train, n - n_train])


class FederatedClient:
    def __init__(self, model_fn, optimizer_fn, dataset, cid, batch_size=128):
        self.id = cid
        self.model = model_fn().to(device)
        self.optimizer = optimizer_fn(self.model.parameters())
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.dW = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.W_old = {k: torch.zeros_like(v) for k, v in self.W.items()}

        train_data, eval_data = split_dataset(dataset)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_data, batch_size=batch_size)

    def synchronize(self, server):
        copy_weights(self.W, server.W)

    def update(self, epochs=1):
        copy_weights(self.W_old, self.W)
        train_model(self.model, self.train_loader, self.optimizer, epochs)
        subtract_weights(self.dW, self.W, self.W_old)

    def evaluate(self):
        return evaluate_model(self.model, self.eval_loader)


# Federated server
class FederatedServer:
    def __init__(self, model_fn, dataset):
        self.model = model_fn().to(device)
        self.W = {k: v for k, v in self.model.named_parameters()}
        self.dataset = dataset

    def evaluate(self):
        loader = DataLoader(self.dataset, batch_size=128)
        return evaluate_model(self.model, loader)

    def aggregate(self, clients):
        average_weights([self.W], [client.dW for client in clients])

def build_vocab_and_transform(dataset_list, tokenizer, max_len):
    def yield_tokens():
        for ds in dataset_list:
            for _, text in ds:
                yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(), min_freq=2, specials=['<pad>', '<sos>', '<eos>', '<unk>'], special_first=True)
    vocab.set_default_index(vocab['<unk>'])
    transform = T.Sequential(
        T.VocabTransform(vocab),
        T.AddToken(1, begin=True),
        T.Truncate(max_seq_len=max_len),
        T.AddToken(2, begin=False),
        T.ToTensor(padding_value=0)
    )
    return vocab, transform

def run_federated_training(config, client_dataframes):
    tokenizer = get_tokenizer("basic_english")
    client_datasets = [TextDataset(df) for df in client_dataframes]
    vocab, text_transform = build_vocab_and_transform(client_datasets, tokenizer, config['max_len'])

    def model_fn():
        return LSTM(
            vocab_size=len(vocab),
            output_size=config['num_classes'],
            num_layers=config['num_layers'],
            hidden_size=config['hidden_size']
        )

    def optimizer_fn(params):
        return optim.Adam(params, lr=config['learning_rate'])

    clients = [
        FederatedClient(model_fn, optimizer_fn, ds, cid=i)
        for i, ds in enumerate(client_datasets)
    ]

    full_dataset = torch.utils.data.ConcatDataset(client_datasets)
    server = FederatedServer(model_fn, full_dataset)

    for round in range(config['rounds']):
        print(f"\n--- Round {round+1} ---")

        for client in clients:
            client.synchronize(server)
            client.update(config['local_epochs'])

        server.aggregate(clients)
        round_acc = server.evaluate()
        print(f"Round {round+1} Accuracy: {round_acc:.4f}")


if __name__ == "__main__":
    config = load_config()
    dataframe_train, dataframe_test = load_data(config["data_path"])
    print(f"Train samples: {len(dataframe_train)}, Test samples: {len(dataframe_test)}")
    client_idcs = split_noniid_text(dataframe_train, alpha=1.0, n_clients=6)
    client_data = [dataframe_train.iloc[idcs].reset_index(drop=True) for idcs in client_idcs]

    print(f"Client data sizes: {[len(data) for data in client_data]}")
    # run_federated_training(config, client_data)
