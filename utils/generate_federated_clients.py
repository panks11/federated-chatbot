# utils/generate_federated_clients.py

import json
import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_client_data(data, client_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=[d['label'] for d in data])

    with open(os.path.join(out_dir, f"client_{client_id}_train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(out_dir, f"client_{client_id}_test.json"), 'w') as f:
        json.dump(test_data, f, indent=2)

def partition_iid(data, num_clients):
    random.shuffle(data)
    chunk_size = len(data) // num_clients
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_clients)]

def partition_non_iid(data, num_clients, shards_per_client=2):
    """
    Simulate non-IID by label-based shard assignment.
    """
    label_map = defaultdict(list)
    for sample in data:
        label_map[sample['label']].append(sample)

    all_shards = []
    for label, samples in label_map.items():
        random.shuffle(samples)
        num_shards = len(samples) // 20  # approx 20 samples/shard
        shards = [samples[i * 20:(i + 1) * 20] for i in range(num_shards)]
        all_shards.extend(shards)

    random.shuffle(all_shards)
    client_shards = [[] for _ in range(num_clients)]
    for i in range(num_clients * shards_per_client):
        client_id = i % num_clients
        client_shards[client_id].extend(all_shards[i])

    return client_shards

def generate_clients(data_path, out_dir, num_clients=5, iid=True):
    data = load_data(data_path)

    if iid:
        print("Generating IID client data...")
        clients_data = partition_iid(data, num_clients)
    else:
        print("Generating non-IID client data...")
        clients_data = partition_non_iid(data, num_clients)

    for client_id, client_data in enumerate(clients_data):
        save_client_data(client_data, client_id, out_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate federated client data splits.")
    parser.add_argument("--data_path", type=str, default="data/datasets/snips/train.json")
    parser.add_argument("--output_dir", type=str, default="data/datasets/snips/")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--iid", action="store_true", help="Set this flag for IID partitioning")

    args = parser.parse_args()

    generate_clients(args.data_path, args.output_dir, args.num_clients, iid=args.iid)
