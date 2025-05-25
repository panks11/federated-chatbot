
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder

from data.preprocess import load_data

def load_config(path="config/base_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def split_noniid_text(dftrain, alpha, n_clients):
    le = LabelEncoder()
    dftrain['intent_label'] = le.fit_transform(dftrain['label'])
    train_labels = dftrain['intent_label'].values
    train_idcs = np.arange(len(train_labels))
    n_classes = train_labels.max() + 1

    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        splits = np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
        for i, idcs in enumerate(splits):
            client_idcs[i].extend(idcs.tolist())
    return client_idcs
    

if __name__ == "__main__":
    
    config = load_config()
    dataframe_train, dataframe_test = load_data(config["data_path"])
    print(f"Train samples: {len(dataframe_train)}, Test samples: {len(dataframe_test)}")
    client_idcs = split_noniid_text(dataframe_train, alpha=1.0, n_clients=6)
    client_data = [dataframe_train.iloc[idcs].reset_index(drop=True) for idcs in client_idcs]

    print(f"Client data sizes: {[len(data) for data in client_data]}")
    # Save the client data to files
    for i, data in enumerate(client_data):
        data.to_csv(f"data/client_{i}.csv", index=False)
        print(f"Client {i} data saved with {len(data)} samples.")