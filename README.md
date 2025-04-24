
# Federated Text Classification 🧠📡

A modular framework for **centralized** and **federated learning** on **text classification tasks** using **ANN** and **LSTM** models. Supports both **IID** and **non-IID** data distributions across clients.

---

## 🚀 Features

- ✅ Centralized & Federated Training
- 🧠 Support for ANN and LSTM models
- 🔁 FedAvg-based aggregation
- 📊 Client-wise evaluation metrics
- 🔧 Configurable via YAML files
- 📦 SNIPS-style dataset loader

---

## 🗂 Folder Structure

```
federated_text_classification/
├── config/                  # YAML configs (base + client)
├── data/                   # SNIPS-style data and preprocessing
├── models/                 # ANN and LSTM models
├── trainer/                # Centralized and federated trainers
├── utils/                  # Logging, config, metrics, client generation
├── experiments/            # Experiment runners
└── outputs/                # Logs, checkpoints, and results
```

---

## 🛠️ Setup

```bash
git clone https://github.com/your-org/federated-text-classification.git
cd federated-text-classification
pip install -r requirements.txt
```

---

## 📚 Usage

### 1. Preprocess SNIPS Data

```bash
python utils/generate_federated_clients.py --iid       # For IID partitioning
python utils/generate_federated_clients.py             # For non-IID partitioning
```

### 2. Run Centralized Training

```bash
python experiments/run_centralized.py
```

### 3. Run Federated Training

```bash
# Using ANN
python experiments/run_federated_ann.py

# Using LSTM
python experiments/run_federated_lstm.py
```

---

## ⚙️ Configuration

All configs are stored in `config/`:

- `base_config.yaml`: global defaults
- `client_{i}_config.yaml`: per-client overrides

You can customize:
- Model type (`ann`, `lstm`)
- Number of clients, rounds, local epochs
- Learning rate, batch size, and more

---

## 📈 Sample Output

```
--- Federated Round 1 ---
Client 0 Accuracy: 0.78
Client 1 Accuracy: 0.82
Average Accuracy After Round 1: 0.80
```

---

## 📋 TODO

- [ ] Add support for more NLP datasets
- [ ] Add federated metrics (per-class accuracy, fairness, etc.)
- [ ] Integrate FedProx, FedOpt optimizers
- [ ] Add UI dashboard for training monitoring

---

## 👩‍💻 Authors

- **Pankhuri Kulshrestha** – [@pankhuri-k](https://github.com/pankhuri-k)

---

## 📄 License

MIT License. See `LICENSE` file for details.
