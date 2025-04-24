# data/preprocess.py

import json
import os
from collections import Counter
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import re


class Vocabulary:
    def __init__(self, max_size=10000, min_freq=1, specials=["<pad>", "<unk>"]):
        self.max_size = max_size
        self.min_freq = min_freq
        self.specials = specials
        self.word2idx = {}
        self.idx2word = []

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        most_common = [
            word for word, freq in counter.items()
            if freq >= self.min_freq
        ][:self.max_size - len(self.specials)]

        self.idx2word = self.specials + most_common
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)

    def tokenize(self, text):
        # Basic tokenization (can replace with nltk, spacy, etc.)
        return re.findall(r'\b\w+\b', text.lower())

    def numericalize(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

    def pad_sequence(self, tokens, max_len):
        padded = tokens[:max_len] + [self.word2idx["<pad>"]] * max(0, max_len - len(tokens))
        return padded


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_seq_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.vocab.tokenize(self.texts[idx])
        indices = self.vocab.numericalize(tokens)
        padded = self.vocab.pad_sequence(indices, self.max_seq_length)
        return torch.tensor(padded), torch.tensor(self.labels[idx])


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    texts = [entry["text"] for entry in data]
    labels = [entry["label"] for entry in data]

    label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    labels = [label2id[label] for label in labels]

    return texts, labels, label2id


def split_and_save(texts, labels, output_dir, test_size=0.2):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, stratify=labels
    )

    os.makedirs(output_dir, exist_ok=True)

    def save_json(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    save_json([{"text": t, "label": l} for t, l in zip(train_texts, train_labels)],
              os.path.join(output_dir, "train.json"))
    save_json([{"text": t, "label": l} for t, l in zip(test_texts, test_labels)],
              os.path.join(output_dir, "test.json"))


def prepare_snips_dataset(raw_path, output_path, vocab_size=10000, max_seq_length=50):
    texts, labels, label2id = load_data(raw_path)

    vocab = Vocabulary(max_size=vocab_size)
    vocab.build_vocab(texts)

    split_and_save(texts, labels, output_path)

    return vocab, label2id
