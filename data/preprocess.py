# data/preprocess.py
import pandas as pd
import json
import os
from collections import Counter
from torch.utils.data import Dataset
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
    def __init__(self, dataframe: pd.DataFrame):
        # Expect dataframe to have 'label' and 'text' columns
        self.label2idx = {label: idx for idx, label in enumerate(sorted(dataframe['label'].unique()))}
        self.labels = dataframe['label'].apply(lambda x: self.label2idx[str(x).strip()]).tolist()
        self.texts = dataframe['text'].astype(str).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.texts[idx]


def load_data(data_set_root):
    dataframe_train = pd.read_csv(os.path.join(data_set_root,"train.csv"), header=None, names=["text","slot", "label"])
    dataframe_test = pd.read_csv(os.path.join(data_set_root,"test.csv"), header=None, names=["text", "slot", "label"])
    return dataframe_train, dataframe_test



