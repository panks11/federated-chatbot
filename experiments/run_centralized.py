# experiments/run_centralized.py

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from data.preprocess import load_data, TextDataset
from models.lstm import LSTM
from utils.logger import get_logger
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import torchtext; torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T


def load_config(path="config/base_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def main():
    config = load_config()
    logger = get_logger("centralized", config["logging"]["log_dir"])
    device = torch.device(config["device"])
    batch_size = config["batch_size"]
    nepochs = config["epochs"]
    num_layers = config["lstm"]["num_layers"]
    hidden_size = config["lstm"]["hidden_size"]
    max_len = config["max_len"]

    dataframe_train, dataframe_test = load_data(config["data_path"])
    logger.info(f"Train samples: {len(dataframe_train)}, Test samples: {len(dataframe_test)}")

    dataset_train = TextDataset(dataframe_train)
    dataset_test = TextDataset(dataframe_test)



    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)




    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset_train),  # Tokenized data iterator
        min_freq=2,  # Minimum frequency threshold for token inclusion
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],  # Special case tokens
        special_first=True  # Place special tokens first in the vocabulary
    )
    vocab.set_default_index(vocab['<unk>'])

            # Define a text transformation pipeline using TorchText Sequential Transform
    text_tranform = T.Sequential(
    # Convert the sentences to indices based on the given vocabulary
    T.VocabTransform(vocab=vocab),
    # Add <sos> at the beginning of each sentence. 1 is used because the index for <sos> in the vocabulary is 1.
    T.AddToken(1, begin=True),
    # Crop the sentence if it is longer than the max length
    T.Truncate(max_seq_len=max_len),
    # Add <eos> at the end of each sentence. 2 is used because the index for <eos> in the vocabulary is 2.
    T.AddToken(2, begin=False),
    # Convert the list of lists to a tensor. This also pads a sentence with the <pad> token if it is shorter than the max length,
    # ensuring that all sentences are the same length.
    T.ToTensor(padding_value=0))

    text_tokenizer = lambda batch: [tokenizer(x) for x in batch]

    # Select model
    num_classes =  len(dataset_train.label2idx)
    if config["model_type"] == "lstm":
        lstm_classifier = LSTM(vocab_size=len(vocab), output_size=num_classes, 
                        num_layers=num_layers, hidden_size=hidden_size).to(device)
    else:
       pass
   
    optimizer = optim.Adam(lstm_classifier.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

     # Initialize lists to store training and test loss, as well as accuracy
    training_loss_logger = []
    test_loss_logger = []
    training_acc_logger = []
    test_acc_logger = []

    pbar = trange(0, nepochs, leave=False, desc="Epoch")

    # Initialize training and test accuracy
    train_acc = 0
    test_acc = 0

    # Loop through each epoch
    for epoch in pbar:
        # Update progress bar description with current accuracy
        pbar.set_postfix_str('Accuracy: Train %.2f%%, Test %.2f%%' % (train_acc * 100, test_acc * 100))
        
        # Set model to training mode
        lstm_classifier.train()
        steps = 0
        
        # Iterate through training data loader
        for label, text in tqdm(data_loader_train, desc="Training", leave=False):
            bs = label.shape[0]
            
            # Tokenize and transform text to tensor, move to device
            text_tokens = text_tranform(text_tokenizer(text)).to(device)
            label = label.to(device)
            
            # label = (label-1).to(device)
            # Initialize hidden and memory states
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
            memory = torch.zeros(num_layers, bs, hidden_size, device=device)
            
            # Forward pass through the model
            pred, hidden, memory = lstm_classifier(text_tokens, hidden, memory)

            # Calculate the loss
            loss = loss_fn(pred[:, -1, :], label)
                
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Append training loss to logger
            training_loss_logger.append(loss.item())
            
            # Calculate training accuracy
            train_acc += (pred[:, -1, :].argmax(1) == label).sum()
            steps += bs
            
        # Calculate and append training accuracy for the epoch
        train_acc = (train_acc/steps).item()
        training_acc_logger.append(train_acc)
        
        # Set model to evaluation mode
        lstm_classifier.eval()
        steps = 0
        
        # Iterate through test data loader
        with torch.no_grad():
            for label, text in tqdm(data_loader_test, desc="Testing", leave=False):
                bs = label.shape[0]
                # Tokenize and transform text to tensor, move to device
                text_tokens = text_tranform(text_tokenizer(text)).to(device)
                label = label.to(device)
                # label = (label - 1).to(device)


                # Initialize hidden and memory states
                hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
                memory = torch.zeros(num_layers, bs, hidden_size, device=device)
                
                # Forward pass through the model
                pred, hidden, memory = lstm_classifier(text_tokens, hidden, memory)

                # Calculate the loss
                loss = loss_fn(pred[:, -1, :], label)
                test_loss_logger.append(loss.item())

                # Calculate test accuracy
                test_acc += (pred[:, -1, :].argmax(1) == label).sum()
                steps += bs

            # Calculate and append test accuracy for the epoch
            test_acc = (test_acc/steps).item()
            test_acc_logger.append(test_acc)

            plt.figure(figsize=(10, 5))
            plt.plot(np.linspace(0, nepochs, len(training_acc_logger)), training_acc_logger, label="Train")
            plt.plot(np.linspace(0, nepochs, len(test_acc_logger)), test_acc_logger, label="Test")

            plt.legend()
            plt.title("Training Vs Test Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")

            # Save the plot as a PNG file
            plt.savefig("training_vs_test_accuracy.png", dpi=300, bbox_inches='tight')

            # Show the plot (optional)
            plt.show()

if __name__ == "__main__":
    main()
