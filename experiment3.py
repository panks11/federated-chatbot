import create_iid_data
import pandas as pd
import i_data_loader as idl
from i_preprocess import Features
import train as tr
import model as m
import i_train
from sklearn.metrics import classification_report


dev_data = "data/snips/dev.csv"
test_data = "data/snips/test.csv"
train_data = "data/snips/train.csv"

df = pd.read_csv(train_data)
x_train = df["text"]
y_train = df["intent"]

df = pd.read_csv(dev_data)
x_val = df["text"]
y_val = df["intent"]

df = pd.read_csv(test_data)
x_test = df["text"]
y_test = df["intent"]

features = Features(x_train, y_train, x_test, y_test, x_val, y_val)
x_train, y_train, x_test, y_test, x_val, y_val = features.get_count_vect_dataset()
train_dl, val_dl = idl.get_data_loader(
    x_train, y_train, x_val, y_val, batch_size=32, val_batch_size=64
)
test_dl = idl.get_test_data_loader(x_test, y_test)
train = create_iid_data.split_dl(train_dl, n_clients=6)
val = create_iid_data.split_dl(val_dl, n_clients=6)

n_iter = 20

model_0 = m.NNet(len(features.vectorizer.vocabulary_), 128, len(features.label_map))
model_f, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = tr.FedProx(
    model_0, train, n_iter, val, epochs=5, lr=0.001, mu=0
)
test_acc, test_loss, result_tups = i_train.eval(model_f, test_dl, return_preds=True)
i_train.plot_confusion_matrix(result_tups[0], result_tups[1], features.map_label)
print("Test Accuracy: ", test_acc, "Test Loss: ", test_loss)
print(classification_report(result_tups[0], result_tups[1]))
tr.plot_acc_loss("FedAvg MNIST-iid Training ", train_loss_hist, train_acc_hist)
tr.plot_acc_loss("FedAvg MNIST-iid Validation ", test_loss_hist, test_acc_hist)
