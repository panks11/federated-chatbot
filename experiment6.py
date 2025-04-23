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


client0 = "data/snips/noniid/client0.csv"
client1 = "data/snips/noniid/client1.csv"
client2 = "data/snips/noniid/client2.csv"
client3 = "data/snips/noniid/client3.csv"
client4 = "data/snips/noniid/client4.csv"
client5 = "data/snips/noniid/client5.csv"
client6 = "data/snips/noniid/client6.csv"
client7 = "data/snips/noniid/client7.csv"

df = pd.read_csv(train_data)
x_train = df["text"]
y_train = df["intent"]

df = pd.read_csv(dev_data)
x_val = df["text"]
y_val = df["intent"]

df = pd.read_csv(test_data)
x_test = df["text"]
y_test = df["intent"]

df = pd.read_csv(client0)
client0_x = df["text"]
client0_y = df["intent"]

df = pd.read_csv(client1)
client1_x = df["text"]
client1_y = df["intent"]

df = pd.read_csv(client2)
client2_x = df["text"]
client2_y = df["intent"]

df = pd.read_csv(client3)
client3_x = df["text"]
client3_y = df["intent"]

df = pd.read_csv(client4)
client4_x = df["text"]
client4_y = df["intent"]

df = pd.read_csv(client5)
client5_x = df["text"]
client5_y = df["intent"]

df = pd.read_csv(client6)
client6_x = df["text"]
client6_y = df["intent"]

df = pd.read_csv(client7)
client7_x = df["text"]
client7_y = df["intent"]

features = Features(x_train, y_train, x_test, y_test, x_val, y_val)

x_train, y_train, x_test, y_test, x_val, y_val = features.get_tokenized_dataset()

train_dl, val_dl = idl.get_data_loader(
    x_train, y_train, x_val, y_val, batch_size=32, val_batch_size=64
)
test_dl = idl.get_test_data_loader(x_test, y_test)
# train = create_iid_data.split_dl(train_dl, n_clients=6)
val = create_iid_data.split_dl(val_dl, n_clients=8)
client0_x, client0_y = features.get_count_vect(client0_x, client0_y)
client1_x, client1_y = features.get_count_vect(client1_x, client1_y)
client2_x, client2_y = features.get_count_vect(client2_x, client2_y)
client3_x, client3_y = features.get_count_vect(client3_x, client3_y)
client4_x, client4_y = features.get_count_vect(client4_x, client4_y)
client5_x, client5_y = features.get_count_vect(client5_x, client5_y)
client6_x, client6_y = features.get_count_vect(client6_x, client6_y)
client7_x, client7_y = features.get_count_vect(client7_x, client7_y)

client0_dl = idl.get_test_data_loader(client0_x, client0_y)
client1_dl = idl.get_test_data_loader(client1_x, client1_y)
client2_dl = idl.get_test_data_loader(client2_x, client2_y)
client3_dl = idl.get_test_data_loader(client3_x, client3_y)
client4_dl = idl.get_test_data_loader(client4_x, client4_y)
client5_dl = idl.get_test_data_loader(client5_x, client5_y)
client6_dl = idl.get_test_data_loader(client6_x, client6_y)
client7_dl = idl.get_test_data_loader(client7_x, client7_y)


train = [
    client0_dl,
    client1_dl,
    client2_dl,
    client3_dl,
    client4_dl,
    client5_dl,
    client6_dl,
    client7_dl,
]

n_iter = 20

model_0 = m.TextClassificationModel(
    len(features.tokenizer.index_word), 64, len(features.label_map)
)
model_f, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = tr.FedProx(
    model_0, train, n_iter, val, epochs=5, lr=0.01, mu=0.3
)
test_acc, test_loss, result_tups = i_train.eval(model_f, test_dl, return_preds=True)
i_train.plot_confusion_matrix(result_tups[0], result_tups[1], features.map_label)
print("Test Accuracy: ", test_acc, "Test Loss: ", test_loss)
print(classification_report(result_tups[0], result_tups[1]))
tr.plot_acc_loss("FedAvg MNIST-iid Training ", train_loss_hist, train_acc_hist)
tr.plot_acc_loss("FedAvg MNIST-iid Validation ", test_loss_hist, test_acc_hist)
