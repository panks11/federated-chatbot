import pandas as pd
from i_preprocess import Features
import i_data_loader
import i_train
import model as m
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
train_dl, val_dl = i_data_loader.get_data_loader(x_train, y_train, x_val, y_val)
test_dl = i_data_loader.get_test_data_loader(x_test, y_test)
model = m.NNet(len(features.vectorizer.vocabulary_), 128, len(features.label_map))
print(model)
accuracy_list, loss_list, val_accuracy_list, val_loss_list = i_train.train(
    model, train_dl, val_dl, epochs=100
)
test_acc, test_loss, result_tups = i_train.eval(model, test_dl, return_preds=True)
i_train.plot_confusion_matrix(result_tups[0], result_tups[1], features.map_label)
print("Test Accuracy: ", test_acc, "Test Loss: ", test_loss)
print(classification_report(result_tups[0], result_tups[1]))
i_train.plot_graph([accuracy_list, val_accuracy_list], "accuracy")
i_train.plot_graph([loss_list, val_loss_list], "loss")
