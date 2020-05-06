import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import recall_score

def getData(path):
    data = pd.read_csv(path)
    datas = data.drop('filename', 1)
    X_data = np.array(datas.drop('label', 1))
    y_data = np.array(data['label'])
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle='true',stratify=y_data)
    return X_train, X_test, Y_train, Y_test


def validation(test, prediction):
    accuracy = accuracy_score(test, prediction)
    precision = precision_score(test, prediction, average='macro')
    recall = recall_score(test, prediction, average='macro')
    f1 = f1_score(test, prediction, average='macro')
    confusion = confusion_matrix(test, prediction)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1 measure:", f1)
    print("\n\n")
    print("Confusion:\n", confusion)
