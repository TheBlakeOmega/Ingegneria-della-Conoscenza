import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def getData(path):
    data = pd.read_csv(path)
    data = data.drop('filename', 1)
    data = data.drop('tempo', 1)
    datas = np.split(data, [27], axis=1)
    train = datas[0]
    test = datas[1]
    X_train, X_test, Y_train, Y_test = train_test_split(train, test, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


def makePCA(train, test):
    print("TRAIN SIZE BEFORE PCA:", train.shape)
    pca = PCA(n_components=5, whiten=True)
    pca.fit(train)
    train = pca.transform(train)
    print("TRAIN SIZE AFTER PCA:", train.shape)
    test = pca.transform(test)
    return train, test, pca









