import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mode

data = pd.read_csv("C:/Users/wiare/Desktop/musicfeatures/data.csv")
datas = data.drop('filename', 1)
X = np.array(datas.drop('label', 1))
Y = np.array(data['label'])
print(X)

# best n_components for ICA
experiment = []
max_accuracies = []
for j in range(0, 50):
    print("Experiment ", (j+1))
    n_component = []
    accuracy_scores = []
    for i in range(0, 28):
        print("     Classification ", (i+1))
        ica = FastICA(n_components=(i+1))
        X_copy = X.copy()
        Y_copy = Y.copy()
        ica.fit(X_copy)
        ica.transform(X_copy)
        classifier = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                          criterion="gini",
                                          random_state=8, n_jobs=4, bootstrap='false', max_features=20)
        X_train, X_test, Y_train, Y_test = train_test_split(X_copy, Y_copy, test_size=0.2, shuffle='true', stratify=Y_copy)
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        n_component.append((i+1))
        accuracy_scores.append(accuracy_score(Y_test, prediction))

    plt.plot(n_component, accuracy_scores, 'b')
    plt.title('Accuratezza di un classificatore al variare delle componenti della FastICA')
    plt.xlabel('n_component')
    plt.ylabel('Accuracy_score')
    plt.xlim(0.0, 29.0)
    plt.ylim(0.0, 1.0)
    plt.show()
    experiment.append((j+1))
    max_accuracies.append(accuracy_scores.index(max(accuracy_scores)))

plt.plot(experiment, max_accuracies, 'b')
plt.title('Numero componenti con accuratezze massime di ogni esperimento')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 51.0)
plt.ylim(0.0, 29.0)
plt.show()
print(mode(max_accuracies))
