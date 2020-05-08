import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


matplotlib.use('Agg')
data = pd.read_csv("C:/Users/wiare/Desktop/musicfeatures/data.csv")
datas = data.drop('filename', 1)
X = preprocessing.scale(np.array(datas.drop('label', 1)))
Y = np.array(data['label'])

'''
# best n_components for LDA
experiment = []
max_accuracies = []
for j in range(0, 20):
    print("Experiment", (j + 1))
    n_component = []
    accuracy_scores = []

    # Classification without LDA
    print("     Classification without LDA")
    KAccuracies = []
    X_copy = X.copy()
    Y_copy = Y.copy()
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X_copy):
        X_train, X_test = X_copy[train_index], X_copy[test_index]
        Y_train, Y_test = Y_copy[train_index], Y_copy[test_index]
        classifier = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                          criterion="gini",
                                          random_state=8, n_jobs=4, bootstrap='false', max_features=20)
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        KAccuracies.append(accuracy_score(Y_test, prediction))

    print("    ", KAccuracies)
    n_component.append(0)
    accuracy_scores.append(np.mean(KAccuracies))

    # Classification with LDA: every iteration increases n_component
    for i in range(1, 10):                                       #n_components cannot be larger than min(n_features, n_classes - 1). Using min(n_features, n_classes - 1) = min(28, 10 - 1) = 9 components.
        print("     Classification with LDA with", i, "features")
        lda = LinearDiscriminantAnalysis(n_components=i)
        X_copy = X.copy()
        Y_copy = Y.copy()
        X_copy = lda.fit(X_copy, Y_copy).transform(X_copy)
        KAccuracies = []

        # For every value of n_components, LDA is tested on five different split of the dataSet
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X_copy):
            X_train, X_test = X_copy[train_index], X_copy[test_index]
            Y_train, Y_test = Y_copy[train_index], Y_copy[test_index]
            classifier = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                              criterion="gini",
                                              random_state=8, n_jobs=4, bootstrap='false', max_features=i)
            classifier.fit(X_train, Y_train)
            prediction = classifier.predict(X_test)
            KAccuracies.append(accuracy_score(Y_test, prediction))

        print("    ", KAccuracies)
        n_component.append(i)
        accuracy_scores.append(np.mean(KAccuracies))
    print(accuracy_scores)
    experiment.append((j + 1))
    max_accuracies.append(accuracy_scores.index(max(accuracy_scores)))

    # Save accuracy graphics of every experiment
    plt.figure()
    plt.plot(n_component, accuracy_scores, 'b')
    plt.title('Accuratezza del classificatore al variare delle componenti della LDA. Esperimento ' + str(j+1))
    plt.xlabel('n_component')
    plt.ylabel('Accuracy_score')
    plt.xlim(0.0, 10.0)
    plt.ylim(0.0, 1.0)
    namePlot = "Accuracy Plot experiment " + str(j+1) + ".png"
    plt.savefig(namePlot, bbox_inches='tight')
    plt.close()

# Save graphic of the number of features with max accuracy in every experiment
plt.figure()
plt.plot(experiment, max_accuracies, 'b')
plt.title('Numero componenti con accuratezze massime di ogni esperimento')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 21.0)
plt.ylim(0.0, 10.0)
namePlot = "Number of LDA's components with max accuracy for every experiment.png"
plt.savefig(namePlot, bbox_inches='tight')
'''


# select solver
experiment = []
accuracies_svd = []
accuracies_eigen = []
components = 9
for i in range(1, 21):
    print("     Classification ", i)
    lda_svd = LinearDiscriminantAnalysis(n_components=components, solver='svd')
    lda_eigen = LinearDiscriminantAnalysis(n_components=components, solver='eigen')
    X_copy = X.copy()
    Y_copy = Y.copy()

    lda_svd.fit(X_copy, Y_copy)
    lda_eigen.fit(X_copy, Y_copy)

    KAccuracies_svd = []
    KAccuracies_eigen = []
    for k in range(0, 5):
        classifier_svd = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                          criterion="gini",
                                          random_state=8, n_jobs=4, bootstrap='false', max_features=components)
        classifier_eigen = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                            criterion="gini",
                                            random_state=8, n_jobs=4, bootstrap='false', max_features=components)

        X_train, X_test, Y_train, Y_test = train_test_split(X_copy, Y_copy, test_size=0.2, shuffle='true',
                                                            stratify=Y_copy)
        X_train_svd = lda_svd.transform(X_train)
        X_test_svd = lda_svd.transform(X_test)
        X_train_eigen = lda_eigen.transform(X_train)
        X_test_eigen = lda_eigen.transform(X_test)
        classifier_svd.fit(X_train_svd, Y_train)
        classifier_eigen.fit(X_train_eigen, Y_train)
        prediction_svd = classifier_svd.predict(X_test_svd)
        prediction_eigen = classifier_eigen.predict(X_test_eigen)
        KAccuracies_svd.append(accuracy_score(Y_test, prediction_svd))
        KAccuracies_eigen.append(accuracy_score(Y_test, prediction_eigen))

    print("SVD ", KAccuracies_svd)
    print("EIGEN ", KAccuracies_eigen)
    accuracies_svd.append(np.mean(KAccuracies_svd))
    accuracies_eigen.append(np.mean(KAccuracies_eigen))

    experiment.append(i)

print("Accuracies svd: ", accuracies_svd)
print("Accuracies eigen: ", accuracies_eigen)
plt.plot(experiment, accuracies_svd, 'r')
plt.plot(experiment, accuracies_eigen, 'g')
plt.title('Andamento precisione con i diversi tipi di solver')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 21.0)
plt.ylim(0.0, 1.0)
red_patch = matplotlib.patches.Patch(color='red', label='SVD')
green_patch = matplotlib.patches.Patch(color='green', label='EIGEN')
plt.legend(handles=[red_patch, green_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.savefig("LDA's accuracies with different values of solver", bbox_inches='tight')
