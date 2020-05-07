import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


matplotlib.use('Agg')
data = pd.read_csv("C:/Users/wiare/Desktop/musicfeatures/data.csv")
datas = data.drop('filename', 1)
X = preprocessing.scale(np.array(datas.drop('label', 1)))
Y = np.array(data['label'])


# best n_components for PCA
experiment = []
max_accuracies = []

for j in range(0, 20):
    print("Experiment", (j + 1))
    n_component = []
    accuracy_scores = []

    # Classification without PCA
    print("     Classification without PCA")
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

    # Classification with PCA: every iteration increases n_component
    for i in range(1, 29):
        print("     Classification with PCA with", i, "features")
        pca = PCA(n_components=i)
        Y_copy = Y.copy()
        X_copy = pca.fit_transform(X)
        KAccuracies = []

        # For every value of n_components, PCA is tested on five different split of the dataSet
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
    plt.title('Accuratezza del classificatore al variare delle componenti della PCA. Esperimento ' + str(j+1))
    plt.xlabel('n_component')
    plt.ylabel('Accuracy_score')
    plt.xlim(0.0, 31.0)
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
plt.ylim(0.0, 29.0)
namePlot = "Number of components with max accuracy for every experiment.png"
plt.savefig(namePlot, bbox_inches='tight')


# use of whiten
experiment = []
accuracies_T = []
accuracies_F = []
components = 28
for i in range(1, 21):
    print("     Classification ", i)
    pca_F = PCA(n_components=components, whiten=False)
    pca_T = PCA(n_components=components, whiten=True)
    X_copy = X.copy()
    Y_copy = Y.copy()

    pca_F.fit(X_copy)
    pca_T.fit(X_copy)

    KAccuracies_T = []
    KAccuracies_F = []
    for k in range(0, 5):
        classifier_T = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                          criterion="gini",
                                          random_state=8, n_jobs=4, bootstrap='false', max_features=components)
        classifier_F = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                            criterion="gini",
                                            random_state=8, n_jobs=4, bootstrap='false', max_features=components)
        X_train, X_test, Y_train, Y_test = train_test_split(X_copy, Y_copy, test_size=0.2, shuffle='true',
                                                            stratify=Y_copy)
        X_train_T = pca_T.transform(X_train)
        X_test_T = pca_T.transform(X_test)
        X_train_F = pca_F.transform(X_train)
        X_test_F = pca_F.transform(X_test)
        classifier_T.fit(X_train_T, Y_train)
        classifier_F.fit(X_train_F, Y_train)
        prediction_T = classifier_T.predict(X_test_T)
        prediction_F = classifier_F.predict(X_test_F)
        KAccuracies_T.append(accuracy_score(Y_test, prediction_T))
        KAccuracies_F.append(accuracy_score(Y_test, prediction_F))

    print("F ", KAccuracies_F)
    print("T ", KAccuracies_T)
    accuracies_F.append(np.mean(KAccuracies_F))
    accuracies_T.append(np.mean(KAccuracies_T))

    experiment.append(i)

print("Accuracies with whiten: ", accuracies_T)
print("Accuracies without whiten: ", accuracies_F)
plt.plot(experiment, accuracies_T, 'b')
plt.plot(experiment, accuracies_F, 'r')
plt.title('Andamento precisione con e senza sbiancamento della PCA')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 21.0)
plt.ylim(0.0, 1.0)
red_patch = matplotlib.patches.Patch(color='red', label='Senza Sbiancamento')
blue_patch = matplotlib.patches.Patch(color='blue', label='Con Sbiancamento')
plt.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.savefig("PCA's accuracies with and without whiten", bbox_inches='tight')


# select svd_solver
experiment = []
accuracies_auto = []
accuracies_full = []
accuracies_arpack = []
accuracies_randomized = []
components = 28
for i in range(1, 21):
    print("     Classification ", i)
    pca_auto = PCA(n_components=components, svd_solver='auto')
    pca_full = PCA(n_components=components, svd_solver='full')
    pca_arpack = PCA(n_components=(components-1), svd_solver='arpack')
    pca_randomized = PCA(n_components=components, svd_solver='randomized')
    X_copy = X.copy()
    Y_copy = Y.copy()

    pca_auto.fit(X_copy)
    pca_full.fit(X_copy)
    pca_arpack.fit(X_copy)
    pca_randomized.fit(X_copy)

    KAccuracies_auto = []
    KAccuracies_full = []
    KAccuracies_arpack = []
    KAccuracies_randomized = []
    for k in range(0, 5):
        classifier_auto = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                          criterion="gini",
                                          random_state=8, n_jobs=4, bootstrap='false', max_features=components)
        classifier_full = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                            criterion="gini",
                                            random_state=8, n_jobs=4, bootstrap='false', max_features=components)
        classifier_arpack = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                            criterion="gini",
                                            random_state=8, n_jobs=4, bootstrap='false', max_features=(components-1))
        classifier_randomized = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                            criterion="gini",
                                            random_state=8, n_jobs=4, bootstrap='false', max_features=components)

        X_train, X_test, Y_train, Y_test = train_test_split(X_copy, Y_copy, test_size=0.2, shuffle='true',
                                                            stratify=Y_copy)
        X_train_auto = pca_auto.transform(X_train)
        X_test_auto = pca_auto.transform(X_test)
        X_train_full = pca_full.transform(X_train)
        X_test_full = pca_full.transform(X_test)
        X_train_arpack = pca_arpack.transform(X_train)
        X_test_arpack = pca_arpack.transform(X_test)
        X_train_randomized = pca_randomized.transform(X_train)
        X_test_randomized = pca_randomized.transform(X_test)
        classifier_auto.fit(X_train_auto, Y_train)
        classifier_full.fit(X_train_full, Y_train)
        classifier_arpack.fit(X_train_arpack, Y_train)
        classifier_randomized.fit(X_train_randomized, Y_train)
        prediction_auto = classifier_auto.predict(X_test_auto)
        prediction_full = classifier_full.predict(X_test_full)
        prediction_arpack = classifier_arpack.predict(X_test_arpack)
        prediction_randomized = classifier_randomized.predict(X_test_randomized)
        KAccuracies_auto.append(accuracy_score(Y_test, prediction_auto))
        KAccuracies_full.append(accuracy_score(Y_test, prediction_full))
        KAccuracies_arpack.append(accuracy_score(Y_test, prediction_arpack))
        KAccuracies_randomized.append(accuracy_score(Y_test, prediction_randomized))

    print("AUTO ", KAccuracies_auto)
    print("FULL ", KAccuracies_full)
    print("ARPACK ", KAccuracies_arpack)
    print("RANDOMIZED ", KAccuracies_randomized)
    accuracies_auto.append(np.mean(KAccuracies_auto))
    accuracies_full.append(np.mean(KAccuracies_full))
    accuracies_arpack.append(np.mean(KAccuracies_arpack))
    accuracies_randomized.append(np.mean(KAccuracies_randomized))

    experiment.append(i)

print("Accuracies auto: ", accuracies_auto)
print("Accuracies full: ", accuracies_full)
print("Accuracies arpack: ", accuracies_arpack)
print("Accuracies randomized: ", accuracies_randomized)
plt.plot(experiment, accuracies_auto, 'b')
plt.plot(experiment, accuracies_full, 'r')
plt.plot(experiment, accuracies_arpack, 'g')
plt.plot(experiment, accuracies_randomized, 'y')
plt.title('Andamento precisione con i diversi tipi di svd_solver')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 21.0)
plt.ylim(0.0, 1.0)
red_patch = matplotlib.patches.Patch(color='red', label='FULL')
blue_patch = matplotlib.patches.Patch(color='blue', label='AUTO')
green_patch = matplotlib.patches.Patch(color='green', label='ARPACK')
yellow_patch = matplotlib.patches.Patch(color='yellow', label='RANDOMIZED')
plt.legend(handles=[blue_patch, red_patch, green_patch, yellow_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.savefig("PCA's accuracies with different values of svd_solver", bbox_inches='tight')
