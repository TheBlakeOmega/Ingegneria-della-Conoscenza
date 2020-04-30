import songFeature
import classifiers
import dataFunctions
import numpy as np
import matplotlib.pyplot as plt

# load dataset and feature selection
X_train, X_test, Y_train, Y_test = dataFunctions.getData("C:/Users/wiare/Desktop/musicfeatures/data.csv")

# KNN
classifier = classifiers.knnClassifier(X_train, Y_train)
prediction = classifier.predict(X_test)
print(classifier)
classifiers.validation(Y_test, prediction)

# naive bayes
classifier1 = classifiers.bayesianClassifier(X_train, Y_train)
prediction = classifier1.predict(X_test)
print(classifier1)
classifiers.validation(Y_test, prediction)

# extra tree classifier
classifier2 = classifiers.extraTreesClassifier(X_train, Y_train)
prediction = classifier2.predict(X_test)
print(classifier2)
classifiers.validation(Y_test, prediction)

# random forest classifier
classifier3 = classifiers.randomForestClassifier(X_train, Y_train)
prediction = classifier3.predict(X_test)
print(classifier3)
classifiers.validation(Y_test, prediction)
importances = classifier3.feature_importances_ # importanza delle singole feature
std = np.std([tree.feature_importances_ for tree in classifier3.estimators_], axis=0)
indices = np.argsort(importances)[::-1]


# build plot
std = np.std([tree.feature_importances_ for tree in classifier3.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# Prediction on an input song
a = 'y'
while a == 'y':
    print("\n")

    path_song = input("Insert song's path: ")
    print("LOADING SONG")
    song = songFeature.load_song(path_song)
    feature = songFeature.get_song_feature(song)
    print("Features computed:\n", feature, "\n\n")

    feature = feature.reshape(1, -1)
    print("Prediction KNN: ", classifiers.getPrediction(classifier, feature))
    print("Prediction Bayes: ", classifiers.getPrediction(classifier1, feature))
    print("Prediction Extra Tree: ", classifiers.getPrediction(classifier2, feature))
    print("Prediction Random Forest: ", classifiers.getPrediction(classifier3, feature))
    a = input("Have you another song?  {y/n}  ")
