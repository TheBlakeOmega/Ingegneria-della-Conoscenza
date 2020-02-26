import songFeature
import classifiers
import dataFunctions

# load dataset and feature selection
X_train, X_test, Y_train, Y_test = dataFunctions.getData("C:/Users/wiare/Desktop/musicfeatures/data.csv")
X_train, X_test, pca = dataFunctions.makePCA(X_train, X_test)

# KNN
classifier = classifiers.knnClassifier(X_train, Y_train)
prediction = classifier.predict(X_test)
print(classifier)
accuracy = classifiers.accuracy(Y_test, prediction)

# naive bayes
classifier1 = classifiers.bayesianClassifier(X_train, Y_train)
prediction = classifier1.predict(X_test)
print(classifier1)
accuracy = classifiers.accuracy(Y_test, prediction)

# extra tree classifier
classifier2 = classifiers.extraTreesClassifier(X_train, Y_train)
prediction = classifier2.predict(X_test)
print(classifier2)
accuracy = classifiers.accuracy(Y_test, prediction)

# Prediction on an input song
a = 'y'
while a == 'y':
    print("LOADING SONG")
    path_song = input("Insert song's path: ")
    song = songFeature.load_song(path_song)
    feature = songFeature.get_song_feature(song)
    print("Features computed:\n", feature, "\n\n")

    feature = feature.reshape(1, -1)
    feature = pca.transform(feature)
    print("Prediction KNN: ", classifiers.getPrediction(classifier, feature))
    print("Prediction Bayes: ", classifiers.getPrediction(classifier1, feature))
    print("Prediction Extra_Tree: ", classifiers.getPrediction(classifier2, feature))
    a = input("Have you another song?  {y/n}  ")
