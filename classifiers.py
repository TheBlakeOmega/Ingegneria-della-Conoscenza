from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


def knnClassifier(X_train, Y_train):
    print("Building KNN Classifier:")
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, Y_train)
    return classifier


def bayesianClassifier(X_train, Y_train):
    print("Building Bayesian Classifier:")
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    return classifier


def extraTreesClassifier(X_train, Y_train):
    print("Building ExtraTrees Classifier:")
    classifier = ExtraTreesClassifier()
    classifier.fit(X_train, Y_train)
    return classifier


def accuracy(test, prediction):
    accuracy = accuracy_score(test, prediction)
    print(accuracy)
    return accuracy


def getPrediction(classifier,test):
    if test.shape[0] > 1:
        return classifier.predict(test)
    else:
        return classifier.predict(test)[0], max(classifier.predict_proba(test)[0])
