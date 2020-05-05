from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import recall_score

def knnClassifier(X_train, Y_train):
    print("Building KNN Classifier:")
    classifier = KNeighborsClassifier(n_neighbors=4, algorithm='auto', p=2, metric='minkowski', leaf_size=5)
    classifier.fit(X_train, Y_train)
    return classifier


def bayesianClassifier(X_train, Y_train):
    print("Building Bayesian Classifier:")
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    return classifier


def extraTreesClassifier(X_train, Y_train):
    print("Building ExtraTrees Classifier:")
    classifier = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2, criterion="gini",
                                      random_state=8, n_jobs=4, bootstrap='false', max_features=20)
    classifier.fit(X_train, Y_train)
    return classifier


def randomForestClassifier(X_train, Y_train):
    print("Building RandomForest Classifier:")
    classifier = RandomForestClassifier(n_estimators=146, max_features=20, max_depth=15, max_samples=45,
                                   min_samples_leaf=1,
                                   criterion='gini', min_samples_split=2,
                                   random_state=8,
                                   n_jobs=4)

    classifier.fit(X_train, Y_train)
    return classifier


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


def getPrediction(classifier, test):
    return classifier.predict(test)[0], max(classifier.predict_proba(test)[0])
