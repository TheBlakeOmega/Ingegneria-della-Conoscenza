import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import dataFunctions
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score

data = pd.read_csv("D:/Users/Marco/Desktop/DataSet/musicfeatures/data.csv")
datas = data.drop('filename', 1)
X = np.array(datas.drop('label', 1))
y = np.array(data['label'])
X_train, X_test, Y_train, Y_test = dataFunctions.getData("D:/Users/Marco/Desktop/DataSet/musicfeatures/data.csv")

print(list(set(y)))
'''
#best n_estimator for rfc
param_range = np.arange(1,220,dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="n_estimators",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
'''
'''
#best max_depth for rfc
param_range = np.arange(1,25, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="max_depth",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_depth")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best min_samples_split for rfc
param_range = np.arange(1,6, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="min_samples_split",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
'''

'''
#best min_samples_leaf for rfc
param_range = np.arange(1,20, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="min_samples_leaf",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

'''
'''
#best max_samples  for rfc
param_range = np.arange(1,100)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="max_samples",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_samples ")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
'''
'''
#best max_features    for rfc
param_range = np.arange(1,28)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="max_features",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_features")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
'''
'''
#best max_features    for rfc
param_range = np.arange(1,20)
train_scores, test_scores = validation_curve(RandomForestClassifier(),X, y, param_name="random_state",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("random_state")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
'''


#1*3*3*2*2*4 combination
param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 15, 25],
    'max_features': [1, 20, 28],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 150, 200, 250]
}
model = RandomForestClassifier()
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=4, verbose=2)

gridSearch.fit(X, y)
print(gridSearch.best_params_)

print ("Miglior punteggio:", gridSearch.best_score_)
print ("Migliori parametri:", gridSearch.best_params_)

'''

#1*3*3*2*2*4 combination
param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 15, 25],
    'max_features': [1, 20,28],
    'min_samples_leaf': [1,2],
    'min_samples_split': [2,4],
    'n_estimators': [100, 150, 200, 250]
}
model = ExtraTreesClassifier()
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=4, verbose=2)

gridSearch.fit(X, y)


print ("Miglior punteggio:", gridSearch.best_score_)
print ("Migliori parametri:", gridSearch.best_params_)
'''