import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn import svm


data = pd.read_csv("D:/Users/Marco/Desktop/DataSet/musicfeatures/data.csv")
datas = data.drop('filename', 1)
X = np.array(datas.drop('label', 1))
y = np.array(data['label'])

'''
#best n_neighbors for rfc
param_range = np.arange(1,10)
train_scores, test_scores = validation_curve(svm.SVC(),X, y, param_name="C",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVC")
plt.xlabel("C")
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
#best kernel for svc
param_range =  np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(svm.SVC(),X, y, param_name="gamma",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVC")
plt.xlabel("gamma")
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
parameters = {'C':[1, 5, 10],
              'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
              'degree':[1, 3, 6, 9],
              'gamma':['scale', 'auto']}

model = svm.SVC()
gridSearch = GridSearchCV(model, param_grid=parameters, cv=3, n_jobs=4, verbose=2)

gridSearch.fit(X, y)
print(gridSearch.best_params_)
