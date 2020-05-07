import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validation(test, prediction):
    accuracy = accuracy_score(test, prediction)
    precision = precision_score(test, prediction, average='macro')
    recall = recall_score(test, prediction, average='macro')
    f1 = f1_score(test, prediction, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1 measure:", f1)
    print("\n\n")


def confusionMatrix(test, prediction, name):
    # Plot non-normalized confusion matrix
    matrix = metrics.confusion_matrix(test, prediction)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=list(set(test)),
                yticklabels=list(set(test)),
                annot=True,
                fmt='d')
    plt.title(name)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

