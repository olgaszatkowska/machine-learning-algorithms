from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


def plot_confusion_matrix(y_true: Iterable[int], y_pred: list[int]):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(i) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.show()
    
    
def print_numerical_statistics(y_true: Iterable[int], y_pred: list[int]):
    from sklearn.metrics import precision_score, recall_score, f1_score
    """
    Precision: The ratio of correctly predicted positive observations to the total predicted positives.
               It answers the question: "What proportion of positive identifications was actually correct?"

    Recall: The ratio of correctly predicted positive observations to the all observations in actual class.
            It answers the question: "What proportion of actual positives was identified correctly?

    F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
    """

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
