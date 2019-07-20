import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

def get_error_scores(y_true, y_pred):
    """
    Calculate classification performance.

    INPUT
    ----------
    y_true: true class
    y_pred: predicted class

    OUTPUT
    ----------
    df_scores: data frame with error summary
    confusion_matrix: confusion matrix
    fp: int, false positives
    tp: int, true positives
    auc: float, area under the ROC curve
    """

    # Calculate errors of interest
    score_accuracy = accuracy_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)

    df_scores = pd.DataFrame({'Accuracy': [score_accuracy],
                            'F1': [score_f1],
                            'Recall': [score_recall],
                            'Precision': [score_precision]}
                             #,
                            #index=['score']
                            )

    # Calculate Confusion Matrix
    con_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred))
    #df_confusion = pd.DataFrame(cm_raw)

    # Calculate AUC score
    fp_value, tp_value, _ = roc_curve(y_true, y_pred)
    auc_value = auc(fp_value, tp_value)
    
    return df_scores, con_matrix, fp_value, tp_value, auc_value