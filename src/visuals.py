import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import numpy as np

def plot_errors(con_matrix, fp_value, tp_value, auc_value):
    """
    Function to create two plots: confusion matrix and AUC ROC curve

    INPUT
    ----------
    con_matrix: confusion matrix (output from helpers)
    fp_value: False Positives (output from helpers)
    tp_value: True Positives (output from helpers)
    auc_value: AUC Value (output from helpers)

    OUTPUT
    ----------
    conf_matrix: plot, confusion matrix
    roc_curve: plot, ROC curve
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    conf_matrix = sns.heatmap(con_matrix, ax=ax1, annot=True, fmt='g', cmap='Blues')
    conf_matrix.xaxis.set_ticks_position('top')
    conf_matrix.set_ylabel('Observed')
    conf_matrix.set_xlabel('Predicted')
    conf_matrix.xaxis.set_label_position('top')

    ax2.plot(fp_value, tp_value, color='blue', label='ROC curve (area = {})'.format(auc_value))
    plt.title('ROC Curve, AUC:'+str(round(auc_value, 4)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax2.plot([0, 1], [0, 1], color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fig.suptitle("Google's Session Quality Score")
    plt.grid(True)