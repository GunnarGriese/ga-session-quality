import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import numpy as np

def error_plots(df_confusion_matrix, fp_value, tp_value, auc_value):
    """
    Generate two plots (confusion matrix & ROC).

    Parameters
    ----------
    df_confusion_matrix: confusion matrix
    fp_value: false positive
    tp_value: true positive
    auc_value: auc value

    Returns
    ----------
    plot_1: confusion matrix
    plot_2: ROC curve
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
    plot_1 = sns.heatmap(df_confusion_matrix, ax=ax1, annot=True, fmt='g', cmap='Oranges')
    plot_1.xaxis.set_ticks_position('top')
    plot_1.set_ylabel('Actual')
    plot_1.set_xlabel('Predicted')
    plot_1.xaxis.set_label_position('top')

    ax2.plot(fp_value, tp_value, color='orange', label='ROC curve (area = {})'.format(auc_value))
    plt.title('ROC Curve, AUC:'+str(round(auc_value, 4)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax2.plot([0, 1], [0, 1], color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fig.suptitle("Google's Session Quality Score")
    plt.grid(True)