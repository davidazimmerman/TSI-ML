"""
@author: david

Makes clean, labeled, and interpretable confusion matrix with
pretty colors.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels, normalize = None,
                         savefig = False, filename = 'confusion_plot.png'):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by classifier
    
    labels: array-like of shape (n_classes)
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels. 
        If None is given, those that appear at least once 
        in y_true or y_pred are used in sorted order. Also used
        to label the confusion matrix x and y tick labels

    normalize: {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows),
        predicted(columns) or all the population. If None,
        confusion matrix will not be normalized

    savefig: boolean
        Save the figure as an image with extension specified
        in filename

    Returns
    -------
    display 

    See the following documentation
    -------------------------------
    Making Heatmaps
    - https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    sklearn.metrics.confusion_matrix
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    C = confusion_matrix(y_true, y_pred, labels, normalize)

    n = C.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(C, cmap = plt.cm.Blues)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    for i in range(n):

        for j in range(n):

            ax.text(j, i, C[i, j],
                    ha = "center", va = "center", color = 'w')
    
    ax.set_title('Confusion Matrix')
    

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=  45, ha = "right",
            rotation_mode = "anchor")

    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    fig.colorbar(im)
    fig.tight_layout()

    if savefig == True:
        plt.savefig(fname = filename)
    plt.show()


def plot_roc_curve(y_true, y_prob, savefig = False, filename = 'roc_plot.png'):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by classifier

    savefig: boolean
        Save the figure as an image with extension specified
        in filename

    Returns
    -------
    display

    See the following documentation
    -------------------------------
    sklearn.metrics.auc
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
    sklearn.metrics.roc_curve
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    area = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color = 'darkorange', 
            lw = 2, label = 'ROC curve (AUC = %0.2f)' % area)
    plt.plot([0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc = 'lower right')
    if savefig == True:
        plt.savefig(fname = filename)
    plt.show()