"""
Methods for testing out different classifiers.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from hw2 import svm
import matplotlib.pyplot as plt

from hw2.data import ObservabilityData
from hw2.plot import plot_visibility_3d, plot_visibility_3d_numpy, sync_axes


def clf_trial(
    obs: ObservabilityData,
    clf: svm.SVM | Any,  # can also be scipy's svm.SVC
    subj: int,
    label: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> plt.Figure:
    """Run a trial on a classifier setup."""
    clf.fit(X_train.to_numpy(), y_train[subj].to_numpy())

    # classify the test set
    yhat_test = clf.predict(X_test.to_numpy())

    # visualize the output
    fig = plt.figure("libtest - trainout")
    ax = fig.add_subplot(211, projection="3d")
    plot_visibility_3d_numpy(
        obs.source_ds, X_test.to_numpy(), yhat_test, ax, subject=subj
    )
    ax.set_title(r"Training set predicted labels ($\hat{y}$)")

    # compare against ground truth labels
    ax2 = fig.add_subplot(212, projection="3d")
    plot_visibility_3d_numpy(
        obs.source_ds, X_test.to_numpy(), y_test[subj].to_numpy(), ax2, subject=subj
    )
    ax2.set_title("Training set ground truth labels (y)")

    sync_axes(ax, ax2)
    sync_axes(ax2, ax)

    # print % correct
    n_correct = np.count_nonzero(yhat_test == y_test[subj].to_numpy())
    print(label)
    print(f"  {n_correct}/{len(y_test)} correct ({100 * n_correct / len(y_test):.2f}%)")
    # This computes and plots in one step
    disp = ConfusionMatrixDisplay.from_predictions(y_test[subj].to_numpy(), yhat_test)
    disp.ax_.set_title(label)

    fig.suptitle(label)

    return fig
