"""
Methods for testing out different classifiers.
"""

import timeit
from typing import Any
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from hw2 import svm
import matplotlib.pyplot as plt

from hw2.data import ObservabilityData
from hw2.plot import (
    plot_performance_comparison,
    plot_visibility_3d,
    plot_visibility_3d_numpy,
    sync_axes,
)


def clf_trial(
    obs: ObservabilityData,
    clf: svm.SVM | Any,  # can also be scipy's svm.SVC
    subj: int,
    label: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    figid: str | None = None,
) -> None:
    """Run a trial on a classifier setup."""
    from sklearn.metrics import ConfusionMatrixDisplay

    time_s = timeit.timeit(
        lambda: clf.fit(X_train.to_numpy(), y_train[subj].to_numpy()), number=1
    )
    print(f"fit() took {time_s}s to run.")

    # classify the test set
    yhat_test = clf.predict(X_test.to_numpy())

    # visualize the output
    fig = plt.figure(f"trial_{label}" if figid is None else figid, figsize=(6, 10))
    plot_performance_comparison(obs, subj, fig, X_test, y_test, yhat_test)

    # print % correct
    n_correct = np.count_nonzero(yhat_test == y_test[subj].to_numpy())
    accuracy = 100 * n_correct / len(y_test)

    true_pos = np.count_nonzero(
        (yhat_test == y_test[subj].to_numpy()) & (yhat_test == 1)
    )
    false_neg = np.count_nonzero(
        (yhat_test != y_test[subj].to_numpy()) & (yhat_test == 1)
    )
    denom = true_pos + false_neg
    recall = true_pos / denom if denom != 0 else 0

    print(f"  {n_correct}/{len(y_test)} correct ({accuracy:.2f}%)")
    print(f"  Recall (correctly predicted positives): {recall:.2f}%")
    # This computes and plots in one step
    disp = ConfusionMatrixDisplay.from_predictions(y_test[subj].to_numpy(), yhat_test)
    disp.ax_.set_title(label)
    disp.figure_.number = f"trial_conf_{label}"

    fig.suptitle(label + f"\naccuracy={accuracy}, recall={recall}")
    fig.tight_layout()


def clf_trial_sample_points(
    clf: svm.SVM, c1: np.ndarray | tuple, c2: np.ndarray | tuple, fig: Figure
):
    """Demonstrate SVM on a simple bimodal distribution.

    Args:
        clf (svm.SVM): classifier to be trained
        c1: mean of distribution 1
        c2: mean of distribution 2
        title (str): title of this experiment, used for plots.
    """
    # generate some simple data to test a classifier with
    # a bimodal distribution of 1000 points.
    N = 1000
    mean1 = np.array(c1)
    mean2 = np.array(c2)
    cov = np.eye(3)

    X0 = np.random.multivariate_normal(mean1, cov, size=N // 2)
    X1 = np.random.multivariate_normal(mean2, cov, size=N // 2)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(N // 2), np.ones(N // 2)])

    ax = fig.add_subplot(211, projection="3d")
    ax.set_title("Ground Truth Labels")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)  # type: ignore

    # now classify & plot resulting labels
    clf.fit(X, y)
    yhat = clf.predict(X)

    ax2 = fig.add_subplot(212, projection="3d")
    ax2.set_title(r"Classifier Output Labels ($\hat{y}$)")
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=yhat)  # type: ignore
    fig.tight_layout()
