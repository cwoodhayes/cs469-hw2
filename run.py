"""
Entry point.

Simulates A* plus a low level controller for path planning & execution.

author: conor hayes
"""

import argparse
import pathlib
import signal

from hw2.data import Dataset, ObservabilityData

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from hw2.plot import (
    plot_landmark_bars,
    plot_single_observation,
    plot_trajectories_pretty,
    plot_visibility_3d,
    plot_visibility_3d_numpy,
    sync_axes,
)

REPO_ROOT = pathlib.Path(__file__).parent
FIGURES_DIR = REPO_ROOT / "figures"


def main():
    print("cs469 Homework 2")
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.rcParams["legend.fontsize"] = 14

    ns = get_cli_args()

    # my assigned dataset is ds1, so I'm hardcoding this
    dsdir = REPO_ROOT / "data/ds1"
    ds = Dataset.from_dataset_directory(dsdir)

    # partA1(ds)
    # partA2(ds)

    lib_experiments(ds)

    if ns.save:
        print("Saving figures...")
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = fig.get_label() or f"figure_{num}"
            fig.savefig(str(FIGURES_DIR / f"{name}.png"))
    else:
        plt.show()


def get_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("")
    cli.add_argument(
        "-s",
        "--save",
        action="store_true",
    )
    return cli.parse_args()


def partA1(ds: Dataset):
    """Generate a few plots that demonstrate the learning aim."""

    ds = ds.segment_percent(0, 0.2, True)
    obs = ObservabilityData(ds, sliding_window_len_s=2.0, freq_hz=2.0)
    obs.to_file()

    fig = plt.figure("A1 - example states", figsize=(10, 6))
    axes: list[Axes] = fig.subplots(1, 3)

    print(len(obs.data))
    iloc_list = [100, 200, 554]
    for i, iloc in enumerate(iloc_list):
        row = obs.data.iloc[iloc]
        plot_single_observation(obs, axes[i], row)
        axes[i].set_title(f"t={round(row['time_s'], 2)}")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.2, 0.05), ncol=2, fontsize=10)
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle("Landmarks Visible from 3 Example States")

    plot_landmark_bars(obs, figlabel="A1 - landmarks over time")


def partA2(ds: Dataset):
    """Generate a few plots that demonstrate why SVM may work."""

    obs = ObservabilityData(ds, sliding_window_len_s=2.0, freq_hz=2.0)
    # obs.to_file()

    fig = plt.figure("A2 - 3dplot")
    ax = fig.add_subplot(221, projection="3d")
    plot_visibility_3d(obs, ax, {13, 7})

    ax = fig.add_subplot(222, projection="3d")
    plot_visibility_3d(obs, ax, {10, 15})

    ax = fig.add_subplot(223, projection="3d")
    plot_visibility_3d(obs, ax, {19, 20})

    ax = fig.add_subplot(224, projection="3d")
    plot_visibility_3d(obs, ax, {6, 8})


def lib_experiments(ds: Dataset):
    """
    Experiments in learning with libraries.

    To validate my learning aim.

    using sklearn's svm implmementations
    https://scikit-learn.org/stable/modules/svm.html
    """
    from sklearn import svm
    from sklearn.metrics import ConfusionMatrixDisplay

    obs = ObservabilityData(ds, freq_hz=2.0, sliding_window_len_s=2.0)

    X_train, X_test, y_train, y_test = obs.test_train_split()

    # convert X to use sin(theta), cos(theta) rather than just theta
    X = X_train.copy()
    X["sin"] = np.sin(X_train["orientation_rad"])
    X["cos"] = np.cos(X_train["orientation_rad"])
    del X["orientation_rad"]

    clf = svm.SVC(kernel="poly", degree=8)
    # train a classifier for a landmark
    subj = 13
    clf.fit(X.to_numpy(), y_train[subj].to_numpy())

    # classify the training set
    yhat_train = clf.predict(X.to_numpy())

    # visualize the output
    fig = plt.figure("libtest - trainout")
    ax = fig.add_subplot(211, projection="3d")
    plot_visibility_3d_numpy(ds, X_train.to_numpy(), yhat_train, ax, subject=subj)
    ax.set_title(r"Training set predicted labels ($\hat{y}$)")

    # compare against ground truth labels
    ax2 = fig.add_subplot(212, projection="3d")
    plot_visibility_3d(obs, ax2, {subj})
    ax2.set_title("Training set ground truth labels (y)")

    sync_axes(ax, ax2)
    sync_axes(ax2, ax)

    # print % correct
    n_correct = np.count_nonzero(yhat_train == y_train[subj].to_numpy())
    print(f"{n_correct}/{len(y_train)} correct ({100 * n_correct / len(y_train):.2f}%)")
    # This computes and plots in one step
    ConfusionMatrixDisplay.from_predictions(y_train[subj].to_numpy(), yhat_train)


if __name__ == "__main__":
    main()
