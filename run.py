"""
Entry point.

Simulates A* plus a low level controller for path planning & execution.

author: conor hayes
"""

import argparse
import pathlib
import signal

import pandas as pd

from hw2.data import Dataset, ObservabilityData, Opts
from hw2 import svm

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
from hw2.trials import clf_trial, clf_trial_sample_points

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
    # partA3(ds)
    partB(ds)

    # lib_experiments(ds)

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
    iloc_list = [100, 200, 340]
    for i, iloc in enumerate(iloc_list):
        print(iloc)
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


def partA3(ds: Dataset):
    # from sklearn import svm

    # clf = svm.SVC(kernel="rbf")
    cfg = svm.SVM.Config("rbf", 1.0, 1.0)
    clf = svm.SVM(cfg)

    fig = plt.figure("A3 - sample points linsep")
    clf_trial_sample_points(clf, (0, 0, 0), (5, 5, 5), fig)  # type: ignore
    fig.suptitle("SVM demo - linearly separable points")

    # clf = svm.SVC(kernel="rbf")
    clf = svm.SVM(cfg)
    fig = plt.figure("A3 - sample points unsep")
    clf_trial_sample_points(clf, (0, 0, 0), (1, 1, 1), fig)  # type: ignore
    fig.suptitle("SVM demo - non-separable points")


def partB(ds: Dataset, generate_data: bool = True):
    obs = ObservabilityData(ds, freq_hz=2.0, sliding_window_len_s=2.0)
    X_train, X_test, y_train, y_test = obs.preprocess(
        Opts.CONTINUOUS_ROT | Opts.SHUFFLE
    )

    obs = ObservabilityData(ds, freq_hz=2.0, sliding_window_len_s=2.0)
    subj = 11

    cfg = svm.SVM.Config("rbf", 5.0, 1.0)
    clf = svm.SVM(cfg)
    clf_trial(
        obs,
        clf,
        subj,
        "rbf, continuous rotation",
        *obs.preprocess(Opts.CONTINUOUS_ROT | Opts.SHUFFLE),
    )

    return
    # try some values for C and sigma for grid search
    Cs = [0.1, 1, 10, 100]
    sigmas = [0.1, 0.5, 1, 2, 5]
    out_dir = pathlib.Path(__file__).parent / "data/ds0_grid"

    if generate_data:
        print("Writing grid search test dataset to files...")
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

    for c in Cs:
        for sigma in sigmas:
            out_path = out_dir / f"C{c}_S{sigma}.csv".replace(".", "-")

            if generate_data:
                cfg = svm.SVM.Config("rbf", c, sigma)
                clf = svm.SVM(cfg)

                out = X_test.copy()

                # train & test a separate classifier for each landmark
                print(f"C={c}, sigma={sigma}...", end="", flush=True)
                for lm in obs.landmarks:
                    print(f"{lm} ", end="", flush=True)
                    clf.fit(X_train.to_numpy(), y_train[lm].to_numpy())
                    yhat = clf.predict(X_test.to_numpy())

                    out[f"yhat_{lm}"] = yhat
                    out[f"y_{lm}"] = y_test[lm]
                print()

                # write to file
                if out_path.exists():
                    out_path.unlink()
                out.to_csv(out_path)


def lib_experiments(ds: Dataset):
    """
    Experiments in learning with libraries.

    To validate my learning aim.

    using sklearn's svm implmementations
    https://scikit-learn.org/stable/modules/svm.html
    """
    from sklearn import svm

    obs = ObservabilityData(ds, freq_hz=2.0, sliding_window_len_s=2.0)
    subj = 11

    clf = svm.SVC(kernel="rbf")
    clf_trial(
        obs,
        clf,
        subj,
        "rbf, continuous rotation",
        *obs.preprocess(Opts.CONTINUOUS_ROT | Opts.SHUFFLE),
    )

    # this doesn't work as well
    # clf = svm.SVC(kernel="sigmoid", degree=8)
    # clf_trial(
    #     obs,
    #     clf,
    #     13,
    #     "sigmoid, cont. rot",
    #     *obs.preprocess(Opts.CONTINUOUS_ROT | Opts.SHUFFLE),
    # )

    clf = svm.SVC(kernel="poly", degree=8)
    clf_trial(
        obs,
        clf,
        subj,
        "poly 8, cont. rot",
        *obs.preprocess(Opts.CONTINUOUS_ROT | Opts.SHUFFLE),
    )

    # based on this experiment, i'm going to go ahead and
    # implement RBF kernel.


if __name__ == "__main__":
    main()
