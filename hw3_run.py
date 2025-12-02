"""
HW3 - doing jueun's dataset
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

REPO_ROOT = pathlib.Path(__file__).parent
FIGURES_DIR = REPO_ROOT / "figures"


def main():
    print("cs469 Homework 3")
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.rcParams["legend.fontsize"] = 14

    ns = get_cli_args()

    # jeuen's dataset
    jds = pd.read_csv(REPO_ROOT / "jeuen_learning_dataset.txt", sep=" ")

    train_test_controller(jds)

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


def train_test_controller(jds: pd.DataFrame) -> None:
    """Train & test SVM-based coarse grained controller."""

    X_train, X_test, y_train, y_test = test_train_split(preprocess(jds))
    out_dir = pathlib.Path(__file__).parent / "data/hw3"
    csv_path = out_dir / "outputs.csv"

    cfg = svm.SVM.Config("rbf", 10, 0.1)
    # train a classifier for each dimension of the output space
    cls_cols = ["dx_class", "dy_class", "dtheta_class"]
    clfs = {}
    test_df = X_test.copy()
    test_df.join(y_test)

    for dim in cls_cols:
        print(f"Training classifier for {dim} - cfg={cfg}...")
        clf = svm.SVM(cfg)
        clf.fit(X_train.to_numpy(), y_train[dim].to_numpy())
        clfs[dim] = clf

        # run on the test set
        yhat = clf.predict(X_test.to_numpy())
        test_df[f"{dim}_hat"] = yhat

    test_df.to_csv(csv_path)


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset to work with individual classifiers."""

    # for each output bucket, convert the values to -1 and 1
    dat_cols = ["dx_body", "dy_body", "dtheta"]
    cls_cols = ["dx_class", "dy_class", "dtheta_class"]
    means = {}
    for col in dat_cols:
        means[col] = data[col].mean()

    data = data.copy()

    for i in range(len(cls_cols)):
        data[cls_cols[i]] = np.where(data[dat_cols[i]] > means[dat_cols[i]], 1, -1)

    del data["dx_body"]
    del data["dy_body"]
    del data["dtheta"]

    return data


def test_train_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dset into test + training data + labels.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    np.random.seed(0)
    xcols = ["v_cmd", "w_cmd", "dt"]
    X = data[xcols].copy()
    y = data.copy()
    for name in xcols:
        del y[name]

    perm = np.random.permutation(len(X))
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)

    N = int(len(X) * (1 - test_size))
    return X.iloc[:N], X.iloc[N:], y.iloc[:N], y.iloc[N:]


if __name__ == "__main__":
    main()
