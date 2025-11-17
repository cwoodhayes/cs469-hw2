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

    partA1(ds)

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
    ds = ds.segment_percent(0, 0.5, True)
    obs = ObservabilityData.from_dataset(ds)
    # obs.to_file()

    # plot ground-truth observability dataset
    # fig = plt.figure()
    # plot_trajectories_pretty(ds, fig, "Landmark Observability (Ground Truth)")

    fig2 = plt.figure()
    ax = fig2.subplots()
    row = obs.data.iloc[100]
    plot_single_observation(obs, ax, row)
    ax.legend(fontsize=8)

    plot_landmark_bars(obs)


if __name__ == "__main__":
    main()
