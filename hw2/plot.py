"""
plotting functions for robot data
"""

from matplotlib.figure import Figure
import numpy as np
from matplotlib.axes import Axes
from matplotlib import patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

from hw2.data import Dataset


def plot_map_colored_obstacles(ds: Dataset, ax: Axes) -> None:
    """
    Generate a view of the robot's arena where each obstacle is a different color.
    """
    ### plot the landmarks as black discs
    centers = []
    for lm in ds.landmarks.itertuples(index=False):
        # these only actually show up if you zoom wayyyyy in. the stdevs are super small.
        oval = patches.Ellipse(
            (lm.x_m, lm.y_m),  # type: ignore
            width=lm.x_std_dev * 1000,  # type: ignore
            height=lm.y_std_dev * 1000,  # type: ignore
            facecolor="black",
            lw=0.5,
        )
        ax.add_patch(oval)

        x, y = oval.center
        centers.append((x, y))
        # text shows up nicely in black boxes
        ax.text(
            x,
            y,
            f"{lm.subject}",
            ha="center",
            va="center",
            fontsize=8,
            color="#ff0055",
            bbox=dict(facecolor="black", edgecolor="#550000", boxstyle="round,pad=0.2"),
        )
    landmark_proxy = patches.Ellipse(
        facecolor=None,
        edgecolor="#550000",
        width=0.0,
        height=0.0,
        xy=(0, 0),
        label="Landmarks",
    )
    ax.add_patch(landmark_proxy)

    ## Set up axes limits
    # they should be consistent, and at least large enough to admit all the landmarks
    # and reasonable trajectories
    xlim = (min(c[0] for c in centers), max(c[0] for c in centers))
    xrange = xlim[1] - xlim[0]
    ylim = (min(c[1] for c in centers), max(c[1] for c in centers))
    yrange = ylim[1] - xlim[0]
    padding = (xrange * 0.1, yrange * 0.1)

    ax.set_xlim(xmin=xlim[0] - padding[0], xmax=xlim[1] + padding[0])
    ax.set_ylim(ymin=ylim[0] - padding[1], ymax=ylim[1] + padding[1])


def plot_trajectories_pretty(
    ds: Dataset,
    fig: Figure,
    title: str,
    n_seconds_per_arrow: int = 10,
) -> tuple[Figure, Axes]:
    """
    Show a map of the environment, with a given predicted trajectory plotted
    alongside the ground truth trajectory

    :param ds: full robot dataset
    :param traj: predicted robot trajectory, in the same format as ds.groundtruth
    :param label: descriptive name for this trajectory
    :param traj: another trajectory along with its label, if desired.
    no point making this a list cuz if there's more than 2 the plot
    gets impossible to read
    """
    ax = fig.subplots()
    plot_map_colored_obstacles(ds, ax)

    ### plot actual trajectories
    _plot_trajectory(
        ax,
        "Groundtruth Traj.",
        ds.ground_truth,
        n_seconds_per_arrow=n_seconds_per_arrow,
        color="#bbbbff",
        start_color="#09ff00",
        end_color="#3232e4",
    )

    ## Set up the legend & labels
    ax.plot([], [], " ", label=f"*arrows are {n_seconds_per_arrow}s apart")
    fig.subplots_adjust(right=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    return fig, ax


def _plot_trajectory(
    ax: Axes,
    label: str,
    traj: pd.DataFrame,
    color: str,
    start_color: str,
    end_color: str,
    n_seconds_per_arrow: float,
) -> None:
    """
    plots a robot trajectory, with some helpful arrows as it progresses
    """
    ax.plot(
        traj["x_m"],
        traj["y_m"],
        linewidth=0.49,
        color=color,
    )

    ### plot arrows along the path

    # calculate how to distribute our arrows on the trajectory
    # this is based on the data rate of each, so each arrow represents
    # that a certain duration has elapsed
    samples_per_second = len(traj) / (traj.iloc[-1]["time_s"] - traj.iloc[0]["time_s"])
    samples_per_arrow = int(samples_per_second * n_seconds_per_arrow)

    length = 0.2  # in inches
    arrow_locs = traj.iloc[::samples_per_arrow]
    dx = np.cos(arrow_locs["orientation_rad"]) * length
    dy = np.sin(arrow_locs["orientation_rad"]) * length

    ax.quiver(
        arrow_locs["x_m"],
        arrow_locs["y_m"],
        dx,
        dy,
        units="inches",
        angles="xy",
        scale=10,
        scale_units="width",
        color=color,
        width=0.05,
        label=label,
    )

    # add a start vector and end vector
    ax.quiver(
        traj.iloc[0]["x_m"],
        traj.iloc[0]["y_m"],
        np.cos(traj.iloc[0]["orientation_rad"]),
        np.sin(traj.iloc[0]["orientation_rad"]),
        color=start_color,
        label=f"{label} START",
        zorder=2.5,
    )
    ax.quiver(
        traj.iloc[-1]["x_m"],
        traj.iloc[-1]["y_m"],
        np.cos(traj.iloc[-1]["orientation_rad"]),
        np.sin(traj.iloc[-1]["orientation_rad"]),
        color=end_color,
        label=f"{label} END",
        zorder=2.5,
    )
