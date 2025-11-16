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


def plot_map_colored_obstacles(
    ds: Dataset, ax: Axes, unseen: set | list | None = None
) -> dict[int, tuple[float]]:
    """
    Generate a view of the robot's arena where each obstacle is a different color.

    Args:
        ds (Dataset): dataset
        ax (Axes): axes to plot on
        unseen (set | None): set of landmark id's which aren't visible (ie greyed out)

    Returns:
        dict[int, tuple[float]]: map of landmark subj. # to RGBA tuple
    """
    ## generate dictionary of landmarks to colors
    N = len(ds.landmarks)
    cmap = cm.get_cmap("tab20", N)  # or tab10, Set3, Dark2, Paired, etc.
    colors = [cmap(i) for i in range(N)]
    lm_subj = ds.landmarks["subject"]
    lm_to_color: dict[int, tuple[float]] = {subj: c for subj, c in zip(lm_subj, colors)}  # type: ignore

    ## plot the landmarks as colored text + bounding boxes
    centers = []
    for lm in ds.landmarks.itertuples(index=False):
        x: float = lm.x_m  # type: ignore
        y: float = lm.y_m  # type: ignore
        centers.append((x, y))

        # add colored text boxes (this is what really shows up well)
        # adjust their coloring if unseen
        if lm.subject in unseen:  # type: ignore
            edgecolor = "#00000036"
            facecolor = "#00000000"
        else:
            edgecolor = "#550000"
            facecolor = lm_to_color[lm.subject]  # type: ignore

        ax.text(
            x,
            y,
            f"{lm.subject:<2}",
            ha="center",
            va="center",
            fontsize=13,
            color="black",
            bbox=dict(
                facecolor=facecolor,
                edgecolor=edgecolor,
                boxstyle="round,pad=0.2",
            ),
        )
    visible_proxy = patches.Ellipse(
        facecolor="#00000000",
        edgecolor="#550000",
        width=0.0,
        height=0.0,
        xy=(0, 0),
        label="Visible Landmarks (colored)",
    )
    ax.add_patch(visible_proxy)
    unseen_proxy = patches.Ellipse(
        facecolor="#00000000",
        edgecolor="#00000036",
        width=0.0,
        height=0.0,
        xy=(0, 0),
        label="Unseen Landmarks (colorless)",
    )
    ax.add_patch(unseen_proxy)

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

    return lm_to_color


def plot_single_observation(ds: Dataset, ax: Axes, obs: pd.Series, title: str) -> None:
    """Plot landmarks visible from a single state on the map.

    Args:
        ds (Dataset): dataset
        ax (Axes): plt axes
        obs (pd.Series): row of the observability dataset to show
    """
    unseen = [subj for subj in obs["landmarks"] if obs["landmarks"][subj] == 0]
    plot_map_colored_obstacles(ds, ax, unseen)

    # show the robot's location & orientation
    ax.quiver(
        obs["x_m"],
        obs["y_m"],
        np.cos(obs["orientation_rad"]),
        np.sin(obs["orientation_rad"]),
        color="blue",
        label=f"Robot State (t={obs['time_s'].round(2)}s)",
        zorder=2.5,
    )

    ax.set_title(title)


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
    lm_to_c = plot_map_colored_obstacles(ds, ax)

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
