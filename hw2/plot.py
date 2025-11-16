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
import seaborn as sns

from hw2.data import Dataset, ObservabilityData

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def get_landmark_to_color(ds: Dataset) -> dict[int, tuple[float]]:
    """Generate dictionary of landmarks to colors."""
    N = len(ds.landmarks)
    cmap = cm.get_cmap("tab20", N)  # or tab10, Set3, Dark2, Paired, etc.
    colors = [cmap(i) for i in range(N)]
    lm_subj = ds.landmarks["subject"]
    return {subj: c for subj, c in zip(lm_subj, colors)}  # type: ignore


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
    lm_to_c = get_landmark_to_color(ds)

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
            facecolor = lm_to_c[lm.subject]  # type: ignore

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

    return lm_to_c


def plot_single_observation(
    ds: Dataset, obs_data: ObservabilityData, ax: Axes, obs: pd.Series, label: str = ""
) -> None:
    """Plot landmarks visible from a single state on the map.

    Args:
        ds (Dataset): dataset
        ax (Axes): plt axes
        obs (pd.Series): row of the observability dataset to show
    """
    unseen = [subj for subj in obs["landmarks"] if obs["landmarks"][subj] == 0]
    plot_map_colored_obstacles(ds, ax, unseen)

    # show the robot's location & orientation
    t = obs["time_s"].round(2)
    ax.quiver(
        obs["x_m"],
        obs["y_m"],
        np.cos(obs["orientation_rad"]),
        np.sin(obs["orientation_rad"]),
        color="blue",
        label=f"Robot State (t={t}s)",
        zorder=2.5,
    )

    title = (
        f"Visible Landmarks @ {t}s (window={obs_data.sliding_window_len_s}s, "
        f"freq={obs_data.freq_hz}Hz)"
    )
    if len(label) > 0:
        title += "\n" + label
    ax.set_title(title)


def plot_landmark_bars(
    ds: Dataset,
    obs_data: ObservabilityData,
    ax: Axes,
    title: str = "",
) -> None:
    """
    Show visible landmarks over time using seaborn's facetgrid.

    Built off of this example:
    https://seaborn.pydata.org/examples/kde_ridgeplot.html
    """
    # Create the data
    rs = np.random.RandomState(1979)
    x = rs.randn(500)
    g = np.tile(list("ABCDEFGHIJ"), 50)
    df = pd.DataFrame(dict(x=x, g=g))
    m = df.g.map(ord)
    df["x"] += m

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=0.5, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        "x",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)


def plot_trajectories_pretty(
    ds: Dataset,
    fig: Figure,
    label: str,
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
    ax.set_title(label)

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
