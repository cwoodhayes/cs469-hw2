"""
plotting functions for robot data
"""

import warnings

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.axes import Axes
from matplotlib import patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hw2.data import Dataset, ObservabilityData

RGBATuple = tuple[float, float, float, float]


def get_landmark_to_color(ds: Dataset) -> dict[int, RGBATuple]:
    """Generate dictionary of landmarks to colors."""
    N = len(ds.landmarks)
    cmap = cm.get_cmap("tab20", N)  # or tab10, Set3, Dark2, Paired, etc.
    colors = [cmap(i) for i in range(N)]
    lm_subj = ds.landmarks["subject"]
    return {subj: c for subj, c in zip(lm_subj, colors)}  # type: ignore


def plot_map_colored_obstacles(
    ds: Dataset, ax: Axes, unseen: set | list | None = None
) -> dict[int, RGBATuple]:
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

    # plot the landmarks as colored text + bounding boxes
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

    # Set up axes limits
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
    obs_data: ObservabilityData, ax: Axes, obs: pd.Series, label: str = ""
) -> None:
    """Plot landmarks visible from a single state on the map.

    Args:
        ds (Dataset): dataset
        ax (Axes): plt axes
        obs (pd.Series): row of the observability dataset to show
    """
    unseen = [subj for subj in obs["landmarks"] if obs["landmarks"][subj] == 0]
    plot_map_colored_obstacles(obs_data.source_ds, ax, unseen)

    # show the robot's location & orientation
    t = obs["time_s"].round(2)
    ax.quiver(
        obs["x_m"],
        obs["y_m"],
        np.cos(obs["orientation_rad"]),
        np.sin(obs["orientation_rad"]),
        color="blue",
        label="Robot State",
        zorder=2.5,
        scale=0.9,  # smaller scale → larger arrows
        scale_units="xy",  # interpret scale in data units
        width=0.1,  # optional: thickens the arrow shaft
    )

    title = (
        f"Visible Landmarks @ {t}s (window={obs_data.sliding_window_len_s}s, "
        f"freq={obs_data.freq_hz}Hz)"
    )
    if len(label) > 0:
        title += "\n" + label
    ax.set_title(title)


def plot_landmark_bars(
    obs_data: ObservabilityData,
    title: str = "",
    figlabel: str | None = None,
) -> None:
    """
    Show visible landmarks over time using seaborn's facetgrid.

    Built off of this example:
    https://seaborn.pydata.org/examples/kde_ridgeplot.html
    """
    # put the data in a shape seaborn's FacetGrid will recognize
    df_long = pd.DataFrame(
        dict(
            time=obs_data.data_long["time_s"],
            g=obs_data.data_long["subject"],
            x=obs_data.data_long["count"],
        )
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Tight Layout not applied.*")

        with mpl.rc_context():
            sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

            # create the color palette for seaborn
            lm_to_color = get_landmark_to_color(obs_data.source_ds)
            unique_g = df_long["g"].unique()
            pal = [lm_to_color[g_val] for g_val in unique_g]

            g = sns.FacetGrid(
                df_long,
                row="g",
                hue="g",
                aspect=15,
                height=0.5,
                palette=pal,  # type: ignore
            )

            def ridge_plot(data, color, label, **kwargs):
                ax = plt.gca()
                ax.fill_between(data["time"], data["x"], color=color, alpha=1)
                ax.plot(data["time"], data["x"], color="w", lw=1.5)
                ax.text(
                    -0.05,
                    0.1,
                    label,
                    transform=ax.transAxes,
                    fontweight="bold",
                    color=color,
                )

            g.map_dataframe(ridge_plot)
            # add reference line at count=1 for each
            g.refline(y=1, linestyle=":")

            g.figure.subplots_adjust(hspace=-0.25)
            g.set_titles("")
            g.set(yticks=[], ylabel="")
            g.despine(bottom=True, left=True)

            g.figure.suptitle(
                "Number of Observations Per Landmark vs. Time\n(dotted line = 1 "
                f"observation, window={obs_data.sliding_window_len_s}s)"
            )
            g.set_axis_labels("Time (s)", "")
            g.figure.text(
                0.05,
                0.5,
                "# of Observations (by landmark ID)",
                ha="center",
                va="center",
                rotation="vertical",
                fontsize=12,
            )
            g.figure.set_figheight(8)
            g.figure.set_figwidth(12)
            if figlabel is not None:
                g.figure.set_label(figlabel)


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
    plot_map_colored_obstacles(ds, ax)

    # plot actual trajectories
    _plot_trajectory(
        ax,
        "Groundtruth Traj.",
        ds.ground_truth,
        n_seconds_per_arrow=n_seconds_per_arrow,
        color="#bbbbff",
        start_color="#09ff00",
        end_color="#3232e4",
    )

    # Set up the legend & labels
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

    # plot arrows along the path

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


def plot_visibility_3d(obs: ObservabilityData, ax: Axes3D, subjects: set) -> None:
    """
    Plot the visibility of a set of obstacles in a 3D point cloud of states.
    """
    lm_to_c = get_landmark_to_color(obs.source_ds)

    # set all colors that aren't in subjects list to grey
    for subj in set(lm_to_c.keys()) - subjects:
        lm_to_c[subj] = (0.5, 0.5, 0.5, 0.05)

    df = obs.data_long_unwindowed
    colors = [lm_to_c[row["subject"]] for _, row in df.iterrows()]
    ax.scatter(df["x_m"], df["y_m"], df["orientation_rad"], c=colors)  # type: ignore
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel(r"$\theta$ (radians)")

    ax.set_title("Landmark visibility vs. robot state")

    # add a legend with the colors on it
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=lm_to_c[s][:3],  # RGB only
            markerfacecolor=lm_to_c[s],
            label=f"Landmark {s}",
        )
        for s in sorted(subjects)
    ]

    ax.legend(handles=legend_elements)
