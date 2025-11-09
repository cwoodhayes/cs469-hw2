"""
plotting functions for robot data
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib import patches
from matplotlib.ticker import MultipleLocator

from hw1.map import Map
from hw1 import astar


def plot_map(
    map: Map, ax: Axes, groundtruth_map: Map | None = None, set_limits: bool = True
) -> None:
    ##### Plot the obstacles
    patch = None
    obs_locs = map.get_obstacle_locs()
    for loc in obs_locs:
        obs = map.grid_loc_to_world_coords_corner(loc)
        patch = patches.Rectangle(
            obs,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            facecolor="black",
        )
        ax.add_patch(patch)
    if patch is not None:
        patch.set_label("Obstacle")

    ### If supplied, plot ground-truth obstacles behind those in light grey
    if groundtruth_map is not None:
        patch = None
        obs_locs = groundtruth_map.get_obstacle_locs()
        for loc in obs_locs:
            obs = map.grid_loc_to_world_coords_corner(loc)
            patch = patches.Rectangle(
                obs,  # type: ignore
                map.c.cell_size,
                -map.c.cell_size,
                facecolor="#00000037",
            )
            ax.add_patch(patch)
        if patch is not None:
            patch.set_label("Undiscovered Obstacle")

    ##### Plot start and goal
    goal_corner = map.grid_loc_to_world_coords_corner(map._goal_loc)
    start_corner = map.grid_loc_to_world_coords_corner(map._start_loc)
    ax.add_patch(
        patches.Rectangle(
            goal_corner,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            color="#FFD90099",
            label="Goal",
        )
    )
    ax.add_patch(
        patches.Rectangle(
            start_corner,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            color="#00FF5599",
            label="Start",
        )
    )

    ##### Make limits and grid
    if set_limits:
        xlim = map.c.dimensions[0, :]
        ylim = map.c.dimensions[1, :]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax.grid(True)
    ax.xaxis.set_minor_locator(MultipleLocator(map.c.cell_size))
    ax.yaxis.set_minor_locator(MultipleLocator(map.c.cell_size))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))


def plot_path_on_map(
    map: Map,
    ax: Axes,
    p: astar.Path,
    groundtruth_map: Map | None = None,
    plot_centers: bool = True,
    show_full_map: bool = True,
    is_superimposed: bool = False,
) -> None:
    """
    plots the robot path discoverd by A* on the map.

    if supplied, groundtruth_map supplies the obstacles that were
    not discovered by the robot, which are displayed in grey
    """
    if is_superimposed:
        path_color = "#E5D54537"
    else:
        plot_map(map, ax, groundtruth_map, set_limits=show_full_map)
        path_color = "#4590E57B"

    # Fill in every cell visited in light blue
    centers = []
    rect = None
    for loc in p.locs:
        corner = map.grid_loc_to_world_coords_corner(loc)
        rect = patches.Rectangle(
            corner,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            color=path_color,
        )
        ax.add_patch(rect)
        center_x = corner[0] + map.c.cell_size / 2
        center_y = corner[1] - map.c.cell_size / 2
        centers.append((center_x, center_y))

    if rect is not None:
        if is_superimposed:
            rect.set_label("Robot Path (coarse grid)")
        else:
            rect.set_label("Robot Path")

    c_arr = np.array(centers)

    if plot_centers:
        ax.plot(c_arr[:, 0], c_arr[:, 1], "bo-", ms=4, label="Robot Path")

    # add a single dot for start and goal colors (for when the grid cell is small)
    ax.plot(c_arr[0, 0], c_arr[0, 1], marker="o", color="#00FF55", ms=10, zorder=1.1)
    ax.plot(c_arr[-1, 0], c_arr[-1, 1], marker="o", color="#FFD900", ms=10, zorder=1.1)

    if not show_full_map:
        ax.dataLim.update_from_data_xy(c_arr, ignore=True)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="datalim")


def plot_trajectory_over_waypoints(
    ax: Axes,
    traj: np.ndarray | None,
    waypoints: np.ndarray,
    distance_threshold: float,
    secondary_trajectory: bool = False,
) -> None:
    """
    Plot a controlled robot trajectory

    and the waypoints it's attempting to reach.

    :param ax: plt axes
    :param traj: trajectory: [[x, y], ...]
    :param waypoints: list of control target waypoints [[x, y], ...]
    :param distance_threshold: "close enough" radius used to evaluate
        whether a waypoint was reached
    """

    ax.scatter(waypoints[:, 0], waypoints[:, 1], c="#BB4C4C", s=8, label="Waypoint")
    if traj is not None and traj.size > 0:
        if secondary_trajectory:
            ax.plot(
                traj[:, 0], traj[:, 1], "ro-", ms=3, label="Robot Path (coarse grid)"
            )
        else:
            ax.plot(traj[:, 0], traj[:, 1], "bo-", ms=3, label="Robot Path")

    c = None
    for wp in waypoints:
        c = patches.Circle(
            wp, distance_threshold, edgecolor="#4F8FF6", facecolor=(1, 1, 1, 0)
        )
        ax.add_patch(c)
    if c is not None:
        if not secondary_trajectory:
            c.set_label("Waypoint radius")
