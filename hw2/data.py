"""
Code for loading in the experimental data
"""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field, fields
from enum import Enum, Flag, auto
import pathlib

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """
    Container for the data in the UTIAS Multi-Robot Cooperative Localization and Mapping Datasets

    Originally sourced from here: http://asrl.utias.utoronto.ca/datasets/mrclam/index.html
    And included in the course canvas website here: https://canvas.northwestern.edu/courses/239182/files/folder/Datasets
    """

    path: pathlib.Path

    barcodes: pd.DataFrame
    control: pd.DataFrame
    ground_truth: pd.DataFrame
    landmarks: pd.DataFrame
    measurement: pd.DataFrame

    # measurement file, but using the landmark ground truth
    # subject labels (translated using barcodes)
    # (this way landmarks & measurements can be used together more easily)
    measurement_fix: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        # rectify labels between measurement & ground truth

        # in barcodes data file:
        # "subject" = landmark subject  <- what we want to use (cuz they're nicer numbers)
        # "barcode" = measurement subject
        bc = self.barcodes

        msr_subj_to_landm_subj = pd.Series(
            bc["subject"].values, index=bc["barcode"].values
        )

        fix = self.measurement.copy()
        fix["subject"] = self.measurement["subject"].map(msr_subj_to_landm_subj)
        # also, extremely annoyingly, there are measurements corresponding
        # to landmarks which we don't have groundtruth values for.
        # remove these here.
        valid_subjects = self.landmarks["subject"].to_list()
        fix_cleaned = fix[fix["subject"].isin(valid_subjects)]

        print(f"Removed {len(fix) - len(fix_cleaned)} invalid measurements.")

        self.measurement_fix = fix_cleaned

    @classmethod
    def from_dataset_directory(cls, p: pathlib.Path) -> Dataset:
        """
        Load in

        :param p: path to the dataset directory
        :return: Dataset instance
        """
        # grab the prefix from directory name
        # this isn't particularly robust but it works fine for
        # this assignment
        prefix = p.stem
        barcodes = pd.read_csv(
            p / f"{prefix}_Barcodes.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["subject", "barcode"],
        )

        control = pd.read_csv(
            p / f"{prefix}_Control.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["time_s", "forward_velocity_mps", "angular_velocity_radps"],
        )

        groundtruth = pd.read_csv(
            p / f"{prefix}_Groundtruth.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["time_s", "x_m", "y_m", "orientation_rad"],
        )

        landmark = pd.read_csv(
            p / f"{prefix}_Landmark_Groundtruth.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["subject", "x_m", "y_m", "x_std_dev", "y_std_dev"],
        )

        measurement = pd.read_csv(
            p / f"{prefix}_Measurement.dat",
            sep=r"\s+",
            engine="python",
            comment="#",
            names=["time_s", "subject", "range_m", "bearing_rad"],
        )

        return cls(p, barcodes, control, groundtruth, landmark, measurement)

    def segment(
        self, t0: float, tf: float, normalize_timestamps: bool = False
    ) -> Dataset:
        """
        Make a copy of the dataset where only the timestamps in the range
        [t0, tf) are included
        Timestamps are normalized to t0 on request
        """
        ds_dict = asdict(self)
        for name in ds_dict:
            if type(ds_dict[name]) is not pd.DataFrame:
                continue
            df: pd.DataFrame = ds_dict[name]
            if "time_s" not in df.columns:
                ds_dict[name] = df.copy()
                continue

            print(f"Segmenting {name}...")
            new_df = (
                df[(t0 <= df["time_s"]) & (df["time_s"] < tf)]
                .copy()
                .reset_index(drop=True)
            )
            if normalize_timestamps:
                new_df["time_s"] -= t0
            ds_dict[name] = new_df

        ds_dict.pop("measurement_fix")

        return Dataset(**ds_dict)

    def segment_percent(
        self, p0: float, pf: float, normalize_timestamps: bool = False
    ) -> Dataset:
        """
        Returns copy of the dataset such that we only include entries in the range
        [p0%, pf%), where both p0 and pf represent percentages of the dataset.
        for instance,
        - segment_percent(0, 1) gets the whole dataset;
        - segment_percent (.25, .5) gets the 2nd quantile of timestamped entries

        timestamp ranges are taken from groundtruth and applied to all.

        :param p0: start percent expressed from 0-1
        :param pf: end percent expressed from 0-1
        :param normalize_timestamps: if true, start timestamps at 0
        """
        t_start = self.ground_truth["time_s"].iloc[0]
        t_end = self.ground_truth["time_s"].iloc[-1]
        range = t_end - t_start

        t0 = t_start + (p0 * range)
        tf = t_start + (pf * range)

        return self.segment(t0, tf, normalize_timestamps)

    def print_info(self) -> None:
        """
        print some helpful info about the dataset
        """
        out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

        # measurement & control frequency
        out += self._fmt_timeseries_info(self.measurement, "measurement")
        out += self._fmt_timeseries_info(self.control, "control")

        out += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print(out)

    @staticmethod
    def _fmt_timeseries_info(ser: pd.DataFrame, name: str) -> str:
        out = "\n"
        n_meas = len(ser)
        t0_meas = ser["time_s"].iloc[0]
        tf_meas = ser["time_s"].iloc[1]
        out += f"{name:_<12} n={n_meas: <4} | t=({t0_meas:.3f}, {tf_meas:.3f})\n"

        # period info
        periods = ser["time_s"].diff()
        out += "PERIOD INFO\n"
        out += str(periods.agg(["mean", "min", "max", "std"]))
        out += "\n"

        return out


class Opts(Flag):
    """Options for how to preprocess observability data."""

    CONTINUOUS_ROT = auto()
    """Use sin(theta), cos(theta) rather than theta."""
    SHUFFLE = auto()
    """Randomly shuffle the states so they're not in timeseries order."""

    NONE = auto()
    """don't do anything (used as a placeholder for None.)"""


@dataclass
class ObservabilityData:
    """
    Observability dataset, containing pose data+labels describing what landmarks are visible at time t.
    """

    source_ds: Dataset
    """Source dataset."""
    freq_hz: float
    """Frequency of the derived measurements."""
    sliding_window_len_s: float
    """Length of the sliding window.
    
    landmark visibility is determined by whether a measurement of that landmark has occurred
    during this window prior to each time t.
    """

    data: pd.DataFrame = field(init=False)
    """
    columns: time_s, x_m, y_m, orientation_rad, landmarks

    landmarks is a dict[int, int] mapping (lm subj) -> # of times lm seen in window
    """

    data_long: pd.DataFrame = field(init=False)
    """
    columns: time_s, x_m, y_m, orientation_rad, subject, landmark_count

    The same data as `data` in "long" form, useful for plotting in seaborn
    (this is super annoying)
    """

    data_long_unwindowed: pd.DataFrame = field(init=False)
    """
    columns: time_s, x_m, y_m, orientation_rad, subject

    each row represents a single measurement from measurement.dat, at whatever frequency it has.
    the robot state is taken from the nearest neighbor interpolation in the groundtruths
    """

    def __post_init__(self) -> None:
        # sanity check -- if the sample period is longer than the sliding window, we
        # will miss samples
        if self.sliding_window_len_s < (1 / self.freq_hz):
            raise ValueError("Sliding window len cannot be < 1/freq")

        # generate base dataset used for training
        self._generate_data()
        self._generate_data_long()
        self._generate_data_unwindowed()

    def preprocess(
        self,
        opts: Opts = Opts.NONE,
        test_size: float = 0.2,
        asdict: bool = False,
        boolval: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into test & training data+labels

        Args:
            test_size: % of the dataset to reserve as testing data
            subject: if true, return landmarks as dictionary. if false,
            break each landmark into its own column.
            boolval: if true, and subject=True, make the values of each lm
            column either 0 or 1, rather than [0, N] for max N observations.

        Returns:
            (X_train, X_test, y_train, y_test)
        """

        def lm_map(lm: pd.Series, subj: int) -> int:
            if boolval:
                return 1 if lm[subj] > 0 else 0
            else:
                return lm[subj]

        X = self.data[["x_m", "y_m", "orientation_rad"]].copy()
        if Opts.CONTINUOUS_ROT in opts:
            X["sin"] = np.sin(X["orientation_rad"])
            X["cos"] = np.cos(X["orientation_rad"])
            del X["orientation_rad"]

        if asdict:
            y = self.data[["landmarks"]]
        else:
            y = pd.DataFrame()
            for subj in self.data["landmarks"].iloc[0].keys():
                y[subj] = self.data["landmarks"].map(lambda lm: lm_map(lm, subj))

        if Opts.SHUFFLE in opts:
            perm = np.random.permutation(len(X))
            X = X.iloc[perm].reset_index(drop=True)
            y = y.iloc[perm].reset_index(drop=True)

        N = int(len(X) * (1 - test_size))
        return X.iloc[:N], X.iloc[N:], y.iloc[:N], y.iloc[N:]

    def _generate_data(self) -> None:
        # make 1 column per obstacle that appears in the measurement dataset
        ds = self.source_ds
        msr = ds.measurement_fix
        subjects = ds.measurement_fix["subject"].unique()
        time_grid = np.arange(
            msr["time_s"].iloc[0], msr["time_s"].iloc[-1], 1 / self.freq_hz
        )
        df = pd.DataFrame({"time_s": time_grid})

        # add ground truth pose data, grabbing the nearest value to each timestamp
        df = pd.merge_asof(
            df,
            ds.ground_truth[["time_s", "x_m", "y_m", "orientation_rad"]],
            on="time_s",
            direction="nearest",
        )

        df["landmarks"] = [dict() for _ in range(len(df))]

        for subj in subjects:
            # For each subject, get a boolean mask for its rows
            times = msr.loc[msr["subject"] == subj, "time_s"].to_numpy()  # type: ignore

            # For each timestamp in the grid, count how many subject occurrences
            # fall within the sliding window [t - window, t)
            counts = np.array(
                [
                    ((times >= t - self.sliding_window_len_s) & (times < t)).sum()
                    for t in time_grid
                ]
            )
            for idx in range(len(counts)):
                df["landmarks"].iloc[idx][int(subj)] = int(counts[idx])

        self.data = df

    def _generate_data_long(self) -> None:
        # generate the long form of the data dataframe
        rows = []
        for idx, ser in self.data.iterrows():
            for lm, count in ser["landmarks"].items():
                rows.append(
                    dict(
                        time_s=ser["time_s"],
                        x_m=ser["x_m"],
                        y_m=ser["y_m"],
                        orientation_rad=["orientation_rad"],
                        subject=lm,
                        count=count,
                    )
                )
        self.data_long = pd.DataFrame(rows)

    def _generate_data_unwindowed(self) -> None:
        ds = self.source_ds
        df = pd.DataFrame(
            data={
                "time_s": ds.measurement_fix["time_s"],
                "subject": ds.measurement_fix["subject"],
            }
        )

        # add ground truth pose data, grabbing the nearest value to each timestamp
        df = pd.merge_asof(
            df,
            ds.ground_truth[["time_s", "x_m", "y_m", "orientation_rad"]],
            on="time_s",
            direction="nearest",
        )

        self.data_long_unwindowed = df

    def to_file(self, path: pathlib.Path | None = None) -> None:
        if path is None:
            path = self.source_ds.path / "learning_dataset.csv"
        self.data.to_csv(path, index=False)
