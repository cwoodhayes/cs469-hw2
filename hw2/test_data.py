"""
pytest tests for data.py
"""

from pathlib import Path

from hw2.data import Dataset, ObservabilityData

REPO_ROOT = Path(__file__).parent.parent


def test_from_dataset_directory():
    p = REPO_ROOT / "data/ds1"
    ds = Dataset.from_dataset_directory(p)

    assert ds.barcodes.shape == (20, 2)


def test_preprocess():
    p = REPO_ROOT / "data/ds1"
    ds = Dataset.from_dataset_directory(p)

    # try default values
    obs = ObservabilityData(ds, freq_hz=1.0, sliding_window_len_s=2.0)

    assert len(obs.data) > 0
    assert len(obs.data_long) > 0
    # assert len(obs.data_long_unwindowed) == len(ds.measurement_fix)
