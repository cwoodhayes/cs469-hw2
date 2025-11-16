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
    obs = ObservabilityData.from_dataset(ds)

    assert len(obs.data) > 0
