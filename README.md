# SVM for Landmark Observability Prediction

**Author:** Conor Hayes  
Written for CS/ME469: Machine Learning and Artificial Intelligence for Robotics, Northwestern University ([Prof. Brenna Argall](https://www.argallab.northwestern.edu/people/brenna/))

---

## Overview

This repo implements a Support Vector Machine (SVM) classifier with an RBF kernel to predict landmark visibility for a mobile wheeled robot, given the robot's current pose (x, y, θ). It is applied to real-world data from the [UTIAS Multi-Robot Cooperative Localization and Mapping Dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html).

The SVM is implemented from scratch; `scikit-learn` is used only for evaluation utilities (confusion matrix display) and comparison experiments.

A full writeup including problem framing, algorithm derivation, and discussion of results is available in [writeup.pdf](writeup.pdf).

The implementation includes:
- A **from-scratch SVM classifier** with RBF kernel, solving the quadratic programming problem via the `clarabel` solver (through `qpsolvers`)
- A **dataset preprocessing pipeline** that generates binary visibility labels from raw range measurements using a 2-second sliding window
- A **multi-label classifier** that trains one SVM per landmark (N=15 landmarks) and aggregates results
- A **grid search** over hyperparameters C and σ to select the best model

![Classifier output vs. ground truth on the test set](figures/B%20-%20best%20accuracy%203D%20plot.png)

*Fig B2 (see writeup) — Predicted landmark visibility (top) vs. ground truth (bottom) on the test set across all 15 landmarks. The classifier's predictions closely match ground truth (97% accuracy, 88% recall).*

---

## Install

```bash
uv sync
```

Requires Python 3.13+.

Alternatively, with pip:

```bash
pip install -r requirements.txt
```

---

## Run

```bash
uv run run.py
```

This outputs all plots from the writeup. Close each plot window to advance to the next.

To save all figures to the `figures/` directory instead of displaying interactively:

```bash
uv run run.py -s
```

---

## Data

Place the dataset files in `data/ds1/`:

```
data/ds1/
  ds1_Barcodes.dat
  ds1_Control.dat
  ds1_Groundtruth.dat
  ds1_Landmark_Groundtruth.dat
  ds1_Measurement.dat
```

Files are available on the [UTIAS Multi-Robot Cooperative Localization and Mapping Dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html) site, under MRCLAM_Dataset9, Robot3.

A preprocessed `learning_dataset.csv` is also included in the repo root for convenience.

---

## Repo Contents

| File/Folder | Description |
|-------------|-------------|
| `run.py` | Entry point; generates all figures from the writeup |
| `hw2/svm.py` | From-scratch SVM classifier with RBF kernel (via `clarabel`/`qpsolvers`) |
| `hw2/data.py` | Dataset loading, sliding-window observability generation, train/test split |
| `hw2/plot.py` | All plotting functions (maps, 3D state-space views, ridge plots) |
| `hw2/trials.py` | Classifier trial and grid search evaluation utilities |
| `hw2/test_svm.py` | Unit tests for the SVM implementation |
| `hw2/test_data.py` | Unit tests for data loading and preprocessing |
| `learning_dataset.csv` | Preprocessed observability dataset (generated from ds1) |
| `citations.txt` | References |

---

## Key Results

- The **RBF kernel** was selected after an initial qualitative exploration of kernel types, motivated by the observation that visibility regions in (x, y, θ) space are not linearly separable.
- Robot orientation is encoded as **(sin θ, cos θ)** rather than raw θ to avoid angle-wrapping discontinuities in the input space.
- **Grid search** over C ∈ {0.1, 1, 10, 100} and σ ∈ {0.1, 0.5, 1, 2, 5} identified **(C=10, σ=0.1)** as the best configuration, balancing accuracy and recall while limiting overfitting risk.
- The final classifier achieves **97% accuracy** and **88% recall** on a held-out 20% test set (randomly shuffled to avoid trajectory-ordering bias).
- Recall is the more informative metric here, since the class imbalance (most landmarks are invisible most of the time) means a trivial all-negative classifier would score deceptively high on accuracy alone.

---

## Acknowledgements

- Thanks to [Prof. Brenna Argall](https://www.argallab.northwestern.edu/people/brenna/) for the assignment from her course *Machine Learning and Artificial Intelligence for Robotics* (CS/ME469) at Northwestern University.
- Thanks to the UTIAS Multi-Robot Cooperative Localization and Mapping Dataset team for the curated dataset.
- SVM reference: Burges, C.J. *A Tutorial on Support Vector Machines for Pattern Recognition.* Data Mining and Knowledge Discovery 2, 121–167 (1998).
- QP solver: Caron et al. [qpsolvers](https://github.com/qpsolvers/qpsolvers), v4.8.1.