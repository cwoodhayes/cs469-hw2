"""
Support Vector Machine implementations.
"""

import numpy as np


class SVMError(Exception):
    """Base error for SVM module."""

    pass


class SVM:
    """
    Linear-kernel support vector machine.

    Implemented following the paper:

    @inproceedings{Fletcher2010SupportVM,
        title={Support Vector Machines Explained},
        author={Tristan Fletcher},
        year={2010},
        url={https://api.semanticscholar.org/CorpusID:218451389}
    }

    With API inspiration taken from
    https://scikit-learn.org/stable/modules/svm.html

    """

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier on a dataset.

        Args:
            X (np.ndarray): training data, shape=(n_samples, n_features)
            y (np.ndarray): training labels, shape=(n_samples,)
        """

        self._fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform classification on inputs x

        Args:
            x (np.ndarray): one or many inputs x of shape (n_samples, n_features)

        Returns:
            np.ndarray: output classes, of shape (n_samples)
        """
        if not self._fitted:
            raise SVMError("fit() has not been called.")

        return np.zeros(x.shape[0])
