"""Entry point for training and evaluating the Python port of MvTPMSVM."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from mv_tpmsvm import mv_tpmsvm


def _load_data(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(mat_path)
    xa = data["X1"]
    xb = data["X2"]
    y = data["y"].reshape(-1)
    y[y == 0] = -1
    return xa, xb, y


def main() -> None:
    xa, xb, y = _load_data(Path("aus.mat"))
    xa = np.column_stack((xa, y))
    xb = np.column_stack((xb, y))

    xa_train, xa_test, xb_train, xb_test = train_test_split(
        xa, xb, test_size=0.3, stratify=y, random_state=42
    )

    sig_best = 1.0
    c1_best = 1.0

    result = mv_tpmsvm(
        xa_train,
        xb_train,
        xa_test,
        xb_test,
        c1_best,
        c1_best,
        c1_best,
        c1_best,
        c1_best,
        c1_best,
        [sig_best],
        [sig_best],
        epsilon=0.1,
    )

    print("Overall accuracy: {:.2f}%".format(result.accuracy))
    print("View A accuracy: {:.2f}%".format(result.accuracy_view_a))
    print("View B accuracy: {:.2f}%".format(result.accuracy_view_b))
    print("Recall: {:.2f}%".format(result.recall))
    print("Precision: {:.2f}%".format(result.precision))
    print("F1-score: {:.2f}".format(result.f1_score))
    print("G-means: {:.2f}%".format(result.g_means))
    print("Elapsed time: {:.2f}s".format(result.time_elapsed))


if __name__ == "__main__":
    main()
