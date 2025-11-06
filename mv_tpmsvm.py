"""Python implementation of the multi-view twin parametric margin SVM (MvTPMSVM).

This module is a line-by-line port of the original MATLAB implementation with a
lightweight cross-view attention mechanism.  The attention module adaptively
reweights the predictions coming from view A and view B before the final
classification decision is made.

The solver relies on CVXOPT for the quadratic programming sub-problems and
NumPy for matrix manipulation.  Both packages are widely available in the
scientific Python ecosystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from cvxopt import matrix, solvers


solvers.options["show_progress"] = False


def kernel_function(kernel_type: str, x_train: np.ndarray, x_test: Optional[np.ndarray], kernel_pars: Iterable[float]) -> np.ndarray:
    """Compute kernel matrix.

    Parameters
    ----------
    kernel_type: str
        Either ``"rbf"``, ``"lin"`` or ``"poly"``.
    x_train: np.ndarray
        Matrix of training samples of shape (n_samples, n_features).
    x_test: np.ndarray
        Matrix of evaluation samples.  When ``x_test`` is ``None`` the kernel
        matrix is computed for ``x_train`` against itself.
    kernel_pars: Iterable[float]
        Parameters used by the kernel function.
    """

    if x_test is None:
        x_test = x_train

    if kernel_type == "rbf":
        gamma = float(kernel_pars[0])
        x_norm = np.sum(x_train ** 2, axis=1)[:, None]
        y_norm = np.sum(x_test ** 2, axis=1)[None, :]
        kernel = x_norm + y_norm - 2.0 * x_train @ x_test.T
        kernel = np.exp(-kernel / (2.0 * gamma))
        return kernel

    if kernel_type == "lin":
        return x_train @ x_test.T

    if kernel_type == "poly":
        coef0 = float(kernel_pars[0])
        degree = int(kernel_pars[1])
        return (x_train @ x_test.T + coef0) ** degree

    raise ValueError(f"Unsupported kernel type: {kernel_type}")


@dataclass
class JudgementResult:
    accuracy: float
    recall: float
    precision: float
    f1_score: float
    g_means: float


def judgement(predicted_label: np.ndarray, real_label: np.ndarray) -> JudgementResult:
    """Compute standard classification metrics."""
    predicted = predicted_label.astype(int)
    truth = real_label.astype(int)

    tp = int(np.sum((predicted == 1) & (truth == 1)))
    tn = int(np.sum((predicted == -1) & (truth == -1)))
    fp = int(np.sum((predicted == 1) & (truth == -1)))
    fn = int(np.sum((predicted == -1) & (truth == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100 if total else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    g_means = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) * 100 if (tp + fn) and (tn + fp) else 0.0

    return JudgementResult(accuracy, recall, precision, f1_score, g_means)


def _build_bounds(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert box bounds into inequality constraint matrices."""
    identity = np.eye(len(lb))
    lower = -identity
    upper = identity

    lb_vector = -lb
    ub_vector = ub
    return np.vstack((upper, lower)), np.concatenate((ub_vector, lb_vector))


def solve_qp(h: np.ndarray, f: np.ndarray, a: Optional[np.ndarray], b: Optional[np.ndarray], lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Solve a convex quadratic program using CVXOPT.

    The problem solved is ``0.5 x^T H x + f^T x`` subject to ``Ax <= b`` and
    ``lb <= x <= ub``.
    """

    if h.size == 0:
        raise ValueError("H must be non-empty")

    num_vars = h.shape[0]

    g_box, h_box = _build_bounds(lb, ub)

    if a is not None and a.size:
        g = np.vstack((a, g_box))
        h_vec = np.concatenate((b, h_box))
    else:
        g = g_box
        h_vec = h_box

    h = np.asarray(h, dtype=float)
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    h_vec = np.asarray(h_vec, dtype=float)

    h_mat = matrix(h)
    f_vec = matrix(f)
    g_mat = matrix(g)
    h_mat_vec = matrix(h_vec)

    solution = solvers.qp(h_mat, f_vec, g_mat, h_mat_vec)
    return np.array(solution["x"]).reshape(num_vars)


def _round_vector(vector: np.ndarray, num_truncation: int) -> np.ndarray:
    return np.round(vector, num_truncation)


def apply_attention(view_a_logits: np.ndarray, view_b_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply a simple attention mechanism to combine logits from both views.

    Attention weights are computed per-sample using a softmax over the
    magnitude of each view's logits.  Higher magnitude implies higher
    confidence and therefore a larger attention weight.
    """

    stacked = np.stack((view_a_logits, view_b_logits), axis=1)
    scores = np.abs(stacked) / max(temperature, 1e-6)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    combined = np.sum(weights * stacked, axis=1)
    return combined


@dataclass
class MvTPMSVMResult:
    accuracy: float
    accuracy_view_a: float
    accuracy_view_b: float
    recall: float
    precision: float
    f1_score: float
    g_means: float
    time_elapsed: float


def mv_tpmsvm(data1: np.ndarray,
              data2: np.ndarray,
              test_set1: np.ndarray,
              test_set2: np.ndarray,
              ca1: float,
              cb1: float,
              ca2: float,
              cb2: float,
              d1: float,
              d2: float,
              kernelparam_a: Iterable[float],
              kernelparam_b: Iterable[float],
              epsilon: float = 0.1,
              kerneltype: str = "rbf") -> MvTPMSVMResult:
    """Train and evaluate the MvTPMSVM with an attention fusion step."""

    import time

    num_truncation = 6
    start_time = time.time()

    xa = data1[:, :-1]
    xb = data2[:, :-1]
    y = data2[:, -1]

    a1 = xa[y == 1]
    b1 = xa[y == -1]
    a2 = xb[y == 1]
    b2 = xb[y == -1]

    l_p = a1.shape[0]
    l_n = b1.shape[0]

    e_p = np.eye(l_p)
    e_n = np.eye(l_n)
    o_p = np.zeros((l_p, l_p))
    o_n = np.zeros((l_n, l_n))

    p1 = kernel_function(kerneltype, a1, None, kernelparam_a)
    h1 = np.block([[p1, -p1, -p1, o_p],
                   [-p1, p1, p1, o_p],
                   [-p1, p1, p1, o_p],
                   [o_p, o_p, o_p, o_p]])

    p2 = kernel_function(kerneltype, a2, None, kernelparam_b)
    h2 = np.block([[p2, -p2, o_p, p2],
                   [-p2, p2, o_p, -p2],
                   [o_p, o_p, o_p, o_p],
                   [p2, -p2, o_p, p2]])
    h = h1 + h2 + epsilon * np.eye(h1.shape[0])

    b2_a2 = kernel_function(kerneltype, b2, a2, kernelparam_b)
    b1_a1 = kernel_function(kerneltype, b1, a1, kernelparam_a)
    linear_block_p = np.hstack((
        b2_a2 - b1_a1,
        -b2_a2 + b1_a1,
        b1_a1,
        b2_a2,
    ))
    linear_obj_p = ca1 * (np.ones((1, l_n)) @ linear_block_p).ravel()

    a_matrix = np.hstack((e_p, e_p, np.zeros((l_p, l_p)), np.zeros((l_p, l_p))))
    b_vector = d1 * np.ones(l_p)

    lb = np.zeros(4 * l_p)
    ub = np.concatenate((np.zeros(2 * l_p), np.ones(l_p) * ca1, np.ones(l_p) * cb1))

    pai = solve_qp(h, linear_obj_p.astype(float), a_matrix, b_vector, lb, ub)
    pai = _round_vector(pai, num_truncation)

    p1_b = kernel_function(kerneltype, b1, None, kernelparam_a)
    h1_b = np.block([[p1_b, -p1_b, p1_b, o_n],
                     [-p1_b, p1_b, -p1_b, o_n],
                     [p1_b, -p1_b, p1_b, o_n],
                     [o_n, o_n, o_n, o_n]])

    p2_b = kernel_function(kerneltype, b2, None, kernelparam_b)
    h2_b = np.block([[p2_b, -p2_b, o_n, p2_b],
                     [-p2_b, p2_b, o_n, -p2_b],
                     [o_n, o_n, o_n, o_n],
                     [p2_b, -p2_b, o_n, p2_b]])
    h_b = h1_b + h2_b + epsilon * np.eye(h1_b.shape[0])

    a1_b1 = kernel_function(kerneltype, a1, b1, kernelparam_a)
    a2_b2 = kernel_function(kerneltype, a2, b2, kernelparam_b)
    linear_block_f = np.hstack((
        a1_b1 + a2_b2,
        -a1_b1 - a2_b2,
        a1_b1,
        a2_b2,
    ))
    linear_obj_f = cb1 * (np.ones((1, l_p)) @ linear_block_f).ravel()

    a_matrix_f = np.hstack((np.zeros((l_n, l_n)), np.zeros((l_n, l_n)), e_n, e_n))
    b_vector_f = d2 * np.ones(l_n)

    lb_f = np.zeros(4 * l_n)
    ub_f = np.concatenate((np.ones(l_n) * ca2, np.ones(l_n) * cb2, d2 * np.ones(2 * l_n)))

    fai = solve_qp(h_b, linear_obj_f.astype(float), a_matrix_f, b_vector_f, lb_f, ub_f)
    fai = _round_vector(fai, num_truncation)

    test_xa = test_set1[:, :-1]
    test_xb = test_set2[:, :-1]
    test_y = test_set1[:, -1]

    kermat_a = kernel_function(kerneltype, test_xa, a1, kernelparam_a)
    kermat_aa = kernel_function(kerneltype, test_xa, b1, kernelparam_a)
    kermat_b = kernel_function(kerneltype, test_xb, a2, kernelparam_b)
    kermat_bb = kernel_function(kerneltype, test_xb, b2, kernelparam_b)

    origin_view_ap = np.hstack((-kermat_a, kermat_a, kermat_a, np.zeros_like(kermat_a))) @ pai - ca1 * (kermat_aa @ np.ones(l_n))
    origin_view_an = cb1 * (kermat_a @ np.ones(l_p)) - np.hstack((-kermat_aa, kermat_aa, -kermat_aa, np.zeros_like(kermat_aa))) @ fai

    origin_view_bp = np.hstack((kermat_b, -kermat_b, np.zeros_like(kermat_b), kermat_b)) @ pai - ca1 * (kermat_bb @ np.ones(l_n))
    origin_view_bn = cb1 * (kermat_b @ np.ones(l_p)) - np.hstack((-kermat_bb, kermat_bb, -kermat_bb, np.zeros_like(kermat_bb))) @ fai

    origin_result_a = np.abs(origin_view_an) - np.abs(origin_view_ap)
    origin_result_b = np.abs(origin_view_bn) - np.abs(origin_view_bp)

    combined_result = apply_attention(origin_result_a, origin_result_b)
    binary_result = np.sign(combined_result)
    binary_a = np.sign(origin_result_a)
    binary_b = np.sign(origin_result_b)

    metrics = judgement(binary_result, test_y)
    metrics_a = judgement(binary_a, test_y)
    metrics_b = judgement(binary_b, test_y)

    elapsed = time.time() - start_time

    return MvTPMSVMResult(
        accuracy=metrics.accuracy,
        accuracy_view_a=metrics_a.accuracy,
        accuracy_view_b=metrics_b.accuracy,
        recall=metrics.recall,
        precision=metrics.precision,
        f1_score=metrics.f1_score,
        g_means=metrics.g_means,
        time_elapsed=elapsed,
    )
