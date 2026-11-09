"""Auditable mean-loss KRR and compatible train/cross-kernel repair."""
import numpy as np
from scipy.linalg import eigh, solve


def preprocess(train_x, eval_x, train_y):
    median = np.median(train_x, axis=0)
    iqr = np.quantile(train_x, .75, axis=0) - np.quantile(train_x, .25, axis=0)
    varying = iqr > 1e-12
    scale = np.where(varying, iqr, 1.)
    def transform(x):
        z = np.clip((x - median) / scale, -3., 3.) / 3.
        z[:, ~varying] = 0.
        return z
    ym, ys = float(np.mean(train_y)), float(np.std(train_y, ddof=1))
    if not np.isfinite(ys) or ys <= 1e-12:
        raise ValueError('Degenerate training targets')
    return transform(train_x), transform(eval_x), (train_y - ym) / ys, {
        'median': median.tolist(), 'iqr': iqr.tolist(), 'y_mean': ym, 'y_sd': ys,
        'constant_columns': np.flatnonzero(~varying).tolist()}


def kernel_map(train, cross):
    train = (np.asarray(train) + np.asarray(train).T) * .5
    d, q = eigh(train, check_finite=True)
    threshold = 1e-8 * max(1., float(np.max(np.abs(d))))
    keep = d > threshold
    if not keep.any():
        raise ValueError('No retained positive eigenspace')
    zt = q[:, keep] * np.sqrt(d[keep])
    zv = (cross @ q[:, keep]) / np.sqrt(d[keep])
    mean = zt.mean(axis=0)
    zt, zv = zt - mean, zv - mean
    scale = float(np.mean(np.sum(zt * zt, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Degenerate centered kernel')
    zt, zv = zt / np.sqrt(scale), zv / np.sqrt(scale)
    return zt, zv, {'min_eigenvalue': float(d.min()), 'negative_eigenvalues': int((d < -threshold).sum()),
                    'retained_rank': int(keep.sum()), 'threshold': threshold, 'centered_scale': scale}


def predict_mapped(zt, zv, y, regularization):
    ym = float(np.mean(y))
    n = len(y)
    alpha = solve(zt @ zt.T + n * regularization * np.eye(n), y - ym, assume_a='pos')
    return zv @ (zt.T @ alpha) + ym


def krr(train, cross, y, regularization):
    zt, zv, diagnostics = kernel_map(train, cross)
    return predict_mapped(zt, zv, y, regularization), diagnostics


def explicit_ridge(train, evaluation, y, regularization, match_spectral_threshold=False):
    train, evaluation = np.asarray(train), np.asarray(evaluation)
    if match_spectral_threshold:
        _, s, vt = np.linalg.svd(train, full_matrices=False)
        keep = s*s > 1e-8 * max(1., float(np.max(s*s)))
        if not keep.any():
            raise ValueError('No positive feature spectrum')
        train, evaluation = train @ vt[keep].T, evaluation @ vt[keep].T
    mean = train.mean(axis=0)
    train, evaluation = train - mean, evaluation - mean
    scale = float(np.mean(np.sum(train * train, axis=1)))
    if scale <= 1e-12:
        raise ValueError('Degenerate feature scale')
    train, evaluation = train / np.sqrt(scale), evaluation / np.sqrt(scale)
    ym = float(np.mean(y))
    beta = solve(train.T @ train + len(y)*regularization*np.eye(train.shape[1]), train.T@(y-ym), assume_a='pos')
    return evaluation @ beta + ym


def rbf(x, y, gamma):
    d = np.maximum(np.sum(x*x, axis=1)[:,None] + np.sum(y*y, axis=1)[None,:] - 2*x@y.T, 0.)
    return np.exp(-gamma*d)
