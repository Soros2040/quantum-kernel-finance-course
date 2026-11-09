"""Auditable financial kernel models and date-equal diagnostic metrics."""
from __future__ import annotations
import numpy as np
try:
    from scipy.linalg import eigh, solve
    LINEAR_ALGEBRA_BACKEND = 'scipy.linalg'
except ImportError:
    LINEAR_ALGEBRA_BACKEND = 'numpy.linalg'

    def eigh(a, check_finite=True):
        if check_finite and not np.isfinite(a).all():
            raise ValueError('Nonfinite matrix')
        return np.linalg.eigh(a)

    def solve(a, b, assume_a=None):
        return np.linalg.solve(a, b)


def rankdata(values, method='average'):
    if method != 'average':
        raise ValueError('Only average tie ranks are supported')
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        stop = start+1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start+1+stop)/2.
        start = stop
    return ranks

LAMBDAS = (.001, .01, .1)
RBF_GAMMAS = (.1, 1., 10.)


def _fit_map(train):
    train = np.asarray(train, dtype=float)
    if train.ndim != 2 or train.shape[0] != train.shape[1] or not np.isfinite(train).all():
        raise ValueError('Finite square training kernel required')
    d, q = eigh((train+train.T)*.5, check_finite=True)
    threshold = 1e-8 * max(1., float(np.max(np.abs(d))))
    keep = d > threshold
    if not keep.any():
        raise ValueError('No retained positive eigenspace')
    root = np.sqrt(d[keep])
    features = q[:, keep] * root
    mean = features.mean(axis=0)
    centered = features-mean
    scale = float(np.mean(np.sum(centered*centered, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Degenerate centered kernel')
    diagnostics = {'min_eigenvalue': float(d.min()), 'negative_eigenvalues': int((d < -threshold).sum()),
                   'retained_rank': int(keep.sum()), 'threshold': threshold, 'centered_scale': scale,
                   'linear_algebra_backend': LINEAR_ALGEBRA_BACKEND}
    state = {'eigenvectors': q[:, keep].tolist(), 'sqrt_eigenvalues': root.tolist(),
             'feature_mean': mean.tolist(), 'centered_scale': scale,
             'n_train': len(train), 'diagnostics': diagnostics}
    return centered/np.sqrt(scale), state


def transform_cross(state, cross):
    cross = np.asarray(cross, dtype=float)
    if cross.ndim != 2 or cross.shape[1] != state['n_train'] or not np.isfinite(cross).all():
        raise ValueError('Cross-kernel columns must match finite training order')
    z = cross @ np.asarray(state['eigenvectors']) / np.asarray(state['sqrt_eigenvalues'])
    return (z-np.asarray(state['feature_mean'])) / np.sqrt(state['centered_scale'])


def kernel_map(train, cross):
    """Course spectral repair with compatible out-of-sample projection."""
    zt, state = _fit_map(train)
    return zt, transform_cross(state, cross), state['diagnostics']


def predict_mapped(zt, zv, y, regularization):
    zt, zv, y = np.asarray(zt, float), np.asarray(zv, float), np.asarray(y, float)
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError('Regularization must be finite and positive')
    if y.shape != (len(zt),) or not np.isfinite(y).all():
        raise ValueError('Training targets do not match training kernel')
    ym = float(y.mean())
    alpha = solve(zt @ zt.T + len(y)*regularization*np.eye(len(y)), y-ym, assume_a='pos')
    return zv @ (zt.T @ alpha) + ym


def fit_krr(train, y, regularization):
    """Return a JSON-safe bundle; objective is mean squared loss + lambda norm²."""
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError('Regularization must be finite and positive')
    zt, mapping = _fit_map(train)
    y = np.asarray(y, float)
    if y.shape != (len(zt),) or not np.isfinite(y).all():
        raise ValueError('Invalid training targets')
    mean = float(y.mean())
    alpha = solve(zt @ zt.T + len(y)*regularization*np.eye(len(y)), y-mean, assume_a='pos')
    return {'kind': 'course_krr', 'regularization': float(regularization), 'n_train': len(y),
            'objective': 'mean squared loss + lambda RKHS norm squared; system K+n*lambda*I',
            'target_mean': mean, 'alpha': alpha.tolist(), 'z_train': zt.tolist(), 'mapping': mapping,
            'target_scale_convention': 'raw target centering; equivalent to training target z-score followed by inverse transform for this linear quadratic fit'}


def predict_krr(bundle, cross):
    if bundle.get('kind') != 'course_krr':
        raise ValueError('Unsupported model bundle')
    zv = transform_cross(bundle['mapping'], cross)
    return zv @ (np.asarray(bundle['z_train']).T @ np.asarray(bundle['alpha'])) + bundle['target_mean']


def krr(train, cross, y, regularization):
    bundle = fit_krr(train, y, regularization)
    return predict_krr(bundle, cross), bundle['mapping']['diagnostics']


def svc(train, cross, y, C=1.):
    """Classification control uses the uncentered input kernel and fixed C=1.

    It deliberately records a distinct pipeline from the regression PSD map.
    The exact-kernel caller is responsible for constructing a valid Gram matrix.
    """
    from sklearn.svm import SVC
    train, cross, y = np.asarray(train, float), np.asarray(cross, float), np.asarray(y)
    if train.shape != (len(y), len(y)) or cross.ndim != 2 or cross.shape[1] != len(y):
        raise ValueError('Kernel/target shape mismatch')
    if not np.isfinite(train).all() or not np.isfinite(cross).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Finite kernels and both binary classes are required')
    if C != 1.:
        raise ValueError('Approved classification control fixes C=1')
    fitted = SVC(C=1., kernel='precomputed', probability=False).fit((train+train.T)*.5, y)
    prediction = fitted.predict(cross)
    return prediction, {'kernel_processing': 'raw uncentered precomputed kernel; symmetrized train',
                        'C': 1., 'classes': fitted.classes_.tolist(),
                        'decision_scores': np.asarray(fitted.decision_function(cross)).tolist(),
                        'support_indices': fitted.support_.tolist()}


def fit_predict(train, cross, y, regularization=.1, task='regression', C=1.):
    if task == 'regression':
        return krr(train, cross, y, regularization)
    if task == 'classification':
        return svc(train, cross, y, C=C)
    raise ValueError('Unknown task')


def _date_values(meta_or_dates):
    values = list(meta_or_dates)
    return np.asarray([str(v['date']) if isinstance(v, dict) else str(v) for v in values])


def _metric_arrays(y, prediction, meta_or_dates):
    y, prediction, dates = np.asarray(y, float), np.asarray(prediction, float), _date_values(meta_or_dates)
    if y.ndim != 1 or prediction.shape != y.shape or dates.shape != y.shape or not len(y):
        raise ValueError('Metric inputs must be nonempty matching one-dimensional records')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Score only explicitly admitted finite targets and predictions')
    return y, prediction, dates


def date_equal_mse(y, prediction, meta_or_dates):
    y, prediction, dates = _metric_arrays(y, prediction, meta_or_dates)
    errors = (prediction-y)**2
    return float(np.mean([errors[dates == date].mean() for date in np.unique(dates)]))


def date_equal_accuracy(y, prediction, meta_or_dates):
    y, prediction, dates = _metric_arrays(y, prediction, meta_or_dates)
    return float(np.mean([(prediction[dates == d] == y[dates == d]).mean() for d in np.unique(dates)]))


def rank_ic(y, prediction, meta_or_dates):
    y, prediction, dates = _metric_arrays(y, prediction, meta_or_dates)
    per_date, valid, skipped = [], [], []
    for date in np.unique(dates):
        mask = dates == date
        a, b = rankdata(y[mask], method='average'), rankdata(prediction[mask], method='average')
        if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
            score = None
            skipped.append(str(date))
        else:
            score = float(np.corrcoef(a, b)[0, 1])
            valid.append(score)
        per_date.append({'date': str(date), 'n': int(mask.sum()), 'rank_ic': score})
    return {'mean': float(np.mean(valid)) if valid else None, 'n_dates': len(valid),
            'skipped_dates': skipped, 'total_dates': len(per_date), 'per_date': per_date}


def select_lambda(train, cross, y_train, y_valid, meta_valid, lambdas=LAMBDAS):
    grid, bundles = [], {}
    for regularization in lambdas:
        try:
            bundle = fit_krr(train, y_train, regularization)
            loss = date_equal_mse(y_valid, predict_krr(bundle, cross), meta_valid)
            grid.append({'lambda': float(regularization), 'validation_mse': loss, 'status': 'ok'})
            bundles[float(regularization)] = bundle
        except (ValueError, np.linalg.LinAlgError) as exc:
            grid.append({'lambda': float(regularization), 'validation_mse': None,
                         'status': 'failed', 'error': type(exc).__name__+': '+str(exc)})
    passed = [row for row in grid if row['status'] == 'ok']
    if not passed:
        raise ValueError('No regularization candidate produced a valid fit')
    # Do not use a fixed absolute 1e-12 tie tolerance on tiny raw-return MSE.
    best_loss = min(row['validation_mse'] for row in passed)
    best = max(row['lambda'] for row in passed if row['validation_mse'] == best_loss)
    return best, bundles[best], grid


def rbf(x, y, gamma):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError('Gamma must be finite and positive')
    distance = np.maximum((x*x).sum(axis=1)[:, None] + (y*y).sum(axis=1)[None, :] - 2*x@y.T, 0.)
    return np.exp(-gamma*distance)


def explicit_ridge(train, evaluation, y, regularization, match_spectral_threshold=False):
    train, evaluation, y = np.asarray(train, float), np.asarray(evaluation, float), np.asarray(y, float)
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError('Regularization must be finite and positive')
    if match_spectral_threshold:
        _, singular, vt = np.linalg.svd(train, full_matrices=False)
        keep = singular**2 > 1e-8*max(1., float(np.max(singular**2)))
        if not keep.any():
            raise ValueError('No retained linear feature spectrum')
        train, evaluation = train@vt[keep].T, evaluation@vt[keep].T
    mean = train.mean(axis=0)
    train, evaluation = train-mean, evaluation-mean
    scale = float(np.mean(np.sum(train*train, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Degenerate feature scale')
    train, evaluation = train/np.sqrt(scale), evaluation/np.sqrt(scale)
    ym = float(y.mean())
    beta = solve(train.T@train + len(y)*regularization*np.eye(train.shape[1]), train.T@(y-ym), assume_a='pos')
    return evaluation@beta + ym
