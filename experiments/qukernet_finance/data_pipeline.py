"""Frozen, calendar-aligned D0 windows for a financial QuKerNet adaptation.

Inputs are local CSV snapshots only. No provider API or network share is used.
Raw-price labels are diagnostic targets, not verified investment total returns.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

FEATURES = ([f'momentum{h}' for h in (1, 5, 10, 20, 60)] +
            [f'volatility{h}' for h in (5, 10, 20, 60)] +
            [f'relative_volume{h}' for h in (5, 10, 20)])
OLD4 = ['momentum5', 'momentum20', 'volatility20', 'relative_volume20']
QUARTERS = ('2022Q1', '2023Q1', '2024Q1')
DATA_SEED = 20261109


def _network(path):
    if str(path).startswith(('\\\\', '//')):
        return True
    p = Path(path)
    if os.name == 'nt' and p.drive:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(p.anchor) == 4
    return False


def local_path(value):
    if _network(value):
        raise ValueError('Only existing local snapshots are admitted; network inputs are forbidden')
    p = Path(value).resolve()
    if _network(p):
        raise ValueError('Resolved path is not local')
    return p


def file_identity(path):
    path = local_path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Local snapshot changed during identity collection')
    return {'size': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': digest.hexdigest()}


def _load_inputs(panel, calendar):
    identities, paths = {}, {}
    if isinstance(panel, pd.DataFrame):
        raw = panel.copy(deep=True)
    else:
        paths['panel'] = local_path(panel)
        identities['panel'] = file_identity(paths['panel'])
        raw = pd.read_csv(paths['panel'], dtype={'symbol': str})
    if isinstance(calendar, pd.DataFrame):
        dates = calendar['date']
    elif isinstance(calendar, (str, os.PathLike)):
        paths['calendar'] = local_path(calendar)
        identities['calendar'] = file_identity(paths['calendar'])
        dates = pd.read_csv(paths['calendar'])['date']
    else:
        dates = calendar
    required = {'symbol', 'date', 'open', 'close', 'volume'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing required panel columns: {sorted(required-set(raw.columns))}')
    raw['date'] = pd.to_datetime(raw['date'], errors='raise')
    cal = pd.DatetimeIndex(pd.to_datetime(dates, errors='raise'))
    if raw['date'].isna().any() or cal.hasnans:
        raise ValueError('Missing dates')
    if raw['date'].dt.tz is not None or cal.tz is not None:
        raise ValueError('Use naive daily market dates')
    if not raw['date'].eq(raw['date'].dt.normalize()).all() or not cal.equals(cal.normalize()):
        raise ValueError('Daily input dates must be normalized to midnight')
    if ((raw['date'].dt.year >= 2025) | (raw['date'].dt.year < 2019)).any() or ((cal.year >= 2025) | (cal.year < 2019)).any():
        raise ValueError('Only the frozen 2019-2024 diagnostic period is admitted')
    if cal.has_duplicates:
        raise ValueError('Duplicate calendar dates')
    cal = cal.sort_values()
    if raw.duplicated(['symbol', 'date']).any():
        raise ValueError('Duplicate symbol-date keys')
    if raw['symbol'].isna().any():
        raise ValueError('Missing security identifier')
    raw['symbol'] = raw['symbol'].astype(str)
    if raw['symbol'].nunique() != 20:
        raise ValueError('This fixed protocol requires exactly 20 securities')
    if not raw['date'].isin(cal).all():
        raise ValueError('Panel contains dates absent from supplied market calendar')
    for key, path in paths.items():
        if file_identity(path) != identities[key]:
            raise RuntimeError(f'{key} snapshot changed while reading')
    return raw, cal, identities


def build_features(raw, calendar):
    """Use calendar reindexing; no forward fill, backward fill, or date compaction."""
    tables = []
    for symbol, group in raw.groupby('symbol', sort=True):
        series = group.set_index('date').reindex(calendar)
        close, opening, volume = [pd.to_numeric(series[col], errors='coerce').where(lambda x: x > 0)
                                  for col in ('close', 'open', 'volume')]
        ret = close / close.shift(1) - 1.0
        table = pd.DataFrame({'symbol': symbol, 'date': calendar, 'j': np.arange(len(calendar))})
        for h in (1, 5, 10, 20, 60):
            table[f'momentum{h}'] = (close / close.shift(h) - 1.0).to_numpy()
        for h in (5, 10, 20, 60):
            table[f'volatility{h}'] = ret.rolling(h, min_periods=h).std(ddof=1).to_numpy()
        for h in (5, 10, 20):
            prior_mean = volume.shift(1).rolling(h, min_periods=h).mean()
            table[f'relative_volume{h}'] = np.log(volume / prior_mean).to_numpy()
        table['y_raw'] = (opening.shift(-6) / opening.shift(-1) - 1.0).to_numpy()
        table['label_start_j'], table['label_end_j'] = table['j'] + 1, table['j'] + 6
        table['feature_valid'] = np.isfinite(table[FEATURES].to_numpy()).all(axis=1)
        table['target_valid'] = np.isfinite(table['y_raw'])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _discretize(values, bins=10):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not len(values):
        raise ValueError('Finite nonempty vector required for quantile bins')
    edges = np.unique(np.quantile(values, np.arange(1, bins)/bins))
    # Constant variables receive one bin, including repeated quantile edges.
    edges = edges[(edges > values.min()) & (edges < values.max())]
    return np.searchsorted(edges, values, side='right'), edges.tolist()


def _mi(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    counts = np.zeros((int(a.max())+1, int(b.max())+1), dtype=float)
    np.add.at(counts, (a, b), 1.)
    joint = counts / counts.sum()
    product = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask]/product[mask])))


def mrmr_select(x, y, names=FEATURES, count=8, task='regression'):
    """Empirical discrete MI relevance minus mean selected-feature redundancy.

    Quantile edges, target discretization and MI are fitted on the 64 controlled
    training rows only. Numerical ties (1e-12) follow the declared feature order.
    """
    x, y = np.asarray(x, float), np.asarray(y)
    if x.shape != (len(y), len(names)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid mRMR training data')
    if not 1 <= count <= len(names):
        raise ValueError('Invalid mRMR feature count')
    discrete, edges = [], {}
    for j, name in enumerate(names):
        values, bounds = _discretize(x[:, j])
        discrete.append(values)
        edges[name] = bounds
    if task == 'classification':
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError('Classification labels must be 0 or 1')
        target, target_edges = y.astype(int), None
    elif task == 'regression':
        target, target_edges = _discretize(y)
    else:
        raise ValueError('Unknown task')
    relevance = np.array([_mi(v, target) for v in discrete])
    redundancy = np.array([[_mi(a, b) for b in discrete] for a in discrete])
    selected, steps = [], []
    for _ in range(count):
        scores = {j: float(relevance[j] - (np.mean(redundancy[j, selected]) if selected else 0.))
                  for j in range(len(names)) if j not in selected}
        best = max(scores.values())
        choice = next(j for j in range(len(names)) if j in scores and scores[j] >= best-1e-12)
        steps.append({'selected': names[choice], 'score': scores[choice],
                      'remaining_scores': {names[j]: scores[j] for j in scores}})
        selected.append(choice)
    return selected, {'method': '10-quantile-bin MI relevance minus mean redundancy',
                      'fit_rows': len(y), 'target_kind': task, 'target_bin_edges': target_edges,
                      'feature_bin_edges': edges, 'relevance_nats': dict(zip(names, relevance.tolist())),
                      'redundancy_nats': redundancy.tolist(), 'steps': steps,
                      'tie_rule': 'declared feature order within 1e-12'}


def fit_scaler(x):
    x = np.asarray(x, float)
    median = np.median(x, axis=0)
    iqr = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    return {'median': median.tolist(), 'iqr': iqr.tolist(),
            'constant_columns': np.flatnonzero(iqr <= 1e-12).tolist(),
            'fit_rows': len(x), 'formula': 'pi * clip((x-median)/IQR, -3, 3)/3; constant columns -> 0'}


def apply_scaler(x, scaler):
    x = np.asarray(x, float)
    median, iqr = np.array(scaler['median']), np.array(scaler['iqr'])
    varying = iqr > 1e-12
    transformed = np.clip((x-median)/np.where(varying, iqr, 1.), -3., 3.)/3. * np.pi
    transformed[:, ~varying] = 0.
    return transformed


def _metadata(rows, calendar):
    return [{'symbol': str(row.symbol), 'date': pd.Timestamp(row.date).strftime('%Y-%m-%d'),
             'label_start': calendar[int(row.label_start_j)].strftime('%Y-%m-%d'),
             'label_end': calendar[int(row.label_end_j)].strftime('%Y-%m-%d')}
            for row in rows.itertuples()]


def _select_train(eligible, t_days, seed):
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(sorted(eligible.symbol.unique()), 0)
    indices, log = [], []
    for number, box in enumerate(np.array_split(t_days, 16)):
        subset = eligible[eligible.j.isin(box)]
        days = sorted(j for j, rows in subset.groupby('j') if len(rows) >= 4)
        if not days:
            raise ValueError(f'Training date bin {number} has no date with four valid securities')
        chosen_day = days[len(days)//2]
        day = subset[subset.j == chosen_day].sort_values('symbol').copy()
        day['_count'] = day.symbol.map(counts)
        day['_tie'] = rng.random(len(day))
        chosen = day.sort_values(['_count', '_tie', 'symbol']).head(4)
        indices.extend(chosen.index.tolist())
        for symbol in chosen.symbol:
            counts[symbol] += 1
        log.append({'bin': number, 'calendar_first_j': int(box[0]), 'calendar_last_j': int(box[-1]),
                    'eligible_dates': len(days), 'chosen_j': int(chosen_day), 'symbols': chosen.symbol.tolist()})
    train = eligible.loc[indices].sort_values(['j', 'symbol']).copy()
    if len(train) != 64:
        raise ValueError('Exactly 64 controlled training rows required')
    return train, {'seed': int(seed), 'date_bins': log, 'security_counts': counts,
                   'date_choice': 'upper middle of dates with at least four eligible securities in each bin',
                   'security_choice': 'lowest cumulative exposure, seeded random tie, stable symbol'}


def prepare_window(panel, calendar, quarter, task='regression', seed=DATA_SEED):
    started = time.perf_counter()
    if str(quarter) not in QUARTERS:
        raise ValueError(f'Quarter must be one of {QUARTERS}')
    if task not in ('regression', 'classification'):
        raise ValueError('Task must be regression or classification')
    raw, cal, identities = _load_inputs(panel, calendar)
    period = pd.Period(quarter, freq='Q')
    outer_days = np.flatnonzero(cal.to_period('Q') == period)
    if not len(outer_days) or outer_days[0] < 252:
        raise ValueError('Insufficient calendar history')
    cutoff = int(outer_days[0])-1
    history = np.arange(cutoff-251, cutoff+1)
    t_days, v_days = history[:168], history[168:]
    frame = build_features(raw, cal)
    good = frame.feature_valid & frame.target_valid
    full = frame[frame.j.isin(t_days) & (frame.label_end_j < v_days[0]) & good].sort_values(['j', 'symbol']).copy()
    train, sampling = _select_train(full, t_days, seed)
    valid_pool = frame[frame.j.isin(v_days) & (frame.label_end_j <= cutoff) & good]
    complete = [int(j) for j, rows in valid_pool.groupby('j', sort=True) if rows.symbol.nunique() == 20 and len(rows) == 20]
    if len(complete) < 2:
        raise ValueError('Two complete 20-security validation cross-sections unavailable')
    valid_days = complete[:2]
    valid = valid_pool[valid_pool.j.isin(valid_days)].sort_values(['j', 'symbol']).copy()
    prediction_days = [int(j) for j in outer_days[::5] if j+6 < len(cal) and cal[j+6].to_period('Q') == period]
    test_pool = frame[frame.j.isin(prediction_days)].copy()
    test = test_pool[test_pool.feature_valid & test_pool.target_valid].sort_values(['j', 'symbol']).copy()
    if test.empty:
        raise ValueError('No eligible outer diagnostic records')
    # Mature labels, not random row splitting, establish all boundaries.
    assert train.label_end_j.max() < v_days[0] <= valid.j.min()
    assert full.label_end_j.max() < v_days[0]
    assert valid.label_end_j.max() <= cutoff < test.j.min()
    assert all(cal[int(j)].to_period('Q') == period for j in test.label_end_j)
    assert len(valid) == 40 and len(train) == 64
    y_train_raw = train.y_raw.to_numpy(dtype=float)
    target = y_train_raw if task == 'regression' else (y_train_raw > 0.).astype(int)
    selected, selection = mrmr_select(train[FEATURES].to_numpy(), target, task=task)
    selected_names = [FEATURES[j] for j in selected]
    scaler = fit_scaler(train[selected_names].to_numpy())
    scaler4 = fit_scaler(train[OLD4].to_numpy())
    result, extras = {}, {}
    for split, rows in [('train', train), ('valid', valid), ('test', test), ('full', full)]:
        y = rows.y_raw.to_numpy(dtype=float)
        result['x_'+split] = apply_scaler(rows[selected_names].to_numpy(), scaler)
        result['y_'+split] = y if task == 'regression' else (y > 0.).astype(np.int64)
        result['meta_'+split] = _metadata(rows, cal)
        extras['x4_'+split] = apply_scaler(rows[OLD4].to_numpy(), scaler4)
        extras['y_raw_'+split] = y
        extras['momentum20_'+split] = rows.momentum20.to_numpy(dtype=float)
    result['selected_features'] = selected_names
    result['preprocessing'] = {'feature_order': FEATURES, 'selected_indices': selected,
                               'mrmr': selection, 'selected_scaler': scaler, 'old4_features': OLD4,
                               'old4_scaler': scaler4, 'angle_unit': 'radians; x = pi*z',
                               'baseline_scale': 'use x/pi for ridge and RBF',
                               'fitted_on': 'controlled 64 T records only; full T reuses this fit',
                               'training_target_mean': float(y_train_raw.mean()),
                               'training_target_sd': float(y_train_raw.std(ddof=1))}
    result['extras'] = extras
    result['audit'] = {'scope': 'D0 raw-price diagnostic; G4 unverified; no total-return or profitability interpretation',
                       'quarter': str(quarter), 'task': task, 'data_seed': int(seed),
                       'classification_rule': 'raw five-open-interval return > 0 -> 1, otherwise 0',
                       'input_identities': identities, 'securities': sorted(raw.symbol.unique()),
                       'quarter_first_market_date': cal[outer_days[0]].strftime('%Y-%m-%d'),
                       'fit_cutoff': cal[cutoff].strftime('%Y-%m-%d'),
                       'T': {'first': cal[t_days[0]].strftime('%Y-%m-%d'), 'last': cal[t_days[-1]].strftime('%Y-%m-%d'), 'calendar_days': 168},
                       'V': {'first': cal[v_days[0]].strftime('%Y-%m-%d'), 'last': cal[v_days[-1]].strftime('%Y-%m-%d'), 'calendar_days': 84},
                       'validation_dates': [cal[j].strftime('%Y-%m-%d') for j in valid_days],
                       'outer_decision_dates': [cal[j].strftime('%Y-%m-%d') for j in prediction_days],
                       'rows': {key: len(result['y_'+key]) for key in ('train', 'valid', 'test', 'full')},
                       'outer_grid_security_records': len(test_pool),
                       'outer_feature_available': int(test_pool.feature_valid.sum()),
                       'outer_features_available_but_missing_target': int((test_pool.feature_valid & ~test_pool.target_valid).sum()),
                       'outer_scoring_selection': 'Only finite-feature/finite-target records are scored; missing targets counted separately, not a trading-universe rule',
                       'sampling': sampling, 'missing_policy': 'market-calendar reindex; no filling or date compaction',
                       'validation_selection_limit': 'earliest two complete future-labelled cross-sections; conditional diagnostic sample, not unbiased historical selection',
                       'financial_time_status': '2022-2024 development replay, not a newly unseen sealed test',
                       'boundaries_verified': True, 'seconds': time.perf_counter()-started}
    return result


def protected_source_root(script_path):
    script = Path(script_path).resolve()
    for parent in script.parents:
        if (parent/'课程大纲.md').is_file():
            return parent
    return next((p for p in script.parents if p.name.casefold() == 'code'), script.parent)


def private_output(root, input_paths=(), script_path=__file__):
    destination = local_path(root)
    protected = protected_source_root(script_path)
    if destination == protected or destination.is_relative_to(protected):
        raise ValueError('Financial outputs must remain outside the public source tree')
    for value in input_paths:
        source = local_path(value).parent
        if destination == source or destination.is_relative_to(source):
            raise ValueError('Output overlaps a read-only input directory')
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_window(window, output, stem, input_paths=()):
    """Save a JSON manifest and non-object NPZ arrays without overwriting a freeze."""
    output = private_output(output, input_paths)
    if Path(stem).name != stem or not stem:
        raise ValueError('Snapshot stem must be a single filename component')
    arrays = {k: v for k, v in window.items() if isinstance(v, np.ndarray)}
    arrays.update({'extras__'+k: v for k, v in window['extras'].items()})
    npz, record = output/(stem+'.npz'), output/(stem+'.json')
    if npz.exists() or record.exists():
        raise FileExistsError('Frozen window already exists')
    with npz.open('xb') as f:
        np.savez_compressed(f, **arrays)
    metadata = {k: v for k, v in window.items() if not isinstance(v, np.ndarray) and k != 'extras'}
    metadata['created_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['code_identities'] = {name: file_identity(Path(__file__).parent/name)
                                   for name in ('data_pipeline.py', 'modeling.py', 'verify_data.py')
                                   if (Path(__file__).parent/name).is_file()}
    metadata['npz'] = {'file': npz.name, 'sha256': file_identity(npz)['sha256'],
                       'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
    with record.open('x', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return record, npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--calendar', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quarter', action='append', choices=QUARTERS)
    parser.add_argument('--task', choices=['regression', 'classification'], default='regression')
    parser.add_argument('--seed', type=int, default=DATA_SEED)
    args = parser.parse_args()
    output = private_output(args.output, [args.input, args.calendar])
    for quarter in args.quarter or QUARTERS:
        window = prepare_window(args.input, args.calendar, quarter, args.task, args.seed)
        manifest, arrays = save_window(window, output, quarter+'_'+args.task, [args.input, args.calendar])
        print(json.dumps({'quarter': quarter, 'rows': window['audit']['rows'],
                          'selected_features': window['selected_features'],
                          'manifest': manifest.name, 'arrays': arrays.name}, ensure_ascii=False))


if __name__ == '__main__':
    main()
