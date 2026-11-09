"""Bounded synthetic checks, optionally followed by read-only frozen-data checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

try:
    from .data_pipeline import (FEATURES, QUARTERS, build_features, prepare_window,
                                mrmr_select, fit_scaler, apply_scaler, private_output, save_window)
    from .modeling import (fit_krr, predict_krr, kernel_map, explicit_ridge,
                           date_equal_mse, rank_ic, select_lambda, LAMBDAS)
except ImportError:
    from data_pipeline import (FEATURES, QUARTERS, build_features, prepare_window,
                               mrmr_select, fit_scaler, apply_scaler, private_output, save_window)
    from modeling import (fit_krr, predict_krr, kernel_map, explicit_ridge,
                          date_equal_mse, rank_ic, select_lambda, LAMBDAS)


def synthetic_panel():
    dates = pd.bdate_range('2019-01-01', '2024-12-31')
    rng = np.random.default_rng(431)
    rows = []
    for number in range(20):
        close = (10.+number)*np.exp(np.cumsum(rng.normal(.0001, .008, len(dates))))
        opening = close*np.exp(rng.normal(0., .002, len(dates)))
        volume = np.exp(rng.normal(12.+number*.01, .2, len(dates)))
        rows.append(pd.DataFrame({'symbol': f'TEST{number:02d}', 'date': dates,
                                  'open': opening, 'close': close, 'volume': volume}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame({'date': dates})


def run_checks(input_path=None, calendar_path=None):
    started = time.perf_counter()
    checks = []
    raw, calendar = synthetic_panel()
    cal = pd.DatetimeIndex(calendar.date)
    a = prepare_window(raw, calendar, '2022Q1')
    assert a['x_train'].shape == (64, 8) and a['x_valid'].shape == (40, 8)
    assert len({r['date'] for r in a['meta_train']}) == 16
    assert len({r['date'] for r in a['meta_valid']}) == 2
    assert max(r['label_end'] for r in a['meta_train']) < min(r['date'] for r in a['meta_valid'])
    assert max(r['label_end'] for r in a['meta_valid']) <= a['audit']['fit_cutoff'] < min(r['date'] for r in a['meta_test'])
    assert all(pd.Period(r['label_end'], freq='Q') == pd.Period('2022Q1') for r in a['meta_test'])
    checks.append('64/40 grouped sampling, 168/84 windows and label-maturity boundaries')

    # Alter every numeric value after the training block; training selections,
    # mRMR edges and scaling must remain exactly unchanged.
    altered = raw.copy(deep=True)
    future = altered.date >= pd.Timestamp(a['audit']['V']['first'])
    altered.loc[future, 'close'] *= 7.
    altered.loc[future, 'open'] *= 3.
    altered.loc[future, 'volume'] *= 11.
    b = prepare_window(altered, calendar, '2022Q1')
    assert a['meta_train'] == b['meta_train']
    assert a['selected_features'] == b['selected_features']
    assert a['preprocessing'] == b['preprocessing']
    assert np.array_equal(a['x_train'], b['x_train']) and np.array_equal(a['y_train'], b['y_train'])
    assert not np.array_equal(a['x_valid'], b['x_valid'])
    checks.append('future-value mutation cannot change controlled training rows, mRMR or scaler')

    table = build_features(raw, cal)
    one = raw[raw.symbol == 'TEST00'].set_index('date').reindex(cal)
    row = table[(table.symbol == 'TEST00') & (table.j == 400)].iloc[0]
    returns = one.close/one.close.shift(1)-1
    assert np.isclose(row.momentum60, one.close.iloc[400]/one.close.iloc[340]-1)
    assert np.isclose(row.volatility20, returns.iloc[381:401].std(ddof=1))
    assert np.isclose(row.relative_volume20, np.log(one.volume.iloc[400]/one.volume.iloc[380:400].mean()))
    assert np.isclose(row.y_raw, one.open.iloc[406]/one.open.iloc[401]-1)
    checks.append('12-feature formulas, volume excludes current day, ddof=1 and t+1/t+6 label')

    gap_date = cal[300]
    gapped = raw[~((raw.symbol == 'TEST00') & (raw.date == gap_date))].copy()
    gap = build_features(gapped, cal)
    gap = gap[gap.symbol == 'TEST00'].set_index('j')
    assert pd.isna(gap.loc[300, 'momentum1']) and pd.isna(gap.loc[301, 'momentum1'])
    assert pd.isna(gap.loc[360, 'volatility60']) and pd.notna(gap.loc[361, 'volatility60'])
    assert pd.isna(gap.loc[294, 'y_raw']) and pd.isna(gap.loc[299, 'y_raw'])
    assert len(gap) == len(cal)
    checks.append('calendar gap retained; adjacent returns and missing label endpoints remain missing')

    selected, info = mrmr_select(np.ones((64, 12)), np.linspace(-1, 1, 64))
    assert selected == list(range(8)) and all(v == 0 for v in info['relevance_nats'].values())
    train = np.array([[0., 5.], [1., 5.], [2., 5.], [3., 5.]])
    scale = fit_scaler(train)
    transformed = apply_scaler(np.array([[1e8, -1e8], [-1e8, 1e8]]), scale)
    assert np.allclose(transformed[:, 0], [np.pi, -np.pi]) and np.all(transformed[:, 1] == 0.)
    checks.append('mRMR deterministic feature-order ties; training quantiles and constant-column scaling')

    class_window = prepare_window(raw, calendar, '2022Q1', task='classification')
    assert class_window['meta_train'] == a['meta_train']
    assert np.array_equal(class_window['y_train'], (a['y_train'] > 0).astype(int))
    poisoned_calendar = pd.concat([calendar, pd.DataFrame({'date': [pd.Timestamp('2025-01-02')]})], ignore_index=True)
    try:
        prepare_window(raw, poisoned_calendar, '2022Q1')
    except ValueError:
        pass
    else:
        raise AssertionError('Sealed 2025 calendar was admitted')
    checks.append('classification threshold and unchanged sample IDs; reject any 2025 input')

    rng = np.random.default_rng(710)
    x, v, y = rng.normal(size=(24, 5)), rng.normal(size=(9, 5)), rng.normal(size=24)
    gram, cross = x@x.T, v@x.T
    bundle = fit_krr(gram, y, .01)
    pred = predict_krr(bundle, cross)
    recovered = json.loads(json.dumps(bundle, allow_nan=False))
    assert np.allclose(pred, predict_krr(recovered, cross), atol=1e-12, rtol=1e-12)
    assert np.allclose(pred, explicit_ridge(x, v, y, .01, True), atol=1e-10, rtol=1e-10)
    zt, zv, _ = kernel_map(gram-.1*np.eye(24), cross)
    joint = np.vstack([zt, zv])
    assert np.linalg.eigvalsh(joint@joint.T).min() > -1e-9
    checks.append('n*lambda KRR equals primal ridge; JSON round trip; compatible repaired joint PSD block')

    dates = ['a', 'b', 'b', 'b']
    assert np.isclose(date_equal_mse([0, 0, 0, 0], [2, 1, 1, 1], dates), 2.5)
    ic = rank_ic([0, 1, 2, 3], [0, 3, 2, 1], dates)
    assert ic['n_dates'] == 1 and ic['skipped_dates'] == ['a'] and np.isclose(ic['mean'], -1.)
    chosen, _, grid = select_lambda(gram, cross, y, np.arange(9)/10, ['a']*4+['b']*5)
    assert chosen in LAMBDAS and len(grid) == 3
    checks.append('date-equal metric is not row-weighted; constant/singleton RankIC skipped; full lambda grid retained')

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        public = root/'public'
        code = public/'experiments'/'qukernet_finance'
        code.mkdir(parents=True)
        (public/'课程大纲.md').write_text('test marker', encoding='utf-8')
        (public/'README.md').write_text('test marker', encoding='utf-8')
        script = code/'data_pipeline.py'
        script.write_text('synthetic path fixture', encoding='utf-8')
        input_dir = root/'input'
        input_dir.mkdir()
        dummy_input = input_dir/'panel.csv'
        dummy_input.write_text('immutable', encoding='utf-8')
        for forbidden in (public/'results', input_dir/'results'):
            try:
                private_output(forbidden, [dummy_input], script)
            except ValueError:
                pass
            else:
                raise AssertionError('Protected output path accepted')
        remote = root/'remote'/'code'/'qukernet_finance'
        remote.mkdir(parents=True)
        remote_script = remote/'data_pipeline.py'
        remote_script.write_text('synthetic remote fixture', encoding='utf-8')
        destination = private_output(root/'remote'/'runs', [dummy_input], remote_script)
        record, arrays = save_window(a, destination, 'synthetic_window')
        with np.load(arrays, allow_pickle=False) as frozen:
            assert np.array_equal(frozen['x_train'], a['x_train'])
            assert np.array_equal(frozen['extras__x4_full'], a['extras']['x4_full'])
        manifest = json.loads(record.read_text(encoding='utf-8'))
        assert manifest['audit']['boundaries_verified']
        assert dummy_input.read_text(encoding='utf-8') == 'immutable'
    checks.append('public/input write guards, remote runs allowed, non-object snapshot round trip')

    actual = []
    if bool(input_path) != bool(calendar_path):
        raise ValueError('Real input and calendar must be supplied together')
    if input_path:
        for quarter in QUARTERS:
            window = prepare_window(input_path, calendar_path, quarter)
            assert np.isfinite(window['x_train']).all() and np.isfinite(window['y_train']).all()
            assert np.max(np.abs(window['x_train'])) <= np.pi+1e-15
            actual.append({'quarter': quarter, 'rows': window['audit']['rows'],
                           'selected_features': window['selected_features'],
                           'T': window['audit']['T'], 'V': window['audit']['V'],
                           'validation_dates': window['audit']['validation_dates'],
                           'fit_cutoff': window['audit']['fit_cutoff'],
                           'input_identities': window['audit']['input_identities']})
    return {'passed': True, 'synthetic_checks': checks, 'synthetic_check_count': len(checks),
            'real_windows_read_only': actual, 'experiment_training_run': False,
            'seconds': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--calendar')
    parser.add_argument('--output', help='Optional private verification record directory')
    args = parser.parse_args()
    result = run_checks(args.input, args.calendar)
    if args.output:
        output = private_output(args.output, [x for x in (args.input, args.calendar) if x])
        with (output/'data_verification.json').open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
