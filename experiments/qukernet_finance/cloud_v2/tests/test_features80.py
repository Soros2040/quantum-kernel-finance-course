"""Synthetic checks only: formulas, missing bars, timing, fitting and snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features80 import FEATURES, FEATURE_DICTIONARY, build_features, _r2, _moment
from data80 import (apply_scaler, fit_scaler, load_inputs, mrmr_select, prepare_window,
                    private_output, save_window, load_window)


def synthetic_panel():
    dates = pd.bdate_range("2019-01-01", "2024-12-31")
    rng = np.random.default_rng(31907)
    pieces = []
    for number in range(20):
        close = (10.+number)*np.exp(np.cumsum(rng.normal(.0001, .009, len(dates))))
        opening = close*np.exp(rng.normal(0., .003, len(dates)))
        high = np.maximum(opening, close)*np.exp(rng.uniform(.0001, .02, len(dates)))
        low = np.minimum(opening, close)*np.exp(-rng.uniform(.0001, .02, len(dates)))
        volume = np.exp(rng.normal(12.+number*.01, .3, len(dates)))
        pieces.append(pd.DataFrame({"symbol": f"SYNTHETIC{number:02d}", "date": dates,
                                   "open": opening, "high": high, "low": low, "close": close, "volume": volume}))
    return pd.concat(pieces, ignore_index=True), pd.DataFrame({"date": dates})


class FeatureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.calendar = synthetic_panel()
        raw, cal, identities = load_inputs(cls.raw, cls.calendar)
        cls.frame = build_features(raw, cal)
        cls.prepared = (raw, cal, identities, cls.frame)
        cls.window = prepare_window(None, None, "2022Q1", 32, _prepared=cls.prepared)

    def test_01_dictionary_and_formulas(self):
        self.assertEqual(len(FEATURES), 80)
        self.assertEqual(len(set(FEATURES)), 80)
        self.assertEqual([r["id"] for r in FEATURE_DICTIONARY], [f"F{i:02d}" for i in range(1, 81)])
        one = self.raw[self.raw.symbol == "SYNTHETIC00"].reset_index(drop=True)
        row = self.frame[(self.frame.symbol == "SYNTHETIC00") & (self.frame.j == 400)].iloc[0]
        r = np.log(one.close/one.close.shift(1))
        self.assertAlmostEqual(row.log_momentum_120, np.log(one.close[400]/one.close[280]), places=13)
        self.assertAlmostEqual(row.mean_deviation_120, one.close[400]/one.close.iloc[281:401].mean()-1, places=13)
        self.assertAlmostEqual(row.return_sd_20, r.iloc[381:401].std(ddof=1), places=13)
        self.assertAlmostEqual(row.volume_ratio_20, one.volume[400]/one.volume.iloc[381:401].mean()-1, places=13)
        segment = one.close.iloc[341:401].to_numpy()
        self.assertAlmostEqual(row.maximum_drawdown_60, np.max(1-segment/np.maximum.accumulate(segment)), places=13)
        centered = r.iloc[381:401].to_numpy()-r.iloc[381:401].mean()
        self.assertAlmostEqual(row.return_skew_20, np.mean(centered**3)/np.mean(centered**2)**1.5, places=11)
        self.assertAlmostEqual(row.y_raw, one.open[406]/one.open[401]-1, places=13)

    def test_02_all_sizes_and_purged_boundaries(self):
        for size, dates in ((16, 8), (32, 16), (150, 15)):
            w = prepare_window(None, None, "2022Q1", size, _prepared=self.prepared)
            self.assertEqual(w["x_train"].shape, (size, 40))
            self.assertEqual(w["x_valid"].shape, (size, 40))
            self.assertEqual(len({r["date"] for r in w["meta_train"]}), dates)
            self.assertEqual(len({r["date"] for r in w["meta_valid"]}), dates)
            self.assertLess(max(r["label_end"] for r in w["meta_train"]), w["audit"]["V"]["first"])
            self.assertLessEqual(max(r["label_end"] for r in w["meta_valid"]), w["audit"]["fit_cutoff"])
            self.assertTrue(all(pd.Period(r["label_end"], freq="Q") == pd.Period("2022Q1") for r in w["meta_test"]))
            counts = w["audit"]["sampling"]["train"]["security_counts"].values()
            self.assertLessEqual(max(counts)-min(counts), 1)
            self.assertFalse(w["x_train"].flags.writeable)
            self.assertTrue(np.max(np.abs(w["x_test"])) <= np.pi)

    def test_03_future_values_cannot_fit_preprocessing(self):
        changed = self.raw.copy(deep=True)
        future = changed.date >= pd.Timestamp(self.window["audit"]["V"]["first"])
        changed.loc[future, ["open", "high", "low", "close"]] *= 7.
        changed.loc[future, "volume"] *= 11.
        changed_frame = build_features(changed, self.prepared[1])
        w = prepare_window(None, None, "2022Q1", 32, _prepared=(changed, self.prepared[1], {}, changed_frame))
        self.assertEqual(w["meta_train"], self.window["meta_train"])
        self.assertEqual(w["preprocessing"], self.window["preprocessing"])
        np.testing.assert_array_equal(w["x_train"], self.window["x_train"])
        np.testing.assert_array_equal(w["y_train"], self.window["y_train"])
        self.assertFalse(np.array_equal(w["x_valid"], self.window["x_valid"]))

    def test_04_missing_history_no_compaction(self):
        one = self.raw[self.raw.symbol == "SYNTHETIC00"].iloc[:300].copy()
        cal = pd.DatetimeIndex(self.calendar.date.iloc[:300])
        one = one[one.date != cal[150]]
        frame = build_features(one, cal)
        self.assertFalse(frame.loc[150:270, "history_complete"].any())
        self.assertTrue(frame.loc[271, "history_complete"])
        self.assertTrue(np.isnan(frame.loc[151, "log_momentum_1"]))
        self.assertEqual(frame.loc[151, "date"], cal[151])

    def test_05_edge_policies(self):
        one = self.raw[self.raw.symbol == "SYNTHETIC00"].iloc[:220].copy()
        cal = pd.DatetimeIndex(self.calendar.date.iloc[:220])
        one.loc[:, ["open", "high", "low", "close"]] = 10.
        one.loc[:, "volume"] = 0.
        frame = build_features(one, cal)
        row = frame.iloc[200]
        self.assertEqual(row.body_fraction, 0.)
        self.assertEqual(row.wick_imbalance, 0.)
        self.assertEqual(row.path_efficiency_20, 0.)
        self.assertEqual(row.zero_volume_fraction_20, 1.)
        self.assertEqual(row.maximum_drawdown_60, 0.)
        self.assertTrue(np.isnan(row.return_skew_20))
        self.assertTrue(np.isnan(row.return_sign_autocorr_20))
        self.assertTrue(np.isnan(row.illiquidity_20))
        self.assertTrue(np.isnan(row.log_price_r2_20))
        self.assertTrue(np.isnan(_r2(np.repeat(np.log(10.), 20))))
        self.assertTrue(np.isnan(_moment(np.repeat(.1, 20), 3)))
        self.assertFalse(row.feature_valid)

    def test_06_train_only_filter_and_small_units(self):
        rng = np.random.default_rng(39)
        x = rng.normal(size=(32, 80))
        x[:, 0] = 1.
        x[:, 1] = x[:, 2]
        y = rng.normal(size=32)
        selected, audit = mrmr_select(x, y)
        self.assertNotIn(0, selected)
        self.assertNotIn(2, selected)  # earlier declared exact duplicate retained
        self.assertEqual(audit["eligible_count"], 78)
        tiny = np.array([[1., 1e-15], [2., 2e-15], [3., 3e-15]])
        transformed = apply_scaler(tiny, fit_scaler(tiny))
        np.testing.assert_allclose(transformed[:, 0], transformed[:, 1], atol=1e-14)
        with self.assertRaises(ValueError):
            mrmr_select(np.ones((32, 80)), y)

    def test_07_invalid_input_rejection(self):
        duplicate = pd.concat([self.raw, self.raw.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError): load_inputs(duplicate, self.calendar)
        future = self.raw.copy()
        future.loc[0, "date"] = pd.Timestamp("2025-01-02")
        with self.assertRaises(ValueError): load_inputs(future, self.calendar)
        with self.assertRaises(ValueError): load_inputs(self.raw[self.raw.symbol != "SYNTHETIC00"], self.calendar)

    def test_08_snapshot_roundtrip_and_guard(self):
        with tempfile.TemporaryDirectory(prefix="cloud_v2_synthetic_") as directory:
            record = save_window(self.window, directory, "synthetic")
            restored = load_window(Path(directory)/record["npz"])
            self.assertEqual(restored["preprocessing_hash"], self.window["preprocessing_hash"])
            self.assertEqual(restored["meta_train"], self.window["meta_train"])
            np.testing.assert_array_equal(restored["x_train"], self.window["x_train"])
            self.assertFalse(restored["x_train"].flags.writeable)
            with self.assertRaises(FileExistsError): save_window(self.window, directory, "synthetic")
        with self.assertRaises(ValueError): private_output(Path(__file__).resolve().parents[1]/"private_should_fail")

    def test_09_no_target_based_sample_substitution(self):
        frame = self.frame.copy(deep=True)
        first = self.window["meta_train"][0]
        where = (frame.symbol == first["symbol"]) & (frame.date == pd.Timestamp(first["date"]))
        frame.loc[where, "target_valid"] = False
        frame.loc[where, "y_raw"] = np.nan
        with self.assertRaisesRegex(ValueError, "no label-based substitution"):
            prepare_window(None, None, "2022Q1", 32, _prepared=(*self.prepared[:3], frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FeatureChecks))
    summary = {"kind": "synthetic unit checks only; no financial or quantum model execution", "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
               "seconds": time.perf_counter()-started,
               "details": [{"test": str(test), "traceback": message} for test, message in result.failures+result.errors]}
    if args.output:
        path = Path(args.output)
        root = private_output(path.parent)
        with (root/path.name).open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
