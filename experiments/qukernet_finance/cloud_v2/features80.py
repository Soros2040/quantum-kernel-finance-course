"""Calendar-aligned, trailing OHLCV features for a D0 diagnostic protocol.

All windows include the decision date unless an explicit lag is written. The
longest lag is 120 market sessions, requiring 121 price observations. Missing
calendar sessions are never compressed or filled. This module has no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_dictionary():
    rows = []
    def add(name, formula, window, group, edge="NaN if required observations or denominator are invalid"):
        rows.append({"id": f"F{len(rows)+1:02d}", "name": name, "group": group,
                     "formula": formula, "window_sessions": window, "edge_policy": edge,
                     "timing": "available at decision-date close; trailing calendar-aligned window"})
    for h in (1, 2, 3, 5, 10, 15, 20, 40, 60, 120):
        add(f"log_momentum_{h}", f"log(C[t]/C[t-{h}])", h, "price")
    for name, formula in [("overnight_gap", "log(O[t]/C[t-1])"), ("intraday_return", "log(C[t]/O[t])"),
                          ("log_range", "log(H[t]/L[t])"),
                          ("true_range", "max(H-L,abs(H-Cprev),abs(L-Cprev))/Cprev"),
                          ("body_fraction", "(C-O)/(H-L)"),
                          ("wick_imbalance", "(H-max(O,C)-min(O,C)+L)/(H-L)")]:
        add(name, formula, 1, "price", "H=L gives 0 for body/wick; invalid OHLC gives NaN")
    for h in (5, 10, 20, 40, 60, 120):
        add(f"mean_deviation_{h}", f"C[t]/mean_{h}(C)-1", h, "trend")
    for h in (20, 60):
        add(f"log_price_slope_{h}", f"OLS slope of log(C) on 0..{h-1}", h, "trend")
    for h in (20, 60):
        add(f"log_price_r2_{h}", f"OLS R2 of log(C) on 0..{h-1}", h, "trend", "zero response variance gives NaN")
    for h in (20, 60):
        add(f"high_distance_{h}", f"C[t]/max_{h}(H)-1", h, "trend")
    for h in (5, 10, 20, 40, 60):
        add(f"return_rms_{h}", f"sqrt(mean_{h}(r^2)); r=log(C/Cprev)", h, "volatility")
    for h in (20, 60):
        add(f"return_sd_{h}", f"std_{h}(r,ddof=1)", h, "volatility")
    for h in (20, 60):
        add(f"parkinson_{h}", f"sqrt(mean_{h}(log(H/L)^2)/(4*log(2)))", h, "volatility")
    for h in (20, 60):
        add(f"garman_klass_{h}", f"sqrt(mean_{h}(0.5*log(H/L)^2-(2*log(2)-1)*log(C/O)^2))", h, "volatility")
    add("mean_log_range_20", "mean_20(log(H/L))", 20, "volatility")
    for h in (5, 10, 20, 60):
        add(f"volume_ratio_{h}", f"V[t]/mean_{h}(V)-1", h, "volume")
    for h in (1, 5, 20):
        add(f"log1p_volume_change_{h}", f"u[t]-u[t-{h}]; u=log1p(V)", h, "volume")
    for h in (20, 60):
        add(f"volume_cv_{h}", f"std_{h}(V,ddof=1)/mean_{h}(V)", h, "volume")
    add("log1p_volume_slope_20", "OLS slope of u=log1p(V) on 0..19", 20, "volume")
    add("log1p_volume_change_sd_20", "std_20(u-u.shift(1),ddof=1)", 20, "volume")
    add("zero_volume_fraction_20", "mean_20(1[V=0])", 20, "volume")
    for h in (20, 60):
        add(f"return_volume_change_corr_{h}", f"corr_{h}(r,delta u)", h, "price_volume", "zero variance gives NaN")
    for h in (20, 60):
        add(f"abs_return_volume_corr_{h}", f"corr_{h}(abs(r),u)", h, "price_volume", "zero variance gives NaN")
    for h in (5, 20, 60):
        add(f"signed_volume_{h}", f"sum_{h}(sign(r)*V)/sum_{h}(V)", h, "price_volume")
    for h in (20, 60):
        add(f"clv_volume_{h}", f"sum_{h}(CLV*V)/sum_{h}(V); CLV=(2*C-H-L)/(H-L)", h, "price_volume", "H=L gives CLV=0; zero total volume gives NaN")
    add("volume_weighted_return_20", "sum_20(V*r)/sum_20(V)", 20, "price_volume")
    for h in (10, 20, 60):
        add(f"downside_rms_{h}", f"sqrt(mean_{h}(min(r,0)^2))", h, "tail")
    for h in (20, 60):
        add(f"return_skew_{h}", f"m3/m2^1.5 over {h} r; mk=mean((r-mean(r))^k)", h, "tail", "zero m2 gives NaN")
    for h in (20, 60):
        add(f"return_excess_kurtosis_{h}", f"m4/m2^2-3 over {h} r", h, "tail", "zero m2 gives NaN")
    add("negative_return_fraction_20", "mean_20(1[r<0])", 20, "tail")
    add("maximum_negative_return_20", "max_20(max(-r,0))", 20, "tail")
    add("maximum_drawdown_60", "max_j(1-C[j]/max_{i<=j}(C[i])); both i,j inside latest 60 values", 60, "tail")
    for h in (20, 60):
        add(f"illiquidity_{h}", f"mean(abs(r)/(C*V)) over positive-volume values in {h} sessions", h, "liquidity", "at least ceil(.8*h) positive-volume values; missing price/volume invalidates window")
    for h in (5, 60):
        add(f"relative_range_{h}", f"mean_{h}(2*(H-L)/(H+L))", h, "liquidity")
    for h in (20, 60):
        add(f"return_sign_autocorr_{h}", f"corr_{h}(sign(r[i]),sign(r[i-1]))", h, "path", "zero variance gives NaN")
    for h in (20, 60):
        add(f"path_efficiency_{h}", f"abs(sum_{h}(r))/sum_{h}(abs(r))", h, "path", "all-zero valid path gives 0")
    assert len(rows) == 80
    return rows


FEATURE_DICTIONARY = feature_dictionary()
FEATURES = [row["name"] for row in FEATURE_DICTIONARY]


def _slope(values):
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float(x @ (values-values.mean()) / (x @ x))


def _r2(values):
    if np.ptp(values) == 0.0:
        return np.nan
    centered = values-values.mean()
    ss = float(centered @ centered)
    if ss == 0.0:
        return np.nan
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float(np.clip((x @ centered)**2 / ((x @ x)*ss), 0., 1.))


def _moment(values, power):
    if np.ptp(values) == 0.0:
        return np.nan
    centered = values-values.mean()
    m2 = np.mean(centered**2)
    if m2 == 0.0:
        return np.nan
    return float(np.mean(centered**power)/m2**(power/2.) - (3. if power == 4 else 0.))


def build_features(raw, calendar):
    """Return 80 raw features and D0 targets, with independent validity flags.

    Invalid bars (nonpositive prices, negative volume, inconsistent OHLC) become
    missing, including targets using their opens. Valid zero-volume bars remain.
    """
    calendar = pd.DatetimeIndex(calendar)
    tables = []
    for symbol, group in raw.groupby("symbol", sort=True):
        bars = group.set_index("date").reindex(calendar)
        p = bars[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(p).all(axis=1)
        valid = (finite & (p[["open", "high", "low", "close"]] > 0).all(axis=1)
                 & (p.volume >= 0) & (p.high >= p[["open", "close", "low"]].max(axis=1))
                 & (p.low <= p[["open", "close", "high"]].min(axis=1)))
        p = p.where(valid, np.nan)
        O, H, L, C, V = [p[name] for name in ("open", "high", "low", "close", "volume")]
        r, u = np.log(C/C.shift(1)), np.log1p(V)
        du, width, log_range = u.diff(), H-L, np.log(H/L)
        f = {}
        def mean(x, h): return x.rolling(h, min_periods=h).mean()
        def total(x, h): return x.rolling(h, min_periods=h).sum()
        def corr(a, b, h):
            value = a.rolling(h, min_periods=h).corr(b)
            variance_ok = (a.rolling(h, min_periods=h).var() > 0) & (b.rolling(h, min_periods=h).var() > 0)
            return value.where(variance_ok).clip(-1., 1.)
        for h in (1, 2, 3, 5, 10, 15, 20, 40, 60, 120): f[f"log_momentum_{h}"] = np.log(C/C.shift(h))
        f["overnight_gap"], f["intraday_return"], f["log_range"] = np.log(O/C.shift(1)), np.log(C/O), log_range
        f["true_range"] = pd.concat([width, (H-C.shift(1)).abs(), (L-C.shift(1)).abs()], axis=1).max(axis=1, skipna=False)/C.shift(1)
        f["body_fraction"] = ((C-O)/width).mask(width.eq(0), 0.)
        f["wick_imbalance"] = ((H-p[["open", "close"]].max(axis=1)-p[["open", "close"]].min(axis=1)+L)/width).mask(width.eq(0), 0.)
        for h in (5, 10, 20, 40, 60, 120): f[f"mean_deviation_{h}"] = C/mean(C, h)-1
        for h in (20, 60):
            f[f"log_price_slope_{h}"] = np.log(C).rolling(h, min_periods=h).apply(_slope, raw=True)
            f[f"log_price_r2_{h}"] = np.log(C).rolling(h, min_periods=h).apply(_r2, raw=True)
            f[f"high_distance_{h}"] = C/H.rolling(h, min_periods=h).max()-1
        for h in (5, 10, 20, 40, 60): f[f"return_rms_{h}"] = np.sqrt(mean(r*r, h))
        for h in (20, 60):
            f[f"return_sd_{h}"] = r.rolling(h, min_periods=h).std(ddof=1)
            f[f"parkinson_{h}"] = np.sqrt(mean(log_range**2, h)/(4*np.log(2)))
            gk = mean(.5*log_range**2-(2*np.log(2)-1)*np.log(C/O)**2, h)
            f[f"garman_klass_{h}"] = np.sqrt(gk.where(gk >= -1e-15).clip(lower=0.))
        f["mean_log_range_20"] = mean(log_range, 20)
        for h in (5, 10, 20, 60): f[f"volume_ratio_{h}"] = V/mean(V, h)-1
        for h in (1, 5, 20): f[f"log1p_volume_change_{h}"] = u-u.shift(h)
        for h in (20, 60): f[f"volume_cv_{h}"] = V.rolling(h, min_periods=h).std(ddof=1)/mean(V, h)
        f["log1p_volume_slope_20"] = u.rolling(20, min_periods=20).apply(_slope, raw=True)
        f["log1p_volume_change_sd_20"] = du.rolling(20, min_periods=20).std(ddof=1)
        f["zero_volume_fraction_20"] = mean(V.eq(0).astype(float).where(V.notna()), 20)
        for h in (20, 60):
            f[f"return_volume_change_corr_{h}"] = corr(r, du, h)
            f[f"abs_return_volume_corr_{h}"] = corr(r.abs(), u, h)
        for h in (5, 20, 60): f[f"signed_volume_{h}"] = total(np.sign(r)*V, h)/total(V, h)
        clv = ((2*C-H-L)/width).mask(width.eq(0), 0.)
        for h in (20, 60): f[f"clv_volume_{h}"] = total(clv*V, h)/total(V, h)
        f["volume_weighted_return_20"] = total(V*r, 20)/total(V, 20)
        for h in (10, 20, 60): f[f"downside_rms_{h}"] = np.sqrt(mean(r.clip(upper=0)**2, h))
        for h in (20, 60):
            f[f"return_skew_{h}"] = r.rolling(h, min_periods=h).apply(_moment, raw=True, args=(3,))
            f[f"return_excess_kurtosis_{h}"] = r.rolling(h, min_periods=h).apply(_moment, raw=True, args=(4,))
        f["negative_return_fraction_20"] = mean(r.lt(0).astype(float).where(r.notna()), 20)
        f["maximum_negative_return_20"] = (-r).clip(lower=0).rolling(20, min_periods=20).max()
        f["maximum_drawdown_60"] = C.rolling(60, min_periods=60).apply(lambda a: float(np.max(1-a/np.maximum.accumulate(a))), raw=True)
        for h in (20, 60):
            available = r.notna() & C.notna() & V.notna()
            value = (r.abs()/(C*V)).where(V > 0)
            count = value.rolling(h, min_periods=1).count()
            f[f"illiquidity_{h}"] = value.rolling(h, min_periods=int(np.ceil(.8*h))).mean().where((count >= np.ceil(.8*h)) & available.rolling(h, min_periods=h).sum().eq(h))
        for h in (5, 60): f[f"relative_range_{h}"] = mean(2*width/(H+L), h)
        for h in (20, 60):
            f[f"return_sign_autocorr_{h}"] = corr(np.sign(r), np.sign(r).shift(1), h)
            denominator = total(r.abs(), h)
            f[f"path_efficiency_{h}"] = (total(r, h).abs()/denominator).mask(denominator.eq(0), 0.)
        table = pd.DataFrame({name: f[name].to_numpy() for name in FEATURES})
        table = table.replace([np.inf, -np.inf], np.nan)
        table.insert(0, "j", np.arange(len(calendar)))
        table.insert(0, "date", calendar)
        table.insert(0, "symbol", str(symbol))
        # Explicit chronological history excludes gaps even for endpoint-only lags.
        table["history_complete"] = valid.rolling(121, min_periods=121).sum().eq(121).to_numpy()
        table["feature_valid"] = np.isfinite(table[FEATURES]).all(axis=1) & table.history_complete
        table["bar_valid"] = valid.to_numpy()
        table["y_raw"] = (O.shift(-6)/O.shift(-1)-1.).to_numpy()
        table["target_valid"] = np.isfinite(table.y_raw)
        table["label_start_j"], table["label_end_j"] = table.j+1, table.j+6
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
