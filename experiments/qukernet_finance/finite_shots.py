"""Finite-shot final-selection diagnosis for five frozen exact-search kernels.

This is independent binomial simulation, not cloud execution. The selector
accepts only measured matrices, training/validation labels, dates, and lambda.
Exact reference losses are computed only after all selections are persisted.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

from run_search import (output_root, write_json, saved_stage, file_hash, array_hash,
                        code_hashes, digest, utc, plain, environment)
from modeling import krr, date_equal_mse

SHOTS = (32, 128, 512)
REPETITIONS = 20
DELTAS = (.01, .02, .05)
EXPECTED_NOMINAL_SHOTS = 307507200


def select_candidate(measured_train, measured_cross, y_train, y_valid, dates, regularization=.1):
    """No exact losses, oracle features, pool-search scores, or model bundles.

    Positive-spectrum repair, compatible cross mapping, centering and scale
    are refitted to EACH noisy training matrix by modeling.krr. Selection uses
    date-equal MSE / sample variance of the supplied training targets.
    """
    train, cross = np.asarray(measured_train, float), np.asarray(measured_cross, float)
    y_train, y_valid = np.asarray(y_train, float), np.asarray(y_valid, float)
    if train.ndim != 3 or cross.ndim != 3 or train.shape[0] != cross.shape[0]:
        raise ValueError("Candidate axes must match")
    if train.shape[1:] != (len(y_train), len(y_train)) or cross.shape[1:] != (len(y_valid), len(y_train)):
        raise ValueError("Measured kernel and target dimensions do not match")
    variance = float(np.var(y_train, ddof=1))
    if not np.isfinite(variance) or variance <= 1e-24:
        raise ValueError("Nondegenerate training targets required")
    if regularization != .1:
        raise ValueError("This frozen diagnosis uses lambda=.1 without calibration")
    losses, diagnostics = [], []
    for K, B in zip(train, cross):
        try:
            prediction, info = krr(K, B, y_train, regularization)
            loss = date_equal_mse(y_valid, prediction, dates) / variance
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite candidate loss")
            losses.append(float(loss))
            diagnostics.append({"status": "ok", "mapping": info})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            losses.append(None)
            diagnostics.append({"status": "invalid", "reason": f"{type(exc).__name__}: {exc}"})
    good = [i for i, value in enumerate(losses) if value is not None]
    chosen = min(good, key=lambda i: (losses[i], i)) if good else None
    return {"status": "ok" if chosen is not None else "selection_failed", "selected_index": chosen,
            "measured_losses": losses, "candidate_diagnostics": diagnostics,
            "tie_rule": "Exact floating-point ties choose lowest frozen candidate index"}


def evaluate_choice(selected_index, exact_losses, deltas=DELTAS):
    """Post-selection only. Cannot change or rerun the already saved choice."""
    exact = np.asarray(exact_losses, float)
    if exact.ndim != 1 or not np.isfinite(exact).all():
        raise ValueError("A complete finite exact reference is required")
    if selected_index is None:
        return {"regret": None, "harmful_misselection": {str(d): None for d in deltas}}
    if not 0 <= int(selected_index) < len(exact):
        raise ValueError("Selected candidate out of range")
    regret = float(exact[int(selected_index)] - exact.min())
    return {"regret": regret, "harmful_misselection": {str(d): regret > d for d in deltas}}


def simulate_measurements(prob_train, prob_cross, shots, repetition, seed):
    """Independent candidate/pair streams; known ideal diagonal costs no shots."""
    pK, pB = np.asarray(prob_train, float), np.asarray(prob_cross, float)
    if shots not in SHOTS or not 0 <= repetition < REPETITIONS:
        raise ValueError("Out-of-protocol shots or repetition")
    if not np.isfinite(pK).all() or not np.isfinite(pB).all():
        raise ValueError("Sampling probabilities must be finite")
    if min(pK.min(), pB.min()) < -1e-10 or max(pK.max(), pB.max()) > 1 + 1e-10:
        raise ValueError("Sampling probabilities outside tolerated roundoff")
    candidates, n, _ = pK.shape
    m = pB.shape[1]
    upper = np.triu_indices(n, 1)
    train_counts = np.zeros_like(pK, dtype=np.int64)
    cross_counts = np.zeros_like(pB, dtype=np.int64)
    for candidate in range(candidates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, shots, repetition, candidate, 99173]))
        draws = rng.binomial(shots, np.clip(pK[candidate][upper], 0, 1))
        train_counts[candidate][upper] = draws
        train_counts[candidate][(upper[1], upper[0])] = draws
        np.fill_diagonal(train_counts[candidate], shots)
        cross_counts[candidate] = rng.binomial(shots, np.clip(pB[candidate], 0, 1))
    nominal = candidates * (n * (n - 1) // 2 + m * n) * shots
    return train_counts / shots, cross_counts / shots, {
        "nominal_shots": int(nominal), "real_cloud_shots": 0,
        "train_counts_sha256": array_hash(train_counts), "cross_counts_sha256": array_hash(cross_counts),
        "independence": "Different candidates and upper-triangle/cross pairs sampled independently",
        "known_diagonal": "One by normalized ideal fidelity; no diagonal shots charged",
        "shared_reads": False, "seed": seed, "shots": shots, "repetition": repetition}


def wilson(events, total):
    if total == 0:
        return {"events": int(events), "n": 0, "rate": None, "lower": None, "upper": None}
    z, p = 1.959963984540054, events / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"events": int(events), "n": int(total), "rate": float(p),
            "lower": float(max(0., center - half)), "upper": float(min(1., center + half)),
            "interval": "95% Wilson, repetitions conditional on this fixed candidate/data pool"}


def summarize(records, exact_losses):
    exact = np.asarray(exact_losses, float)
    output = {}
    for shots in SHOTS:
        subset = [r for r in records if r["shots"] == shots]
        successes = [r for r in subset if r["selection"]["selected_index"] is not None]
        counts = np.bincount([r["selection"]["selected_index"] for r in successes], minlength=len(exact))
        n, s = len(subset), len(successes)
        pair_agreement = float(np.sum(counts * (counts - 1)) / (s * (s - 1))) if s > 1 else None
        rows = {"n_requested": REPETITIONS, "n_completed": n,
                "selection_failure": wilson(n - s, n), "selection_counts": counts.tolist(),
                "frequencies_all_repetitions": (counts / n).tolist() if n else None,
                "frequencies_given_success": (counts / s).tolist() if s else None,
                "repeat_pair_agreement_given_success": pair_agreement,
                "modal_selection_fraction_given_success": float(counts.max() / s) if s else None,
                "mean_regret_given_success": float(np.mean([r["posthoc"]["regret"] for r in successes])) if s else None,
                "harmful_misselection": {}, "paired_gaps": [],
                "candidate_invalid_counts": [sum(r["selection"]["candidate_diagnostics"][j]["status"] != "ok" for r in subset)
                                             for j in range(len(exact))]}
        for delta in DELTAS:
            bad = sum(r["posthoc"]["harmful_misselection"][str(delta)] for r in successes)
            rows["harmful_misselection"][str(delta)] = {
                "given_success": wilson(bad, s), "misselection_or_failure": wilson(bad + n - s, n)}
        for a, b in itertools.combinations(range(len(exact)), 2):
            differences = [r["selection"]["measured_losses"][a] - r["selection"]["measured_losses"][b]
                           for r in subset if r["selection"]["measured_losses"][a] is not None
                           and r["selection"]["measured_losses"][b] is not None]
            exact_gap = float(exact[a] - exact[b])
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else None
            reversals = sum(x * exact_gap < 0 for x in differences) if abs(exact_gap) > 1e-12 else None
            rows["paired_gaps"].append({"candidate_i": a, "candidate_j": b, "exact_gap": exact_gap,
                "n_jointly_valid": len(differences), "estimated_gap_mean": float(np.mean(differences)) if differences else None,
                "estimated_gap_sd": sd, "absolute_gap_to_sd": abs(exact_gap) / sd if sd is not None and sd > 0 else None,
                "strict_sign_reversal": wilson(reversals, len(differences)) if reversals is not None else None,
                "exact_tie_tolerance": 1e-12})
        output[str(shots)] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Private finite_input.json from the approved exact search")
    parser.add_argument("--output", required=True, help="Separate private local directory")
    parser.add_argument("--seed", type=int, default=20261109, help="Fixed finite-sampling seed, independent streams per repetition")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from data_pipeline import local_path
    source = local_path(args.source)
    source_record = json.loads(source.read_text(encoding="utf-8"))
    if source_record.get("status") != "complete" or len(source_record.get("candidate_hashes", [])) != 5:
        raise ValueError("Five complete frozen tuned candidates are required; cannot substitute candidates")
    if (source_record.get("task"), source_record.get("quarter"), source_record.get("seed")) != ("regression", "2024Q1", 20261109):
        raise ValueError("Finite source must be the approved 2024Q1 regression first-seed pool")
    arrays_path = source.with_suffix(".npz")
    if file_hash(arrays_path) != source_record["arrays_sha256"]:
        raise ValueError("Frozen probability artifact checksum mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        K, B, y, yv, dates = [np.array(archive[k], copy=True) for k in ("K", "B", "y_train", "y_valid", "dates")]
    if K.shape != (5, 64, 64) or B.shape != (5, 40, 64) or y.shape != (64,) or yv.shape != (40,):
        raise ValueError("Approved finite diagnosis requires exactly five 64/40 candidate kernels")
    if not np.allclose(K, K.transpose(0, 2, 1), atol=1e-10, rtol=0) or not np.allclose(np.diagonal(K, axis1=1, axis2=2), 1., atol=1e-10, rtol=0):
        raise ValueError("Exact source kernels must be symmetric normalized fidelity kernels")
    out = output_root(args.output, source.parent)
    contract = {"source_json_sha256": file_hash(source), "source_npz_sha256": file_hash(arrays_path),
                "candidate_hashes": source_record["candidate_hashes"], "code_hashes": code_hashes(),
                "seed": args.seed, "shots": SHOTS, "repetitions": REPETITIONS, "lambda": .1,
                "deltas": DELTAS, "environment": environment(), "source_search_run_hash": source_record["run_hash"]}
    run_hash = digest(contract)
    manifest = out.path("manifest.json")
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not args.resume or previous.get("run_hash") != run_hash:
            raise RuntimeError("Changed finite protocol/source or existing output; select a new private directory")
    else:
        write_json(out, "manifest.json", {"run_hash": run_hash, "contract": contract, "created_utc": utc()})
    records = []
    started = time.perf_counter()
    try:
        for shots in SHOTS:
            for repetition in range(REPETITIONS):
                def repetition_work():
                    measured_K, measured_B, sampling = simulate_measurements(K, B, shots, repetition, args.seed)
                    # This function receives NO exact losses or exact features.
                    selection = select_candidate(measured_K, measured_B, y, yv, dates, .1)
                    return {"shots": shots, "repetition": repetition, "sampling": sampling, "selection": selection}
                record = saved_stage(out, f"selections/S{shots}_R{repetition:02d}.json", run_hash,
                                     args.resume, repetition_work)
                records.append(record)
            write_json(out, "progress.json", {"state": "selecting", "completed": len(records), "updated_utc": utc()})
        # Exact reference is deliberately evaluated AFTER all 60 frozen choices.
        exact_losses = []
        variance = float(np.var(y, ddof=1))
        for exact_K, exact_B in zip(K, B):
            prediction, _ = krr(exact_K, exact_B, y, .1)
            exact_losses.append(date_equal_mse(yv, prediction, dates) / variance)
        for record in records:
            record["posthoc"] = evaluate_choice(record["selection"]["selected_index"], exact_losses)
        nominal = sum(r["sampling"]["nominal_shots"] for r in records)
        if nominal != EXPECTED_NOMINAL_SHOTS:
            raise AssertionError("Nominal measurement budget differs from frozen protocol")
        result = {"status": "complete", "run_hash": run_hash, "candidate_hashes": source_record["candidate_hashes"],
                  "exact_standardized_date_losses_posthoc": exact_losses,
                  "exact_best_indices": np.flatnonzero(np.asarray(exact_losses) == min(exact_losses)).tolist(),
                  "by_shots": summarize(records, exact_losses), "nominal_shots": nominal, "calibration_shots": 0,
                  "real_cloud_shots": 0, "observed_seconds": time.perf_counter() - started,
                  "pool_selection_scope": "The five candidates were constructed by exact search; only final finite-shot selection is isolated",
                  "delta_units": "Training-variance-standardized date-equal prediction MSE, not investment return or monetary cost",
                  "noise_scope": "Ideal independent binomial sampling; no device noise or physical IID certification",
                  "reference_scope": "The oracle is fixed-lambda exact fitted validation loss; no guarantee about unseen test ranking",
                  "failure_handling": "Candidate failures recorded; all-invalid repetitions remain failures, not zero losses",
                  "records": records}
        write_json(out, "finite_results.json", result)
        write_json(out, "progress.json", {"state": "complete", "completed": len(records), "updated_utc": utc()})
        print(json.dumps({"status": "complete", "repetitions": len(records), "nominal_shots": nominal,
                          "real_cloud_shots": 0}), flush=True)
        return 0
    except BaseException as exc:
        write_json(out, "failure.json", {"state": "stopped", "utc": utc(), "type": type(exc).__name__,
                  "error": str(exc), "completed_repetitions": len(records),
                  "completed_nominal_shots": sum(r["sampling"]["nominal_shots"] for r in records)})
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
