"""Registered stage dry-runs and an explicitly injected cloud execution entry.

CLI dry-runs never authenticate or submit. An authenticated platform/provider is
injected by the private orchestration layer; the adapter still enforces its fee,
real-device-quota, preflight, uncertainty, and account-budget gates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cloud_v2 import circuits
    from cloud_v2.search import Checkpoints, run_search
else:
    from . import circuits
    from .search import Checkpoints, run_search

PROTOCOL = Path(__file__).with_name("protocols.json")


def registry():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def profile(name):
    document = registry()
    if name not in document["stages"]:
        raise ValueError("Unknown registered stage")
    return {**document["common"], **document["stages"][name], "stage": name,
            "protocol_sha256": circuits.digest(document)}


def load_window(path):
    from cloud_v2 import data80
    path = data80.local_path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if data80.file_identity(path)["sha256"] != manifest["npz"]["sha256"]:
        raise ValueError("Prepared NPZ checksum does not match its frozen manifest")
    result = data80.load_window(path)
    metadata = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    if data80._json_hash(metadata["preprocessing"]) != metadata["preprocessing_hash"]:
        raise ValueError("Frozen preprocessing checksum mismatch")
    for name, record in metadata["source_code"].items():
        if Path(name).name != name or data80.file_identity(Path(__file__).parent/name)["sha256"] != record["sha256"]:
            raise ValueError("Data preparation source differs from the frozen snapshot")
    if any(manifest.get(k) != v for k, v in metadata.items()):
        raise ValueError("NPZ metadata differs from its frozen manifest")
    return result


def dry_run(stage, n_test, repetitions=1):
    config = profile(stage)
    cost = circuits.budget_estimate(config["n_train"], config["n_valid"], n_test, config["labels"],
        config["top_k"], config["configurations"], config["steps"], config["batch_size"],
        config["max_parameters"], config["shots"], repetitions)
    return {"status": "not_run", "cloud_submissions": 0, "profile": config,
            "n_test": n_test, "budget_upper_bound": cost,
            "notes": ["Counts are one method arm and task per market window, not the entire experiment matrix.",
                      "Actual O row count is read from the prepared NPZ when provided.",
                      "A code-complete stage is not an executed or scientifically accepted experiment.",
                      "Scaled parameter caps and optimization steps differ from the original paper."]}


def main(argv=None, provider=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(registry()["stages"]), default="pilot8")
    parser.add_argument("--window", help="Admitted local NPZ; no pickle")
    parser.add_argument("--output", help="Private local output directory outside the public course")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--method", choices=["qukernet", "random"], default="qukernet")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--n-test", type=int, default=220, help="Dry-run assumption only when --window is absent")
    parser.add_argument("--repetitions", type=int, default=1, help="Dry-run accounting; execution runs exactly one seed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    window = load_window(args.window) if args.window else None
    result = dry_run(args.stage, len(window["x_test"]) if window else args.n_test, args.repetitions)
    if not args.execute:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    if provider is None:
        raise RuntimeError("Submission blocked: inject the authenticated, preflighted, fee/quota-gated cloud provider programmatically; the CLI does not read credentials or bypass gates")
    if window is None or args.output is None:
        raise ValueError("Execution needs --window and a private --output")
    if args.repetitions != 1:
        raise ValueError("Execute one registered seed at a time; account ledger owns cumulative budgets")
    if not hasattr(provider, "execution_identity"):
        raise ValueError("Production provider must publish a non-secret execution identity")
    result = run_search(window, profile(args.stage), provider, args.output, args.seed,
                        args.task, args.method, source=Path(args.window).resolve().parent)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
