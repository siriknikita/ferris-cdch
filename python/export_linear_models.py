"""Export the archived linear CDCH checkpoints (StandardScaler + LogisticRegression/LinearSVC)
to JSON for the self-contained Rust CLI (`cdch score`). Weights are copied, never refit.

    python export_linear_models.py --out ../models
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ferris_pipeline import RUN, decisions, trusted_load

MODELS = [f"cdch_{rep}_{fam}" for rep in "ABC" for fam in ("logistic", "linear_svm")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = decisions()["thresholds"]
    for name in MODELS:
        path = RUN / "models" / f"{name}.joblib"
        entry = trusted_load(path)
        pipe = entry["fit"]
        scaler, linear = pipe.named_steps["standardscaler"], pipe.steps[-1][1]
        coef = np.asarray(linear.coef_, np.float64).reshape(-1)
        payload = {
            "name": name, "feature": entry["feature"], "kind": type(linear).__name__,
            "dim": int(coef.size), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": coef.tolist(), "intercept": float(np.asarray(linear.intercept_).reshape(-1)[0]),
            "threshold": thresholds[name],
            "source_checkpoint": str(path.relative_to(RUN.parent)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (args.out / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(name, payload["feature"], payload["dim"], payload["kind"])


if __name__ == "__main__":
    main()
