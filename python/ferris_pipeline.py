"""Backend-selectable CDCH inference on the archived, unmodified classifiers.

Backends
--------
canonical_js  Archived path: persistent Node worker running the checksum-verified ip-1.js
              via the archived oracle, base64 float64 transport, NumPy representations.
rust_matrix   Rust canonical matrix (cdch_ferris.cdch) + the archived NumPy representations.
              Isolates the CDCH core: identical features iff the matrix is identical.
rust          Fused Rust: image -> representation in one call; the 360x256 matrix is not
              returned to Python (and for A/C/D is never formed).

The classifier section of ``FerrisPipeline.predict`` is the archived
``native_pipeline.Pipeline.predict`` code, unchanged; only feature extraction is
dispatched to the selected backend. ``tests/test_pipeline_equivalence.py`` checks that the
``canonical_js`` backend reproduces the archived Pipeline outputs exactly.
"""
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "study"
for p in (STUDY / "pilot", STUDY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import cdch_ferris  # noqa: E402
from build_pairs import ROOT  # noqa: E402  (study/pilot)
from experiment import frequency_features  # noqa: E402
from native_features import matrix, read_rgb, representation, worker  # noqa: E402
from native_models import fused, native_score, preprocess_rgb, score_rgb  # noqa: E402
from native_pipeline import Pipeline  # noqa: E402

BACKENDS = ("canonical_js", "rust_matrix", "rust")
RUN = STUDY / "source_run"
RGB_PATH = ROOT / "results/rgb.pt"


def archived_center(rgb):
    height, width, _ = rgb.shape
    return (float(width // 2), float(height // 2))


def cdch_features(backend, rgb, kinds, process=None):
    """Representations ``kinds`` for one image as ``{kind: (dim,) array}`` plus stage timings (ns)."""
    kinds = sorted(kinds)
    clock = {}
    if backend == "canonical_js":
        cdf, stats = matrix(process, rgb)
        clock.update(stats)
        start = perf_counter_ns()
        compact = representation(cdf, "A") if set(kinds) & {"A", "B"} else None
        out = {k: (compact if k == "A" else representation(cdf, k, compact)) for k in kinds}
        clock["feature"] = perf_counter_ns() - start
    elif backend == "rust_matrix":
        start = perf_counter_ns()
        cdf = cdch_ferris.cdch(rgb, center=archived_center(rgb))
        clock["rust_matrix"] = perf_counter_ns() - start
        start = perf_counter_ns()
        compact = representation(cdf, "A") if set(kinds) & {"A", "B"} else None
        out = {k: (compact if k == "A" else representation(cdf, k, compact)) for k in kinds}
        clock["feature"] = perf_counter_ns() - start
    elif backend == "rust":
        start = perf_counter_ns()
        # D_permuted and D consume the C vector (native_models.feature_key maps D -> C).
        out = cdch_ferris.features_multi(rgb, kinds, center=archived_center(rgb))
        clock["rust_fused"] = perf_counter_ns() - start
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
    return {k: np.asarray(v) for k, v in out.items()}, clock


class FerrisPipeline(Pipeline):
    def __init__(self, models, rgb_path, outputs, backend="rust"):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
        super().__init__(models, rgb_path, outputs)
        self.backend = backend

    def worker(self):
        if self.representations and self.backend == "canonical_js":
            return worker()
        return nullcontext()

    def predict(self, path, process=None, rgb=None):
        start = perf_counter_ns()
        rgb = read_rgb(path) if rgb is None else rgb
        clock = {"decode": perf_counter_ns() - start}
        if self.representations:
            start_feature = perf_counter_ns()
            values, stats = cdch_features(self.backend, rgb, self.representations, process)
            clock.update(stats)
            features = {k: v[None] for k, v in values.items()}
            clock["cdch_total"] = perf_counter_ns() - start_feature
        else:
            features = {}
        # ---- archived classifier section (native_pipeline.Pipeline.predict), unchanged ----
        if self.statistical:
            freq_start = perf_counter_ns()
            frequency = np.asarray(frequency_features(rgb.astype(np.float32) / 255))[None]
            clock["frequency"] = perf_counter_ns() - freq_start
        if self.rgb is not None:
            rgb_start = perf_counter_ns()
            prepared = preprocess_rgb(rgb)
            clock["rgb_preprocess"] = perf_counter_ns() - rgb_start
        fit_start = perf_counter_ns()
        predictions = {n: native_score(entry, features[entry["feature"]], self.networks[n])
                       for n, entry in self.native.items()}
        if self.rgb is not None:
            cnn_start = perf_counter_ns()
            predictions["rgb"] = score_rgb(self.rgb, prepared)
            clock["cnn"] = perf_counter_ns() - cnn_start
        for n, (fit, direction) in self.statistical.items():
            vectors = frequency
            if n in {"frequency_cdch", "frequency_plus_selected_features"}:
                vectors = np.c_[frequency, features["A" if n == "frequency_cdch" else self.hybrid_key]]
            predictions[n] = (fit.decision_function(vectors) if hasattr(fit, "named_steps")
                              else direction * fit.score_samples(vectors))
        for n, rule in self.rules.items():
            predictions[n] = fused(predictions, rule)
        clock["classifier_other"] = perf_counter_ns() - fit_start - clock.get("cnn", 0)
        clock["total"] = perf_counter_ns() - start
        return {n: float(predictions[n][0]) for n in self.outputs}, clock


def decisions():
    return json.loads((RUN / "SOURCE_DECISIONS.json").read_text())


def _archived_digests():
    digests = {}
    for line in (STUDY / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        digests[(STUDY / name.lstrip("*")).resolve()] = digest
    return digests


def trusted_load(path):
    """joblib (pickle) is only used on the study's own archived checkpoints, and only after
    their SHA-256 matches study/SHA256SUMS. Never point this at untrusted files."""
    import hashlib
    path = Path(path).resolve()
    expected = _archived_digests().get(path)
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Refusing to unpickle unverified checkpoint {path}")
    return joblib.load(path)


def load_models(names, run=RUN):
    """Archived checkpoints for ``names`` (and the leaves of any fusion rules)."""
    rules = {}
    for name in names:
        rule_path = run / "models" / f"{name}_rule.joblib"
        if rule_path.exists():
            rules[name] = trusted_load(rule_path)
    leaves = set()
    for name in names:
        leaves |= {k for k, _, _ in rules[name]} if name in rules else {name}
    selected = decisions()["selected"]
    leaves.add(selected)
    native, statistical = {}, {}
    for key in sorted(leaves - {"rgb"}):
        loaded = trusted_load(run / "models" / f"{key}.joblib")
        (native if key.startswith("cdch_") else statistical)[key] = loaded
    key = "C" if selected.startswith("cdch_D_") else selected.split("_")[1]
    return dict(native=native, statistical=statistical, rules=rules, selected=selected, selected_feature=key)


def load_deploy(name, backend, run=RUN):
    """Single-output deployment, mirroring native_pipeline.load_deploy."""
    models = load_models([name], run)
    rules = {name: models["rules"][name]} if name in models["rules"] else {}
    leaves = {k for k, _, _ in rules[name]} if rules else {name}
    models["native"] = {k: v for k, v in models["native"].items() if k in leaves}
    models["statistical"] = {k: v for k, v in models["statistical"].items() if k in leaves}
    models["rules"] = rules
    return FerrisPipeline(models, RGB_PATH, [name], backend)


def model_names(run=RUN):
    import csv
    with (run / "model_inventory.csv").open(newline="") as stream:
        return [row["model"] for row in csv.DictReader(stream)]
