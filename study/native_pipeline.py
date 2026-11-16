"""Shared live inference entry point for benchmarks and label-blind scoring."""
from contextlib import nullcontext
from time import perf_counter_ns

import joblib
import numpy as np
import torch

from experiment import TinyRGB, frequency_features
from native_features import matrix, read_rgb, representation, worker
from native_models import fused, prepare_native, preprocess_rgb, score_rgb


class Pipeline:
    def __init__(self, models, rgb_path, outputs):
        self.outputs = outputs
        self.rules = {n: models["rules"][n] for n in outputs if n in models["rules"]}
        leaves = {key for n in outputs if n not in self.rules for key in [n]}
        leaves.update(key for rule in self.rules.values() for key, _, _ in rule)
        self.leaves = sorted(leaves)
        self.native = {n: models["native"][n] for n in leaves if n in models["native"]}
        self.networks = {n: prepare_native(entry) for n, entry in self.native.items()}
        self.statistical = {n: models["statistical"][n] for n in leaves if n in models["statistical"]}
        self.representations = {entry["feature"] for entry in self.native.values()}
        self.hybrid_key = models.get("selected_feature")
        if "frequency_plus_selected_features" in self.statistical:
            if self.hybrid_key is None:
                self.hybrid_key = models["native"][models["selected"]]["feature"]
            self.representations.add(self.hybrid_key)
        if "frequency_cdch" in self.statistical:
            self.representations.add("A")
        self.rgb = None
        if "rgb" in leaves:
            self.rgb = TinyRGB()
            self.rgb.load_state_dict(torch.load(rgb_path, map_location="cpu", weights_only=True))
            self.rgb.eval()
        if leaves != (set(self.native) | set(self.statistical) | ({"rgb"} if self.rgb is not None else set())):
            raise ValueError(f"Missing model for {leaves}")

    def worker(self):
        return worker() if self.representations else nullcontext()

    def predict(self, path, process):
        from native_models import native_score

        start = perf_counter_ns()
        rgb = read_rgb(path)
        clock = {"decode": perf_counter_ns() - start}
        if self.representations:
            cdf, stats = matrix(process, rgb)
            clock.update(stats)
            start_feature = perf_counter_ns()
            compact = representation(cdf, "A") if self.representations & {"A", "B"} else None
            features = {k: (compact if k == "A" else representation(cdf, k, compact))[None]
                        for k in sorted(self.representations)}
            clock["feature"] = perf_counter_ns() - start_feature
        else:
            features = {}
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


def load_deploy(name, run, rgb_path, selected, selected_feature):
    rules = {}
    rule_path = run / "models" / f"{name}_rule.joblib"
    if rule_path.exists():
        rules[name] = joblib.load(rule_path)
    leaves = {key for key, _, _ in rules[name]} if name in rules else {name}
    native, statistical = {}, {}
    for key in leaves - {"rgb"}:
        loaded = joblib.load(run / "models" / f"{key}.joblib")
        (native if key.startswith("cdch_") else statistical)[key] = loaded
    return Pipeline(dict(native=native, statistical=statistical, rules=rules,
                         selected=selected, selected_feature=selected_feature), rgb_path, [name])
