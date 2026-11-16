"""Fixed-capacity model families for the CDCH study."""
import warnings
from time import perf_counter, process_time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch import nn

from experiment import batch_input, cdf, frequency_features, rgb_scores
from native_features import matrix, representation

FAMILIES = ("logistic", "linear_svm", "rbf_svm", "forest", "boosting", "mlp")
CANDIDATES = [f"cdch_{rep}_{name}" for rep in "ABC" for name in FAMILIES] + ["cdch_D_conv"]


class AngularCNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(32, 12, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(12, 8, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8, 1),
        )


def angular_input(x, fit):
    array = np.ascontiguousarray(((x.reshape(-1, 24, 32) - fit["mean"]) / fit["scale"])
                                 [:, fit["order"]].transpose(0, 2, 1), dtype=np.float32)
    return torch.from_numpy(array)


def fit_model(name, x, y, seed):
    rep, _, family = name.removeprefix("cdch_").rpartition("_") if name.endswith("_conv") else ("", "", "")
    if family != "conv":
        rep, family = name.removeprefix("cdch_").split("_", 1)
    converged = True
    wall, cpu = perf_counter(), process_time()
    if family == "conv":
        torch.manual_seed(seed)
        view = x.reshape(-1, 24, 32)
        fitted = {"mean": view.mean(axis=(0, 1)), "scale": np.maximum(view.std(axis=(0, 1)), 1e-6),
                  "order": (np.arange(24) if rep == "D" else np.random.default_rng(197).permutation(24))}
        inputs = angular_input(x, fitted)
        target = torch.from_numpy(y.astype(np.float32))
        net = AngularCNN()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.02)
        loss = nn.BCEWithLogitsLoss()
        random = torch.Generator().manual_seed(seed)
        for _ in range(25):
            for indices in torch.randperm(len(y), generator=random).split(24):
                optimizer.zero_grad()
                loss(net(inputs[indices]).flatten(), target[indices]).backward()
                optimizer.step()
        fitted["state"] = {key: v.detach().numpy().copy() for key, v in net.state_dict().items()}
    else:
        algorithms = {
            "logistic": LogisticRegression(max_iter=2000),
            "linear_svm": LinearSVC(C=0.1, dual="auto", max_iter=5000, random_state=seed),
            "rbf_svm": SVC(C=1, kernel="rbf", gamma="scale"),
            "forest": RandomForestClassifier(n_estimators=128, max_depth=6, min_samples_leaf=4,
                                             random_state=seed, n_jobs=1),
            "boosting": HistGradientBoostingClassifier(max_iter=80, learning_rate=0.07, max_leaf_nodes=7,
                                                        l2_regularization=1, random_state=seed),
            "mlp": MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=200,
                                 learning_rate_init=0.003, random_state=seed),
        }
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted = make_pipeline(StandardScaler(), algorithms[family]).fit(x, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in seen)
    return fitted, {"fit_wall_s": perf_counter() - wall, "fit_cpu_s": process_time() - cpu, "converged": converged}


def prepare_native(entry):
    if entry["name"].endswith("_conv"):
        network = AngularCNN()
        network.load_state_dict({key: torch.from_numpy(value) for key, value in entry["fit"]["state"].items()})
        network.eval()
        return network
    return None


def native_score(entry, x, network=None):
    fit = entry["fit"]
    if entry["name"].endswith("_conv"):
        network = prepare_native(entry) if network is None else network
        with torch.inference_mode():
            return network(angular_input(x, fit)).flatten().numpy()
    if hasattr(fit, "decision_function"):
        return fit.decision_function(x)
    return fit.predict_proba(x)[:, 1]


def feature_key(name):
    rep = name.removeprefix("cdch_").split("_")[0]
    return "C" if rep == "D" else rep


def fused(values, rule):
    # Zero-weight members are omitted from deployment and from cost attribution.
    return sum(weight * cdf(values[key], ecdf) for key, weight, ecdf in rule)


def preprocess_rgb(rgb):
    return batch_input((rgb.astype(np.float32) / 255)[None])


def score_rgb(net, prepared):
    with torch.inference_mode():
        return net(prepared).flatten().numpy()
