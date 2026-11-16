"""The canonical_js backend of FerrisPipeline must reproduce the archived Pipeline exactly,
and the Rust backends must agree with it to within floating-point noise.

Requires FERRIS_PAIRS (regenerated pair directory); skipped otherwise.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ferris_pipeline as fp  # noqa: E402
import native_pipeline  # noqa: E402  (archived)

PAIRS = os.environ.get("FERRIS_PAIRS")
pytestmark = pytest.mark.skipif(not PAIRS, reason="set FERRIS_PAIRS to the regenerated pair directory")
MODELS = ["cdch_A_logistic", "cdch_B_linear_svm", "cdch_C_mlp", "cdch_D_conv", "frequency_cdch",
          "frequency_cdch_score_fusion", "ensemble_cdch_score_fusion", "frequency", "rgb"]


def test_backends_match_archived_pipeline():
    rows = [r for r in json.loads((fp.ROOT / "results/manifest.json").read_text())
            if r["split"] == "test" and r["variant"] != "lr"][:12]
    paths = [Path(PAIRS) / r["path"] for r in rows]
    for name in MODELS:
        decision = fp.decisions()
        archived = native_pipeline.load_deploy(name, fp.RUN, fp.RGB_PATH, decision["selected"], "A")
        ferris = {b: fp.load_deploy(name, b) for b in fp.BACKENDS}
        with archived.worker() as w_arch, ferris["canonical_js"].worker() as w_js:
            for path in paths:
                want, _ = archived.predict(path, w_arch)
                got_js, _ = ferris["canonical_js"].predict(path, w_js)
                assert got_js == want, (name, path)  # bit-identical
                for b in ("rust_matrix", "rust"):
                    got, _ = ferris[b].predict(path, None)
                    assert got.keys() == want.keys()
                    for k in want:
                        assert abs(got[k] - want[k]) <= 1e-12, (name, b, path, got[k], want[k])
