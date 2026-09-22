"""
Smoke and correctness tests.

Run:  pytest -q

These run the real app headlessly via Streamlit's AppTest harness, so a broken
widget, a bad asset path or a metric regression fails here rather than in front
of a visitor.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
APP = str(ROOT / "app.py")
META = json.loads((ASSETS / "meta.json").read_text())
STRATEGIES = META["strategies"]


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------
def test_all_expected_assets_exist():
    expected = ["meta.json", "y_val.npy", "y_test.npy"]
    for tag in STRATEGIES:
        for split in ("val", "test"):
            for model in ("svm", "dt", "ens"):
                expected.append(f"{model}_{tag}_proba_{split}.npy")
    missing = [f for f in expected if not (ASSETS / f).exists()]
    assert not missing, f"missing assets: {missing} — run prepare_data.py"


@pytest.mark.parametrize("tag", STRATEGIES)
@pytest.mark.parametrize("split", ["val", "test"])
def test_shapes_agree(tag, split):
    y = np.load(ASSETS / f"y_{split}.npy")
    for model in ("svm", "dt", "ens"):
        p = np.load(ASSETS / f"{model}_{tag}_proba_{split}.npy")
        assert p.shape == y.shape, f"{model}_{tag}_{split} shape mismatch"
    assert y.shape[1] == 19


@pytest.mark.parametrize("tag", STRATEGIES)
def test_probabilities_are_in_range(tag):
    for model in ("svm", "dt", "ens"):
        for split in ("val", "test"):
            p = np.load(ASSETS / f"{model}_{tag}_proba_{split}.npy")
            assert np.isfinite(p).all(), f"{model}_{tag}_{split} non-finite"
            assert p.min() >= 0.0 and p.max() <= 1.0


@pytest.mark.parametrize("tag", STRATEGIES)
def test_ensemble_is_the_mean_of_its_members(tag):
    """The 'ens' matrix must really be the soft vote, not something else."""
    for split in ("val", "test"):
        svm = np.load(ASSETS / f"svm_{tag}_proba_{split}.npy")
        dt = np.load(ASSETS / f"dt_{tag}_proba_{split}.npy")
        ens = np.load(ASSETS / f"ens_{tag}_proba_{split}.npy")
        assert np.allclose(ens, (svm + dt) / 2.0, atol=1e-6)


def test_the_two_strategies_produced_different_models():
    a = np.load(ASSETS / f"ens_{STRATEGIES[0]}_proba_test.npy")
    b = np.load(ASSETS / f"ens_{STRATEGIES[1]}_proba_test.npy")
    assert not np.allclose(a, b), "both strategies gave identical probabilities"


def test_truth_is_binary():
    for split in ("val", "test"):
        y = np.load(ASSETS / f"y_{split}.npy")
        assert set(np.unique(y)).issubset({0, 1})


# --------------------------------------------------------------------------
# Metric correctness — checked against scikit-learn
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tag", STRATEGIES)
def test_micro_f1_matches_sklearn(tag):
    sk = pytest.importorskip("sklearn.metrics")
    import app as application

    y = np.load(ASSETS / "y_test.npy")
    proba = np.load(ASSETS / f"ens_{tag}_proba_test.npy")
    expected = sk.f1_score(y, (proba >= 0.5).astype(int), average="micro",
                           zero_division=0)
    assert application.at(tag, "test", 0.5)[2] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("tag", STRATEGIES)
def test_macro_f1_matches_sklearn(tag):
    sk = pytest.importorskip("sklearn.metrics")
    import app as application

    y = np.load(ASSETS / "y_test.npy")
    proba = np.load(ASSETS / f"ens_{tag}_proba_test.npy")
    expected = sk.f1_score(y, (proba >= 0.5).astype(int), average="macro",
                           zero_division=0)
    assert application.macro_at(tag, "test", 0.5) == pytest.approx(expected,
                                                                   abs=1e-9)


# --------------------------------------------------------------------------
# The claims the app makes
# --------------------------------------------------------------------------
def test_tuning_helps_on_at_least_one_strategy():
    """
    Deliberately not asserted per strategy. With class_weight='balanced' the
    weighting has already done most of this job, so the remaining gain is
    small and could plausibly vanish. The app says so in its own text.
    """
    import app as application

    truth = np.load(ASSETS / "y_test.npy").astype(bool)
    gains = {}
    for tag in STRATEGIES:
        test = np.load(ASSETS / f"ens_{tag}_proba_test.npy")
        chosen = application.tuned_thresholds(tag)
        base = application.micro_f1(test >= 0.5, truth)
        tuned = application.micro_f1(test >= chosen[None, :], truth)
        gains[tag] = tuned - base
    assert any(g > 0 for g in gains.values()), f"no strategy improved: {gains}"


def test_the_unweighted_model_wants_a_much_lower_cutoff():
    """
    The central claim of the 'Two strategies' tab: an unweighted model needs a
    cutoff well below 0.5, while the balanced model is already near-optimal at
    the default, because class_weight performed the same correction in advance.
    If this stops holding, that tab's explanation is wrong and must be rewritten.
    """
    import app as application

    optima = {}
    for tag in STRATEGIES:
        _, _, f = application.curves(tag, "test")["micro"]
        optima[tag] = float(application.GRID[int(f.argmax())])
    assert optima["none"] <= 0.40, (
        f"naive optimum should sit well below 0.5, got {optima}"
    )
    assert abs(optima["balanced"] - 0.50) <= 0.05, (
        f"balanced optimum should sit at ~0.5, got {optima}"
    )


def test_a_rare_label_is_dead_at_the_default_cutoff_when_unweighted():
    """The app's most dramatic claim, pinned so it cannot silently become false."""
    import app as application

    _, _, per_label = application.curves("none", "test")["per_label"]
    half = int(np.abs(application.GRID - 0.5).argmin())
    dead = [j for j in META["minority_labels"]
            if per_label[half, j] == 0 and per_label[:, j].max() > 0]
    assert dead, "no rare label scores zero at 0.5 — the app's claim is stale"


# --------------------------------------------------------------------------
# The app itself
# --------------------------------------------------------------------------
def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, f"app raised: {at.exception}"


def test_slider_changes_the_reported_score():
    at = AppTest.from_file(APP, default_timeout=90).run()
    before = at.metric[2].value           # F1 (micro)
    at.slider[0].set_value(0.15).run()
    assert not at.exception
    assert at.metric[2].value != before


def test_switching_strategy_changes_the_score():
    at = AppTest.from_file(APP, default_timeout=90).run()
    before = at.metric[2].value
    at.radio[0].set_value(STRATEGIES[0]).run()
    assert not at.exception
    assert at.metric[2].value != before
