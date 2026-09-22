"""
Train the models once and export the artifacts the Streamlit app consumes.

Two imbalance strategies are trained side by side:

  * class_weight=None       — the naive model, which learns to say "no"
  * class_weight='balanced' — the usual textbook correction

The app compares them, because the decision threshold behaves in opposite
directions under the two, and that contrast is the point being made.

The app itself never trains and never loads a model. It reads the probability
matrices written here and does plain numpy on them, which keeps every
interaction instant and memory far below the Streamlit Cloud limit.

    python prepare_data.py --data-dir "path/to/Data"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
N_COMPONENTS = 100
SVM_C = 0.1
DT_DEPTH = 3
# Key is the filename tag; value is the sklearn class_weight argument.
STRATEGIES = {"none": None, "balanced": "balanced"}
ASSETS = Path(__file__).parent / "assets"


def multilabel_split(X, y, test_size, random_state):
    """One stratified multilabel split. Returns (idx_a, idx_b)."""
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    return next(splitter.split(X, y))


def stacked_proba(model, X):
    """
    MultiOutputClassifier.predict_proba returns a list of (n, 2) arrays, one per
    label. Take P(class=1) from each and stack to (n, n_labels).

    A label with only one class present in training yields a (n, 1) array; use
    that single class's value so the output shape stays consistent.
    """
    out = []
    for probs, est in zip(model.predict_proba(X), model.estimators_):
        if probs.shape[1] == 2:
            out.append(probs[:, 1])
        else:
            out.append(np.full(probs.shape[0], 1.0 if est.classes_[0] == 1 else 0.0))
    return np.column_stack(out).astype(np.float32)


def main(data_dir: Path) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("Loading full dataset...")
    X = pd.read_csv(data_dir / "R2_train.csv").values.astype(np.float32)
    y = pd.read_csv(data_dir / "labels_train.csv").values.astype(np.int8)
    print(f"  X={X.shape}  y={y.shape}")

    # 70 / 15 / 15. Thresholds are tuned on validation and scored on test, so
    # the app can demonstrate tuning without leaking the set it reports.
    rest_idx, test_idx = multilabel_split(X, y, 0.15, RANDOM_STATE)
    X_rest, y_rest = X[rest_idx], y[rest_idx]
    tr_idx, val_idx = multilabel_split(X_rest, y_rest, 0.1765, RANDOM_STATE)

    X_train, y_train = X_rest[tr_idx], y_rest[tr_idx]
    X_val, y_val = X_rest[val_idx], y_rest[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    # Preprocessing is fit on TRAIN ONLY. This is the leakage fix — the original
    # version of this project fit PCA on all 25,000 rows before splitting.
    print("Fitting preprocessing on train only...")
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(scaler.transform(X_train))
    variance = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA {N_COMPONENTS} components explain {variance:.1%} of variance")

    def prep(A):
        return pca.transform(scaler.transform(A))

    Xtr, Xva, Xte = prep(X_train), prep(X_val), prep(X_test)

    for tag, weight in STRATEGIES.items():
        print(f"\n=== strategy: class_weight={weight!r} ===")

        # SVC(probability=True) is deprecated in sklearn 1.9.
        # CalibratedClassifierCV(..., ensemble=False) is the documented
        # replacement and yields better-calibrated probabilities, which matters
        # because they get averaged with a tree's.
        print("  Fitting SVM across 19 labels — this takes several minutes...")
        svm = MultiOutputClassifier(
            CalibratedClassifierCV(
                SVC(C=SVM_C, kernel="linear", class_weight=weight,
                    random_state=RANDOM_STATE),
                ensemble=False,
            )
        ).fit(Xtr, y_train)

        print("  Fitting Decision Tree...")
        dt = MultiOutputClassifier(
            DecisionTreeClassifier(max_depth=DT_DEPTH, class_weight=weight,
                                   random_state=RANDOM_STATE)
        ).fit(Xtr, y_train)

        for split, Xs in (("val", Xva), ("test", Xte)):
            svm_p = stacked_proba(svm, Xs)
            dt_p = stacked_proba(dt, Xs)
            np.save(ASSETS / f"svm_{tag}_proba_{split}.npy", svm_p)
            np.save(ASSETS / f"dt_{tag}_proba_{split}.npy", dt_p)
            # The deployed model is the soft vote of the two.
            np.save(ASSETS / f"ens_{tag}_proba_{split}.npy",
                    ((svm_p + dt_p) / 2.0).astype(np.float32))
        print(f"  {tag} done")

    np.save(ASSETS / "y_val.npy", y_val.astype(np.int8))
    np.save(ASSETS / "y_test.npy", y_test.astype(np.int8))

    meta = {
        "n_labels": int(y.shape[1]),
        "n_features_raw": int(X.shape[1]),
        "pca_components": N_COMPONENTS,
        "pca_explained_variance": round(variance, 4),
        "svm_C": SVM_C,
        "dt_depth": DT_DEPTH,
        "strategies": list(STRATEGIES),
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "label_positives_full": {str(i): int(y[:, i].sum())
                                 for i in range(y.shape[1])},
        "minority_labels": [1, 11, 12, 14, 16],
    }
    (ASSETS / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nArtifacts written to {ASSETS}")
    for f in sorted(ASSETS.iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True,
                    help="Folder containing R2_train.csv and labels_train.csv")
    main(Path(ap.parse_args().data_dir))
