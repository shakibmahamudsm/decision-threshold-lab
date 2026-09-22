# Decision Threshold Lab

**Two ways to fix an imbalanced classifier. An interactive demo of why you only need one.**

[![tests](https://github.com/USERNAME/decision-threshold-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/decision-threshold-lab/actions/workflows/tests.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](STREAMLIT_URL_HERE)

> 🔗 **[Live demo](STREAMLIT_URL_HERE)** · Companion project: **[Soft Voting Mixer](https://github.com/USERNAME/soft-voting-mixer)**

---

## The one-line version

`scikit-learn`'s `predict()` hard-codes a 0.5 probability cutoff. On imbalanced
data that silently discards most of what the model knows — **unless you already
corrected the imbalance somewhere else.** This app trains the same model twice,
with and without `class_weight='balanced'`, and lets you drag the cutoff on
either one.

**Headline result: on the unweighted model, moving the cutoff from 0.50 to 0.33
lifts micro-F1 from 0.652 to 0.704. On the `class_weight='balanced'` model the
best cutoff *is* 0.50, and tuning buys +0.002 — nothing.**

Same correction, two places in the pipeline. Doing both is not additive.

---

## Why this exists

The dataset is a university assignment set, deliberately anonymised — **768
features with no stated meaning, 19 labels with no names**. Each row carries
about 2.9 labels at once, and every single label is a minority class:

| | Rarest label | Most common label |
|---|---|---|
| Positives out of 25,000 | **62** (0.25%) | 9,289 (37.2%) |

Because nothing in the input means anything to a human, there is no "type a
sentence and see the prediction" demo to build here. So the app makes the
*model* the thing you operate, rather than the data.

---

## The numbers

Ensemble of SVM + decision tree, scored on the held-out test split (3,766 rows).
Thresholds are chosen on **validation** and scored once on **test**.

| | `class_weight=None` | `class_weight='balanced'` |
|---|---|---|
| micro-F1 @ 0.50 | 0.6516 | **0.6964** |
| Best global threshold (chosen on val) | **0.33** | **0.50** |
| micro-F1 at that threshold | **0.7042** | 0.6964 |
| Gain from tuning | **+0.0526** | +0.0020 |
| macro-F1 @ 0.50 | 0.4733 | **0.5895** |

Three things worth reading off that table:

1. **Untouched, class weighting wins.** At the default cutoff, the balanced model
   is far ahead (0.696 vs 0.652) — which is exactly why it is the usual advice.
2. **Tuned, the plain model edges ahead.** 0.7042 vs 0.6964. The two approaches
   land in the same place; neither is magic.
3. **The balanced model's optimum is the default.** `class_weight` has already
   moved the decision boundary so that 0.5 is correct. There is nothing left for
   a threshold to recover.

### The rare labels

Per-label F1 on test, unweighted model, default cutoff vs its own best cutoff:

| Label | Test positives | F1 @ 0.50 | F1 tuned | At threshold |
|---|---|---|---|---|
| 14 | 9 | **0.000** | **0.417** | 0.05 |
| 1 | 82 | 0.257 | 0.476 | 0.08 |
| 11 | 92 | 0.175 | 0.419 | 0.14 |
| 12 | 106 | 0.303 | 0.585 | 0.13 |
| 16 | 12 | 0.000 | 0.000 | — |

**Label 14 is the one to look at.** At the default cutoff its F1 is exactly
zero: the model never reaches 50% confidence on something that rare, so it
answers "no" every time and the label is effectively deleted from the output.
Drop the cutoff to 0.05 and it scores 0.417. The model knew; the cutoff was
throwing the answer away.

**Label 16 is the honest counterexample.** It stays at 0.000 no matter where the
cutoff goes. With 12 positives in the test split, thresholding cannot recover
what the model never learned. Not every problem is a threshold problem.

---

## What you can do in it

| Tab | What happens |
|---|---|
| **🎚️ Play** | One slider. Precision, recall and F1 move live. Try to beat the default, then reveal the optimum. |
| **🔍 Per label** | Pick any of the 19 labels and see its own precision/recall/F1 curve. The rare ones are dramatic. |
| **🛠️ The fix** | Tune all 19 thresholds on validation, score once on test, see the honest gain. |
| **⚖️ Two strategies** | Both F1-vs-threshold curves overlaid. Where each peaks is the whole argument. |
| **📖 About** | The pipeline, the numbers, and the leakage bug this rebuild fixed. |

A sidebar radio switches which trained model everything is measured against.

---

## How it stays fast

The app **never trains and never loads a model.**

`prepare_data.py` fits everything once, offline, and writes the predicted
probabilities to `assets/*.npy`. The app reads those arrays and does plain
numpy. Every slider move is a comparison against a few hundred KB of floats —
microseconds.

This has three consequences that matter for a free-tier deploy:

- **No cold-start training.** The page is interactive immediately.
- **No `scikit-learn` in `requirements.txt`.** The app has no model to
  deserialise, so there is no pickle/library version mismatch to go wrong.
- **Tiny memory.** Nothing close to the 1 GB limit.

---

## Run it locally

```bash
git clone https://github.com/USERNAME/decision-threshold-lab.git
cd decision-threshold-lab
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

That is all — the committed artifacts mean it works straight after a clone,
with no dataset download and no training.

### Running the tests

```bash
pip install -r requirements-dev.txt pytest
pytest -q
```

21 tests. They check the artifacts are sane, verify the metric code against
`sklearn.metrics`, **pin the app's factual claims** (that the unweighted optimum
sits well below 0.5, that the balanced one sits at ~0.5, and that a rare label
really does score zero at the default), and run the whole app headlessly through
Streamlit's `AppTest` harness.

Those pinned tests exist because an earlier draft of this project asserted
something the data did not support. If the claims ever stop holding, the suite
fails rather than the README quietly lying.

### Regenerating the artifacts

Needs the full dataset (see [`data/README.md`](data/README.md)). Expect **~3
hours** — it fits calibrated SVMs across 19 labels, twice.

```bash
pip install -r requirements-dev.txt
python prepare_data.py --data-dir "path/to/Data"
```

---

## The pipeline behind the numbers

```
25,000 × 768
      │
      ├── multilabel-stratified split ──► 17,490 train / 3,744 val / 3,766 test
      │
      ▼   (fit on TRAIN ONLY)
  Z-score normalisation
      │
      ▼
  PCA → 100 components (91.8% of variance)
      │
      ▼
  For each of 19 labels, and for each of the two strategies:
      soft vote( SVM(C=0.1, linear, calibrated) , DecisionTree(depth 3) )
      │
      ▼
  probability per row per label  ──►  assets/*.npy  ──►  this app
```

Preprocessing sits **inside** the training split rather than around it.

That is a bug fix. In the original version of this project, PCA and scaling were
fit on all 25,000 rows *before* the train/test split, so the test set had
already leaked into the transform. The reported score was inflated as a result.
Everything here is fit on training data only.

`SVC(probability=True)` is deprecated in scikit-learn 1.9, so probabilities come
from `CalibratedClassifierCV(..., ensemble=False)` — which also calibrates them
better, and that matters when they are about to be averaged with a tree's.

---

## Honest limitations

- **Thresholds are tuned on validation, not test.** Tuning on the set you then
  report would be a fresh form of leakage. The gain in **The fix** is measured
  once, on data the thresholds never saw.
- **Per-label tuning slightly *underperforms* one global threshold here** — 0.7027
  vs 0.7042 micro-F1. Nineteen thresholds fitted on 3,744 validation rows is
  enough parameters to start overfitting the validation set. More tuning is not
  automatically better.
- **Tuning thresholds does not make the model smarter.** It recovers information
  the default cutoff discarded. Those are different things.
- **Thresholding cannot fix every rare label.** Label 16 scores zero at every
  cutoff.
- **Micro-F1 is dominated by the common labels.** That is why the app shows
  macro-F1 beside it and breaks results out per label.
- **The features are anonymous.** No feature-importance or SHAP story is possible
  here, because there is nothing for an importance score to refer to.

---

## Licence

MIT — see [LICENSE](LICENSE).
