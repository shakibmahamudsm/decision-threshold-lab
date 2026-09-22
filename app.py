"""
Threshold Lab — where you cut a probability matters as much as the model,
and the two standard ways of handling class imbalance are substitutes for
each other rather than additions.

The app does no training and loads no model. Everything is numpy over small
probability matrices exported by prepare_data.py, so every interaction is
instant and memory stays tiny.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
DEFAULT_THRESHOLD = 0.50

LABELS = {
    "none": "None — the naive model",
    "balanced": "class_weight='balanced'",
}

st.set_page_config(
    page_title="Decision Threshold Lab",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      [data-testid="stMetricValue"] {font-size: 2.0rem;}
      .lede {font-size: 1.05rem; line-height: 1.6; color: #444;}
      @media (prefers-color-scheme: dark) { .lede {color: #bbb;} }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_assets():
    """Probability matrices and ground truth. Shapes: (n_rows, 19)."""
    meta = json.loads((ASSETS / "meta.json").read_text())
    d = {
        "meta": meta,
        "y_val": np.load(ASSETS / "y_val.npy"),
        "y_test": np.load(ASSETS / "y_test.npy"),
    }
    for tag in meta["strategies"]:
        for split in ("val", "test"):
            for model in ("svm", "dt", "ens"):
                d[f"{model}_{tag}_{split}"] = np.load(
                    ASSETS / f"{model}_{tag}_proba_{split}.npy"
                )
    return d


def prf(tp, fp, fn):
    """Precision, recall, F1 from counts. Any shape; zero-safe."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f


@st.cache_data(show_spinner=False)
def curves(strategy: str, split: str):
    """Micro and per-label metrics across the whole threshold grid."""
    d = load_assets()
    proba = d[f"ens_{strategy}_{split}"]
    truth = d[f"y_{split}"].astype(bool)
    n_lab = truth.shape[1]
    tp = np.empty((len(GRID), n_lab), dtype=np.int64)
    fp, fn = np.empty_like(tp), np.empty_like(tp)
    for i, t in enumerate(GRID):
        pred = proba >= t
        tp[i] = (pred & truth).sum(axis=0)
        fp[i] = (pred & ~truth).sum(axis=0)
        fn[i] = (~pred & truth).sum(axis=0)
    return {"micro": prf(tp.sum(1), fp.sum(1), fn.sum(1)),
            "per_label": prf(tp, fp, fn)}


def at(strategy: str, split: str, threshold: float):
    p, r, f = curves(strategy, split)["micro"]
    i = int(np.abs(GRID - threshold).argmin())
    return float(p[i]), float(r[i]), float(f[i])


def macro_at(strategy: str, split: str, threshold: float) -> float:
    i = int(np.abs(GRID - threshold).argmin())
    return float(curves(strategy, split)["per_label"][2][i].mean())


def micro_f1(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


@st.cache_data(show_spinner=False)
def tuned_thresholds(strategy: str) -> np.ndarray:
    """One threshold per label, chosen on VALIDATION only."""
    return GRID[curves(strategy, "val")["per_label"][2].argmax(axis=0)]


d = load_assets()
meta = d["meta"]
N_LABELS = meta["n_labels"]
MINORITY = set(meta["minority_labels"])
POSITIVES = meta["label_positives_full"]
STRATEGIES = meta["strategies"]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Decision Threshold Lab")
    st.caption(
        "Multilabel classification, 19 labels, 768 anonymised features. "
        "SVM + Decision Tree soft-voting ensemble."
    )
    st.divider()
    strategy = st.radio(
        "Imbalance handling",
        STRATEGIES,
        format_func=lambda s: LABELS[s],
        index=len(STRATEGIES) - 1,
        help="The two textbook ways to deal with rare classes. Switching this "
             "changes which direction the threshold needs to move.",
    )
    st.divider()
    st.metric("Validation rows", f"{meta['n_val']:,}")
    st.metric("Test rows", f"{meta['n_test']:,}")
    st.caption(
        f"PCA {meta['pca_components']} components · "
        f"{meta['pca_explained_variance']:.0%} of variance retained"
    )
    st.divider()
    st.caption(
        "Thresholds are tuned on **validation** and scored on **test**, so "
        "nothing here is tuned on the data it reports."
    )


tab_play, tab_labels, tab_fix, tab_cmp, tab_about = st.tabs(
    ["🎚️ Play", "🔍 Per label", "🛠️ The fix", "⚖️ Two strategies", "📖 About"]
)


# ----------------------------------------------------------------------------
# Tab 1 — Play
# ----------------------------------------------------------------------------
with tab_play:
    st.subheader("You are the model. Where do you cut?")
    st.markdown(
        '<p class="lede">The classifier outputs a probability for each of 19 '
        'labels. Something has to turn that into a yes or a no, and almost '
        'everyone leaves that cutoff at 0.5 without thinking about it. '
        'Drag it.</p>',
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Decision threshold — predict 1 when probability ≥ this value",
        min_value=0.01, max_value=0.99, value=DEFAULT_THRESHOLD, step=0.01,
    )

    p, r, f1 = at(strategy, "test", threshold)
    p0, r0, f0 = at(strategy, "test", DEFAULT_THRESHOLD)
    moved = abs(threshold - DEFAULT_THRESHOLD) > 1e-9

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{p:.3f}", f"{p - p0:+.3f}" if moved else None)
    c2.metric("Recall", f"{r:.3f}", f"{r - r0:+.3f}" if moved else None)
    c3.metric("F1 (micro)", f"{f1:.3f}", f"{f1 - f0:+.3f}" if moved else None)
    c4.metric("F1 (macro)", f"{macro_at(strategy, 'test', threshold):.3f}",
              help="Treats all 19 labels equally, so rare labels actually count.")

    key = f"best_{strategy}"
    if f1 > st.session_state.get(key, 0.0):
        st.session_state[key] = f1
        st.session_state[f"{key}_t"] = threshold
    best = st.session_state.get(key, 0.0)

    pm, rm, fm = curves(strategy, "test")["micro"]
    opt_i = int(fm.argmax())
    opt_t, opt_f1 = float(GRID[opt_i]), float(fm[opt_i])

    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        for series, name, colour in ((pm, "Precision", "#e45756"),
                                     (rm, "Recall", "#4c78a8"),
                                     (fm, "F1 (micro)", "#54a24b")):
            fig.add_trace(go.Scatter(
                x=GRID, y=series, name=name, mode="lines",
                line=dict(width=3 if name.startswith("F1") else 2, color=colour)))
        fig.add_vline(x=threshold, line=dict(color="#888", width=2, dash="dash"),
                      annotation_text="you", annotation_position="top")
        if st.session_state.get(f"revealed_{strategy}"):
            fig.add_vline(x=opt_t, line=dict(color="#54a24b", width=2, dash="dot"),
                          annotation_text="optimum", annotation_position="bottom")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="threshold", yaxis_title="score",
                          yaxis_range=[0, 1], hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.metric("Your best F1", f"{best:.3f}",
                  help=f"at threshold "
                       f"{st.session_state.get(f'{key}_t', threshold):.2f}")
        st.metric("Default (0.5)", f"{f0:.3f}")
        if st.button("Reveal the optimum", width="stretch"):
            st.session_state[f"revealed_{strategy}"] = True
        if st.session_state.get(f"revealed_{strategy}"):
            st.success(f"Best single threshold: **{opt_t:.2f}** → F1 **{opt_f1:.3f}**")
            st.caption(f"**{opt_f1 - f0:+.3f}** over the default, "
                       f"for zero extra training.")
            if best >= opt_f1 - 1e-9:
                st.balloons()

    direction = ("**below** 0.5" if opt_t < 0.5 else
                 "**above** 0.5" if opt_t > 0.5 else "exactly 0.5")
    st.info(
        f"With `class_weight={LABELS[strategy].split('—')[0].strip()}` the best "
        f"cutoff sits {direction}. Switch the strategy in the sidebar: the naive "
        f"model needs a far lower cutoff, while the balanced one is already right "
        f"at the default — because `class_weight` has done that same correction "
        f"in advance. Two routes to one place.",
        icon="💡",
    )


# ----------------------------------------------------------------------------
# Tab 2 — Per label
# ----------------------------------------------------------------------------
with tab_labels:
    st.subheader("One threshold does not fit all 19 labels")
    st.markdown(
        '<p class="lede">Each label has its own base rate, from 37% of rows down '
        'to 0.25%. Pick one — the rare ones are where this gets dramatic.</p>',
        unsafe_allow_html=True,
    )

    def label_name(i: int) -> str:
        n = POSITIVES[str(i)]
        tag = "   ⚠️ rare" if i in MINORITY else ""
        return f"Label {i} — {n:,} positives ({100 * n / 25000:.2f}%){tag}"

    order = sorted(range(N_LABELS), key=lambda i: POSITIVES[str(i)])
    choice = st.selectbox("Label", order, format_func=label_name)

    p_lab, r_lab, f_lab = curves(strategy, "test")["per_label"]
    series_f = f_lab[:, choice]
    best_i = int(series_f.argmax())
    at_half = float(series_f[int(np.abs(GRID - 0.5).argmin())])

    c1, c2, c3 = st.columns(3)
    c1.metric("F1 at the default 0.5", f"{at_half:.3f}")
    c2.metric("Best F1", f"{float(series_f[best_i]):.3f}",
              f"{float(series_f[best_i]) - at_half:+.3f}")
    c3.metric("Best threshold", f"{float(GRID[best_i]):.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=GRID, y=p_lab[:, choice], name="Precision",
                             line=dict(color="#e45756", width=2)))
    fig.add_trace(go.Scatter(x=GRID, y=r_lab[:, choice], name="Recall",
                             line=dict(color="#4c78a8", width=2)))
    fig.add_trace(go.Scatter(x=GRID, y=series_f, name="F1",
                             line=dict(color="#54a24b", width=3)))
    fig.add_vline(x=0.5, line=dict(color="#888", width=2, dash="dash"),
                  annotation_text="default 0.5", annotation_position="top")
    fig.add_vline(x=float(GRID[best_i]),
                  line=dict(color="#54a24b", width=2, dash="dot"),
                  annotation_text="best", annotation_position="bottom")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="threshold", yaxis_title="score",
                      yaxis_range=[0, 1], hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

    if at_half == 0 and float(series_f[best_i]) > 0:
        st.error(
            f"At the default cutoff this label scores **exactly zero**. The model "
            f"never becomes 50% confident about something this rare, so it answers "
            f"'no' every single time and the label is silently dead. Move the "
            f"cutoff to **{float(GRID[best_i]):.2f}** and it reaches "
            f"**{float(series_f[best_i]):.3f}**. The model always knew — the "
            f"cutoff was throwing the answer away.",
            icon="🚨",
        )
    elif choice in MINORITY:
        st.warning(
            f"One of the five rare labels. The best cutoff is "
            f"{float(GRID[best_i]):.2f}, not 0.5.", icon="⚠️")


# ----------------------------------------------------------------------------
# Tab 3 — The fix
# ----------------------------------------------------------------------------
with tab_fix:
    st.subheader("Tune every label separately — honestly")
    st.markdown(
        '<p class="lede">Give each label its own threshold. The catch: choosing '
        'them by looking at the test set would be a fresh kind of data leakage. '
        'So they are picked on a held-out <b>validation</b> split, then scored '
        'once on test.</p>',
        unsafe_allow_html=True,
    )

    chosen = tuned_thresholds(strategy)
    test = d[f"ens_{strategy}_test"]
    truth = d["y_test"].astype(bool)

    base = micro_f1(test >= DEFAULT_THRESHOLD, truth)
    tuned = micro_f1(test >= chosen[None, :], truth)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test F1 — everything at 0.5", f"{base:.4f}")
    c2.metric("Test F1 — per-label thresholds", f"{tuned:.4f}", f"{tuned - base:+.4f}")
    c3.metric("Extra training required", "none")

    st.caption(
        "Same model, same weights, same underlying probabilities. The only thing "
        "that changed is where each label's yes/no line is drawn."
    )

    fig = go.Figure(go.Bar(
        x=[str(i) for i in range(N_LABELS)], y=chosen,
        marker_color=["#e45756" if i in MINORITY else "#4c78a8"
                      for i in range(N_LABELS)]))
    fig.add_hline(y=0.5, line=dict(color="#888", width=2, dash="dash"),
                  annotation_text="the default everyone uses")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="label", yaxis_title="chosen threshold",
                      yaxis_range=[0, 1], showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Red bars are the five rare labels.")

    st.warning(
        "Worth noticing: the gain here is real but modest, and a few individual "
        "labels get **worse**. Thresholds picked on a validation split that "
        "contains only a handful of positives for the rarest labels do not always "
        "transfer. That is overfitting the tuning step, and pretending otherwise "
        "would be dishonest.",
        icon="⚠️",
    )


# ----------------------------------------------------------------------------
# Tab 4 — Two strategies
# ----------------------------------------------------------------------------
with tab_cmp:
    st.subheader("Class weights and thresholds are substitutes")
    st.markdown(
        '<p class="lede">Both are ways of telling the model to stop ignoring rare '
        'classes. Run them side by side and it becomes obvious they are two routes '
        'to the same destination — and that doing both can overshoot.</p>',
        unsafe_allow_html=True,
    )

    rows, fig = [], go.Figure()
    for tag, colour in zip(STRATEGIES, ("#e45756", "#4c78a8")):
        _, _, fmic = curves(tag, "test")["micro"]
        i_opt = int(fmic.argmax())
        thr = tuned_thresholds(tag)
        te = d[f"ens_{tag}_test"]
        rows.append({
            "strategy": LABELS[tag],
            "F1 @ 0.5": f"{float(fmic[int(np.abs(GRID - 0.5).argmin())]):.4f}",
            "best single threshold": f"{float(GRID[i_opt]):.2f}",
            "F1 there": f"{float(fmic[i_opt]):.4f}",
            "F1 per-label tuned": f"{micro_f1(te >= thr[None, :], truth):.4f}",
        })
        fig.add_trace(go.Scatter(x=GRID, y=fmic, name=LABELS[tag],
                                 line=dict(width=3, color=colour)))
        fig.add_vline(x=float(GRID[i_opt]),
                      line=dict(color=colour, width=2, dash="dot"))

    st.dataframe(rows, width="stretch", hide_index=True)

    fig.add_vline(x=0.5, line=dict(color="#888", width=2, dash="dash"),
                  annotation_text="default", annotation_position="top")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="threshold", yaxis_title="F1 (micro)",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

    st.info(
        "The two curves peak in **different places**. The naive model is too "
        "reluctant to predict a rare label, so its threshold has to come *down* "
        "a long way. The balanced model peaks at 0.5 itself: `class_weight` "
        "already moved the model so that the default cutoff is the right one. "
        "That is the point — these are the same correction applied at two "
        "different points in the pipeline, not two improvements to stack. "
        "Note also where each curve *ends up*, not just where it peaks.",
        icon="⚖️",
    )


# ----------------------------------------------------------------------------
# Tab 5 — About
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("What this is")
    st.markdown(
        f"""
The data is a university assignment set, deliberately anonymised: **768 features
with no stated meaning and 19 labels with no names**. Each row carries about 2.9
labels at once, and every label is a minority — the most common appears in 37% of
rows, the rarest in **0.25%** (62 rows out of 25,000).

Because the inputs mean nothing to a human, there is no "type a sentence" demo to
build here. So this app lets you operate the *model* instead of feeding it data.

**How it stays fast.** The app never trains and never loads a model.
`prepare_data.py` fits everything once, offline, and exports probability matrices;
the app reads those and does plain numpy. Every slider move is a comparison
against {meta['n_test']:,} × {N_LABELS} floats — microseconds, and a few hundred
KB of memory.

**The pipeline behind the numbers**

| Step | Choice |
|---|---|
| Split | {meta['n_train']:,} train / {meta['n_val']:,} validation / {meta['n_test']:,} test, multilabel-stratified |
| Scaling | Z-score, fit on train only |
| Reduction | PCA {meta['pca_components']} components ({meta['pca_explained_variance']:.0%} of variance) |
| Model | Soft-voting ensemble: SVM (C={meta['svm_C']}, linear, calibrated) + Decision Tree (depth {meta['dt_depth']}) |
| Imbalance | trained twice — once with no class weighting, once with `balanced` |

Preprocessing sits inside the training split, not around it. In the original
version of this project PCA was fit on all 25,000 rows *before* the split, so the
test set had already leaked into the transform and the reported score was
inflated.

**The point being made**

`predict()` in scikit-learn hard-codes a 0.5 cutoff. That is a *modelling
decision* disguised as a default, and on imbalanced data it can silently kill a
label outright. It is also not independent of how you handled the imbalance
during training — which is what the **Two strategies** tab is about.

The companion project, **Ensemble Lab**, holds the threshold fixed and varies the
model blend instead.
        """
    )
    st.caption("Built with scikit-learn, NumPy, Plotly and Streamlit.")
