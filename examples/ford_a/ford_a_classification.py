"""Time Series Classification Example on the UCR FordA Dataset with TS-MULE.

This script:
1. Downloads and caches the UCR FordA train/test TSV files under ``examples/ford_a/data/``.
2. Trains a scikit-learn RBF-SVM classifier directly on the raw 500-step time series.
3. Explains test predictions using ``LimeTS`` across all 7 TS-MULE segmentation strategies.
4. Evaluates explanation faithfulness via ``PerturbationAnalysis`` using both
   accuracy drop and Brier score increase (high-relevance vs random perturbations).
5. Visualizes attributions across all segmentation methods and perturbation evaluation scores.
"""
import argparse
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.svm import SVC

from tsmule.sampling.segment import (
    MatrixProfileSegmentation,
    SAXSegmentation,
    WindowSegmentation,
)
from tsmule.xai.evaluation import PerturbationAnalysis
from tsmule.xai.lime import LimeTS

FORDA_BASE_URL = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FORDA_METHODS = (
    'uniform',
    'exponential',
    'slopes-sorted',
    'slopes-not-sorted',
    'bins-min',
    'bins-max',
    'sax',
)


def load_forda(data_dir=DATA_DIR):
    """Download (if needed) and load the UCR FordA train and test splits."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for split in ('TRAIN', 'TEST'):
        fname = f'FordA_{split}.tsv'
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            url = f'{FORDA_BASE_URL}/{fname}'
            print(f'Downloading {url} -> {fpath}')
            urllib.request.urlretrieve(url, fpath)
        paths[split] = fpath

    train_arr = np.loadtxt(paths['TRAIN'], delimiter='\t')
    test_arr = np.loadtxt(paths['TEST'], delimiter='\t')

    X_train = train_arr[:, 1:].astype(np.float64)
    y_train = (train_arr[:, 0] > 0).astype(int)
    X_test = test_arr[:, 1:].astype(np.float64)
    y_test = (test_arr[:, 0] > 0).astype(int)
    return X_train, y_train, X_test, y_test


def train_forda_classifier(X_train, y_train, seed=42):
    """Train an RBF-SVM classifier with calibrated probabilities on raw FordA series."""
    clf = SVC(C=2.0, kernel='rbf', probability=True, random_state=seed)
    clf.fit(X_train, y_train)
    return clf


def build_segmenter(method_name, partitions=50, win_length=10):
    """Instantiate the TS-MULE segmenter and method key for a named strategy."""
    if method_name in ('uniform', 'exponential'):
        return WindowSegmentation(partitions=partitions, win_length=win_length), method_name
    if method_name == 'sax':
        return SAXSegmentation(partitions=partitions), None
    return MatrixProfileSegmentation(partitions=partitions, win_length=win_length), method_name


def plot_methods_relevance(
    sample,
    relevances_by_method,
    r2_by_method=None,
    seg_counts=None,
    title=None,
):
    """Plot a univariate time series overlaid on relevance heatmaps for all methods."""
    ts = np.asarray(sample, dtype=float).ravel()
    n_steps = len(ts)
    method_names = list(relevances_by_method.keys())
    n_methods = len(method_names)

    vmax = max(
        float(np.max(np.abs(np.asarray(rel))))
        for rel in relevances_by_method.values()
    ) or 1.0

    ymin, ymax = float(np.min(ts)), float(np.max(ts))
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0

    fig, axes = plt.subplots(
        n_methods,
        1,
        figsize=(12, max(2.2 * n_methods, 4)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    x_idx = np.arange(n_steps)
    last_im = None
    for row, method in enumerate(method_names):
        ax = axes[row, 0]
        rel = np.asarray(relevances_by_method[method], dtype=float).ravel()
        last_im = ax.imshow(
            rel[np.newaxis, :],
            aspect='auto',
            cmap='coolwarm',
            vmin=-vmax,
            vmax=vmax,
            alpha=0.55,
            extent=[0, n_steps - 1, ymin - pad, ymax + pad],
        )
        ax.plot(x_idx, ts, color='black', linewidth=1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_ylabel('Value')

        parts = [method]
        if seg_counts and method in seg_counts:
            parts.append(f'segments={seg_counts[method]}')
        if r2_by_method and method in r2_by_method:
            parts.append(f'R2={r2_by_method[method]:.3f}')
        ax.set_title(' | '.join(parts), loc='left', fontsize=10, fontweight='bold')

    axes[-1, 0].set_xlabel('Time step')
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            location='right',
            shrink=0.6,
            pad=0.02,
            label='Attribution Relevance',
        )
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    return fig


def plot_perturbation_comparison(results_by_method):
    """Plot grouped bar charts comparing original, random, and percentile scores across methods."""
    method_names = list(results_by_method.keys())
    x = np.arange(len(method_names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    panels = [
        ('acc', 'Accuracy (higher is better; valid: orig >= rand >= pert)', (0.0, 1.05)),
        ('brier', 'Brier Score (lower is better; valid: orig <= rand <= pert)', None),
    ]

    for ax, (metric_key, subtitle, ylim) in zip(axes, panels):
        orig_vals = [results_by_method[m][metric_key]['original'] for m in method_names]
        rand_vals = [results_by_method[m][metric_key]['random'] for m in method_names]
        pert_vals = [results_by_method[m][metric_key]['percentile'] for m in method_names]

        ax.bar(x - width, orig_vals, width, label='original', color='tab:blue')
        ax.bar(x, rand_vals, width, label='random', color='tab:orange')
        ax.bar(x + width, pert_vals, width, label='percentile', color='tab:red')

        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=30, ha='right')
        ax.set_title(subtitle, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()

    fig.suptitle('FordA Perturbation Faithfulness Across All TS-MULE Methods', fontsize=12, fontweight='bold')
    return fig


def main():
    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description='Explain a FordA time series classifier with TS-MULE.')
    parser.add_argument('--n-explain', type=int, default=40, help='Number of test instances to explain and evaluate.')
    parser.add_argument('--n-samples', type=int, default=200, help='Perturbation samples per explanation.')
    parser.add_argument('--partitions', type=int, default=50, help='Target number of segments.')
    parser.add_argument('--win-length', type=int, default=10, help='Window length for window/MP segmenters.')
    parser.add_argument('--percentile', type=float, default=80.0, help='Relevance percentile threshold for perturbation evaluation.')
    parser.add_argument('--replace-method', type=str, default='zeros', help='Perturbation replacement strategy.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save-plot', type=str, default='', help='Optional path to save multi-method relevance plot PNG.')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_forda()
    print(f'FordA loaded: train={X_train.shape}, test={X_test.shape}')

    model = train_forda_classifier(X_train, y_train, seed=args.seed)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'Classifier test accuracy: {test_acc:.4f}')

    n_eval = min(args.n_explain, len(X_test))
    X_eval = X_test[:n_eval]
    y_eval = y_test[:n_eval]

    first_relevances = {}
    first_r2 = {}
    first_seg_counts = {}
    for method in FORDA_METHODS:
        segmenter, seg_arg = build_segmenter(
            method, partitions=args.partitions, win_length=args.win_length
        )
        seg_mask0 = (
            segmenter.segment(X_eval[0])
            if seg_arg is None
            else segmenter.segment(X_eval[0], segmentation_method=seg_arg)
        )
        first_seg_counts[method] = int(len(np.unique(seg_mask0)))

        explainer = LimeTS(
            kernel=Ridge(alpha=1.0),
            segmenter=segmenter,
            replace_method=args.replace_method,
            n_samples=args.n_samples,
            random_state=args.seed,
        )

        relevances = []
        r2_scores = []
        for x in X_eval:
            rel = explainer.explain(
                x,
                model.predict_proba,
                segmentation_method=seg_arg,
                batch_predict=True,
            )
            relevances.append(rel)
            r2_scores.append(explainer.score)

        relevances = np.asarray(relevances)
        first_relevances[method] = relevances[0]
        first_r2[method] = r2_scores[0]

        pa_acc = PerturbationAnalysis()
        acc_ins = pa_acc.analysis_relevance(
            X_eval,
            y_eval,
            relevances,
            predict_fn=model.predict,
            eval_fn=accuracy_score,
            replace_method=args.replace_method,
            percentile=args.percentile,
            use_abs=True,
            random_state=args.seed,
        )

        pa_brier = PerturbationAnalysis()
        brier_ins = pa_brier.analysis_relevance(
            X_eval,
            y_eval,
            relevances,
            predict_fn=lambda batch: model.predict_proba(batch)[:, 1],
            eval_fn=brier_score_loss,
            replace_method=args.replace_method,
            percentile=args.percentile,
            use_abs=True,
            random_state=args.seed,
        )
        print(
            f'[{method:17s}] R2={np.mean(r2_scores):.3f} | '
            f"acc(o/r/p)={acc_ins['original']:.3f}/{acc_ins['random']:.3f}/{acc_ins['percentile']:.3f} "
            f"(valid={pa_acc.is_valid(higher_is_better=True)}, score={pa_acc.perturbation_score():.2f}) | "
            f"brier(o/r/p)={brier_ins['original']:.3f}/{brier_ins['random']:.3f}/{brier_ins['percentile']:.3f} "
            f"(valid={pa_brier.is_valid(higher_is_better=False)}, score={pa_brier.perturbation_score():.2f})"
        )

    if args.save_plot:
        fig = plot_methods_relevance(
            X_eval[0],
            first_relevances,
            r2_by_method=first_r2,
            seg_counts=first_seg_counts,
            title=f'FordA Sample 0 Relevance Across All 7 TS-MULE Methods (label={y_eval[0]})',
        )
        fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved relevance visualization to {args.save_plot}')


if __name__ == '__main__':
    main()
