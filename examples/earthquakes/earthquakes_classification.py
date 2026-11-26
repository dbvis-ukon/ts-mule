"""Time Series Classification Example on the UCR Earthquakes Dataset with TS-MULE.

This script:
1. Downloads and caches the UCR Earthquakes archive under ``examples/earthquakes/data/``.
2. Trains a balanced Random Forest classifier directly on the raw 512-step seismic series.
3. Explains test predictions using ``LimeTS`` across all 7 TS-MULE segmentation strategies.
4. Evaluates explanation faithfulness via ``PerturbationAnalysis`` using both
   ROC-AUC drop and Brier score increase (high-relevance vs random perturbations).
5. Visualizes attributions across all segmentation methods and perturbation evaluation scores.
"""
import argparse
import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from tsmule.sampling.segment import (
    MatrixProfileSegmentation,
    SAXSegmentation,
    WindowSegmentation,
)
from tsmule.xai.evaluation import PerturbationAnalysis
from tsmule.xai.lime import LimeTS

EARTHQUAKES_ZIP_URL = 'https://www.timeseriesclassification.com/aeon-toolkit/Earthquakes.zip'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
EARTHQUAKES_METHODS = (
    'uniform',
    'exponential',
    'slopes-sorted',
    'slopes-not-sorted',
    'bins-min',
    'bins-max',
    'sax',
)


def _parse_ts_file(filepath):
    """Parse a univariate UCR ``.ts`` file into (X, y) float64/int numpy arrays."""
    rows = []
    labels = []
    in_data = False
    with open(filepath, 'r', encoding='utf-8') as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if not in_data:
                if line.lower() == '@data':
                    in_data = True
                continue
            series_str, label_str = line.rsplit(':', 1)
            rows.append([float(v) for v in series_str.split(',')])
            labels.append(int(float(label_str)))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=int)


def load_earthquakes(data_dir=DATA_DIR):
    """Download (if needed) and load the UCR Earthquakes train and test splits."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, 'Earthquakes_TRAIN.ts')
    test_path = os.path.join(data_dir, 'Earthquakes_TEST.ts')

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        zip_path = os.path.join(data_dir, 'Earthquakes.zip')
        if not os.path.exists(zip_path):
            print(f'Downloading {EARTHQUAKES_ZIP_URL} -> {zip_path}')
            urllib.request.urlretrieve(EARTHQUAKES_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                base = os.path.basename(member)
                if base in ('Earthquakes_TRAIN.ts', 'Earthquakes_TEST.ts'):
                    target = os.path.join(data_dir, base)
                    with zf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())

    X_train, y_train = _parse_ts_file(train_path)
    X_test, y_test = _parse_ts_file(test_path)
    return X_train, y_train, X_test, y_test


def train_earthquakes_classifier(X_train, y_train, seed=42):
    """Train a balanced Random Forest classifier on raw Earthquakes series."""
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=seed,
    )
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
        ('auc', 'ROC-AUC (higher is better; valid: orig >= rand >= pert)', (0.0, 1.05)),
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

    fig.suptitle(
        'Earthquakes Perturbation Faithfulness Across All TS-MULE Methods',
        fontsize=12,
        fontweight='bold',
    )
    return fig


def main():
    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description='Explain an Earthquakes time series classifier with TS-MULE.')
    parser.add_argument('--n-explain', type=int, default=60, help='Number of test instances to explain and evaluate.')
    parser.add_argument('--n-samples', type=int, default=200, help='Perturbation samples per explanation.')
    parser.add_argument('--partitions', type=int, default=50, help='Target number of segments.')
    parser.add_argument('--win-length', type=int, default=10, help='Window length for window/MP segmenters.')
    parser.add_argument('--percentile', type=float, default=80.0, help='Relevance percentile threshold for perturbation evaluation.')
    parser.add_argument('--replace-method', type=str, default='zeros', help='Perturbation replacement strategy.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save-plot', type=str, default='', help='Optional path to save multi-method relevance plot PNG.')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_earthquakes()
    print(f'Earthquakes loaded: train={X_train.shape}, test={X_test.shape}')

    model = train_earthquakes_classifier(X_train, y_train, seed=args.seed)
    probs_test = model.predict_proba(X_test)[:, 1]
    print(
        f'Classifier test acc={accuracy_score(y_test, model.predict(X_test)):.4f}, '
        f'ROC-AUC={roc_auc_score(y_test, probs_test):.4f}, '
        f'Brier={brier_score_loss(y_test, probs_test):.4f}'
    )

    rng = np.random.default_rng(args.seed)
    n_eval = min(args.n_explain, len(X_test))
    eval_idx = rng.permutation(len(X_test))[:n_eval]
    X_eval = X_test[eval_idx]
    y_eval = y_test[eval_idx]

    pos_indices = np.where(y_eval == 1)[0]
    vis_pos = int(pos_indices[0]) if len(pos_indices) > 0 else 0

    vis_relevances = {}
    vis_r2 = {}
    vis_seg_counts = {}

    for method in EARTHQUAKES_METHODS:
        segmenter, seg_arg = build_segmenter(
            method, partitions=args.partitions, win_length=args.win_length
        )
        seg_mask = (
            segmenter.segment(X_eval[vis_pos])
            if seg_arg is None
            else segmenter.segment(X_eval[vis_pos], segmentation_method=seg_arg)
        )
        vis_seg_counts[method] = int(len(np.unique(seg_mask)))

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
        vis_relevances[method] = relevances[vis_pos]
        vis_r2[method] = r2_scores[vis_pos]

        pa_auc = PerturbationAnalysis()
        auc_ins = pa_auc.analysis_relevance(
            X_eval,
            y_eval,
            relevances,
            predict_fn=lambda batch: model.predict_proba(batch)[:, 1],
            eval_fn=roc_auc_score,
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
            f"auc(o/r/p)={auc_ins['original']:.3f}/{auc_ins['random']:.3f}/{auc_ins['percentile']:.3f} "
            f"(valid={pa_auc.is_valid(higher_is_better=True)}, score={pa_auc.perturbation_score():.2f}) | "
            f"brier(o/r/p)={brier_ins['original']:.3f}/{brier_ins['random']:.3f}/{brier_ins['percentile']:.3f} "
            f"(valid={pa_brier.is_valid(higher_is_better=False)}, score={pa_brier.perturbation_score():.2f})"
        )

    if args.save_plot:
        fig = plot_methods_relevance(
            X_eval[vis_pos],
            vis_relevances,
            r2_by_method=vis_r2,
            seg_counts=vis_seg_counts,
            title=f'Earthquakes Sample {vis_pos} Relevance Across All 7 TS-MULE Methods (label={y_eval[vis_pos]})',
        )
        fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved relevance visualization to {args.save_plot}')


if __name__ == '__main__':
    main()
