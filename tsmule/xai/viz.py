"""Visualization supporting tools."""
import matplotlib.pyplot as plt
import numpy as np


def _ensure_2d_pair(arr1, arr2):
    """Ensure two matching 1D or 2D arrays are shaped as (n_steps, n_features)."""
    a1 = np.asarray(arr1, dtype=float)
    a2 = np.asarray(arr2)
    if a1.ndim == 1:
        a1 = a1.reshape(-1, 1)
    if a2.ndim == 1:
        a2 = a2.reshape(-1, 1)
    if a1.shape != a2.shape:
        raise ValueError(f'Shape mismatch: {a1.shape} vs {a2.shape}')
    return a1, a2


def visualize_segmentation_mask(time_series_sample, segmentation_mask, show=True):
    """Visualize a time series and its segmentation boundaries and labels.

    Args:
        time_series_sample (ndarray): Time series with shape ``(n_steps,)`` or
            ``(n_steps, n_features)``.
        segmentation_mask (ndarray): Segmentation mask matching ``time_series_sample``.
        show (bool, optional): Whether to call ``plt.show()``. Defaults to True.

    Returns:
        tuple: ``(fig_lines, fig_scatter)`` matplotlib Figure objects.
    """
    ts, seg = _ensure_2d_pair(time_series_sample, segmentation_mask)
    n_steps, n_features = ts.shape

    fig_lines, axes = plt.subplots(n_features, 1, squeeze=False)
    for i in range(n_features):
        ax = axes[i, 0]
        ax.plot(ts[:, i])
        old_value = seg[0, i]
        for j in range(n_steps):
            value = seg[j, i]
            if old_value != value:
                ax.axvline(x=j, color='red')
                old_value = value
    if show:
        plt.show()

    fig_scatter, axes_s = plt.subplots(n_features, 1, squeeze=False)
    for i in range(n_features):
        axes_s[i, 0].scatter(range(n_steps), ts[:, i], c=seg[:, i])
    if show:
        plt.show()

    return fig_lines, fig_scatter


def visualize_perturbation_masks(
    time_series_sample,
    perturbed_sample,
    mask=None,
    show=True,
):
    """Visualize original vs perturbed time series and optionally highlight replaced regions.

    Args:
        time_series_sample (ndarray): Original time series ``(n_steps,)`` or ``(n_steps, n_features)``.
        perturbed_sample (ndarray): Perturbed time series with the same shape.
        mask (ndarray, optional): Binary mask (0 = replaced, 1 = kept) with the same shape.
        show (bool, optional): Whether to call ``plt.show()``. Defaults to True.

    Returns:
        matplotlib.figure.Figure: Figure containing the comparison subplots.
    """
    orig, pert = _ensure_2d_pair(time_series_sample, perturbed_sample)
    n_steps, n_features = orig.shape
    mask_2d = None
    if mask is not None:
        _, mask_2d = _ensure_2d_pair(time_series_sample, mask)

    fig, axes = plt.subplots(n_features, 1, squeeze=False, sharex=True)
    x_idx = np.arange(n_steps)
    for i in range(n_features):
        ax = axes[i, 0]
        ax.plot(x_idx, orig[:, i], label='original', color='tab:blue')
        ax.plot(x_idx, pert[:, i], label='perturbed', color='tab:orange', linestyle='--')
        if mask_2d is not None:
            off_points = mask_2d[:, i] == 0
            if np.any(off_points):
                ymin, ymax = ax.get_ylim()
                ax.fill_between(
                    x_idx,
                    ymin,
                    ymax,
                    where=off_points,
                    color='tab:red',
                    alpha=0.18,
                    label='replaced',
                )
        ax.set_ylabel(f'Feature {i}')
        if i == 0:
            ax.legend(loc='upper right')

    axes[-1, 0].set_xlabel('Time step')
    if show:
        plt.show()
    return fig


def visualize_relevance(time_series_sample, relevance, title=None, show=True):
    """Visualize a time series overlaid on its attribution relevance heatmap.

    Args:
        time_series_sample (ndarray): Time series ``(n_steps,)`` or ``(n_steps, n_features)``.
        relevance (ndarray): Relevance scores matching ``time_series_sample``.
        title (str, optional): Optional figure title.
        show (bool, optional): Whether to call ``plt.show()``. Defaults to True.

    Returns:
        matplotlib.figure.Figure: Figure containing the relevance visualization.
    """
    ts, rel = _ensure_2d_pair(time_series_sample, relevance)
    n_steps, n_features = ts.shape

    fig, axes = plt.subplots(n_features, 1, squeeze=False, sharex=True)
    vmax = float(np.max(np.abs(rel))) or 1.0
    for i in range(n_features):
        ax = axes[i, 0]
        ymin, ymax = float(np.min(ts[:, i])), float(np.max(ts[:, i]))
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
        ax.imshow(
            rel[:, i][np.newaxis, :],
            aspect='auto',
            cmap='coolwarm',
            vmin=-vmax,
            vmax=vmax,
            alpha=0.5,
            extent=[0, n_steps - 1, ymin - pad, ymax + pad],
        )
        ax.plot(np.arange(n_steps), ts[:, i], color='black', linewidth=1.2)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_ylabel(f'Feature {i}')

    if title:
        fig.suptitle(title)
    axes[-1, 0].set_xlabel('Time step')
    if show:
        plt.show()
    return fig
