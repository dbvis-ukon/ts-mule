"""Test module for visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np

from tsmule.xai.viz import (
    visualize_perturbation_masks,
    visualize_relevance,
    visualize_segmentation_mask,
)

plt.switch_backend('Agg')


def test_visualize_segmentation_mask_univariate_and_multivariate():
    ts_1d = np.sin(np.linspace(0, 6, 30))
    seg_1d = np.repeat([0, 1, 2], 10)
    fig1, fig2 = visualize_segmentation_mask(ts_1d, seg_1d, show=False)
    assert fig1 is not None and fig2 is not None
    plt.close('all')

    ts_2d = np.stack([ts_1d, np.cos(np.linspace(0, 6, 30))], axis=1)
    seg_2d = np.stack([seg_1d, seg_1d + 3], axis=1)
    fig3, fig4 = visualize_segmentation_mask(ts_2d, seg_2d, show=False)
    assert fig3 is not None and fig4 is not None
    plt.close('all')


def test_visualize_perturbation_masks_and_relevance():
    orig = np.sin(np.linspace(0, 6, 30))
    mask = np.ones_like(orig, dtype=int)
    mask[10:15] = 0
    pert = orig * mask

    fig_p = visualize_perturbation_masks(orig, pert, mask=mask, show=False)
    assert fig_p is not None

    rel = np.zeros_like(orig)
    rel[10:15] = 1.0
    fig_r = visualize_relevance(orig, rel, title='Test Relevance', show=False)
    assert fig_r is not None
    plt.close('all')
