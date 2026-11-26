"""Test module for time series segmentation."""
import numpy as np
import pytest

from tsmule.sampling.segment import (
    MatrixProfileSegmentation,
    SAXSegmentation,
    SegmentationPicker,
    WindowSegmentation,
)


def _make_series(n_steps=40, n_features=2, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_steps)
    cols = [np.sin(t * (f + 1)) + 0.05 * rng.standard_normal(n_steps) for f in range(n_features)]
    return np.stack(cols, axis=1)


def test_matrix_profile_all_methods_2d_and_1d():
    mts = _make_series(n_steps=32, n_features=2)
    uts = mts[:, 0]

    for method in ('slopes-sorted', 'slopes-not-sorted', 'bins-min', 'bins-max'):
        segm = MatrixProfileSegmentation(partitions=6, win_length=4)
        mask_2d = segm.segment(mts, segmentation_method=method)
        assert mask_2d.shape == mts.shape
        assert np.issubdtype(mask_2d.dtype, np.integer)
        assert len(np.unique(mask_2d[:, 0])) >= 1
        # Different features must have disjoint segment IDs
        assert set(np.unique(mask_2d[:, 0])).isdisjoint(set(np.unique(mask_2d[:, 1])))

        mask_1d = segm.segment(uts, segmentation_method=method)
        assert mask_1d.shape == uts.shape
        assert np.issubdtype(mask_1d.dtype, np.integer)


def test_matrix_profile_auto_win_length():
    mts = _make_series(n_steps=50, n_features=2)
    segm = MatrixProfileSegmentation(partitions=5, win_length=-1)
    assert segm.win_length == -1

    for method in ('slopes-sorted', 'slopes-not-sorted', 'bins-min', 'bins-max'):
        mask = segm.segment(mts, segmentation_method=method)
        assert mask.shape == mts.shape
        assert len(np.unique(mask)) > 2


def test_matrix_profile_slopes_not_sorted_differs_from_sorted():
    mts = _make_series(n_steps=60, n_features=1)
    segm = MatrixProfileSegmentation(partitions=6, win_length=5)
    m_sorted = segm.segment(mts, segmentation_method='slopes-sorted')
    m_unsorted = segm.segment(mts, segmentation_method='slopes-not-sorted')
    assert m_sorted.shape == m_unsorted.shape == mts.shape
    assert not np.array_equal(m_sorted, m_unsorted)


def test_matrix_profile_constant_series_and_validation():
    const = np.ones((20, 2))
    segm = MatrixProfileSegmentation(partitions=4, win_length=4)
    for method in ('slopes-sorted', 'bins-min', 'bins-max'):
        mask = segm.segment(const, segmentation_method=method)
        assert mask.shape == const.shape

    with pytest.raises(ValueError):
        segm.segment(const, segmentation_method='slopes-max')

    with pytest.raises(ValueError):
        MatrixProfileSegmentation(partitions=2, win_length=4).segment(np.ones((3, 1)))


def test_sax_segmentation_including_constant_and_symbol_zero():
    # Step series with 3 levels ensures ordinal symbol 0 transitions are exercised
    step = np.array([0.0] * 10 + [5.0] * 10 + [10.0] * 10 + [0.0] * 10).reshape(-1, 1)
    sseg = SAXSegmentation(partitions=4)
    mask = sseg.segment(step)
    assert mask.shape == step.shape
    assert np.issubdtype(mask.dtype, np.integer)
    # Boundary at step 10 (transition from symbol 0) must create a new segment
    assert mask[0, 0] != mask[10, 0]
    assert mask[10, 0] != mask[20, 0]
    assert mask[20, 0] != mask[30, 0]

    # Constant series with partitions > n_steps must not ZeroDivisionError
    const = np.ones((5, 2))
    sseg_many = SAXSegmentation(partitions=10)
    mask_const = sseg_many.segment(const)
    assert mask_const.shape == const.shape

    # 1D input
    mask_1d = sseg.segment(step.ravel())
    assert mask_1d.shape == (40,)


def test_window_segmentation_uniform_and_exponential():
    mts = _make_series(n_steps=20, n_features=2)
    wseg = WindowSegmentation(partitions=5, win_length=5)

    u_mask = wseg.segment(mts, segmentation_method='uniform')
    assert u_mask.shape == mts.shape
    assert np.issubdtype(u_mask.dtype, np.integer)
    assert len(np.unique(u_mask[:, 0])) == 4

    e_mask = wseg.segment(mts, segmentation_method='exponential')
    assert e_mask.shape == mts.shape
    assert np.issubdtype(e_mask.dtype, np.integer)
    assert len(np.unique(e_mask[:, 0])) >= 2

    # Short time series edge cases
    assert wseg.segment(np.ones((1, 1)), segmentation_method='exponential').shape == (1, 1)
    assert wseg.segment(np.ones(2), segmentation_method='exponential').shape == (2,)

    with pytest.raises(ValueError):
        wseg.segment(mts, segmentation_method='invalid')

    with pytest.raises(ValueError):
        WindowSegmentation(partitions=5, win_length=25).segment(mts, segmentation_method='uniform')


def test_segmentation_picker():
    assert isinstance(SegmentationPicker.select('sax', 5), SAXSegmentation)
    assert isinstance(SegmentationPicker.select('window', 5, 4), WindowSegmentation)
    assert isinstance(SegmentationPicker.select('matrix-slope', 5, -1), MatrixProfileSegmentation)
    with pytest.raises(ValueError):
        SegmentationPicker.select('unknown', 5)
