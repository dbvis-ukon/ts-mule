"""Test module for evaluation functions."""
import numpy as np
from xai.evaluation import PerturbationAnalysis

ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
                [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0]]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)

def test_mask_percentile():
    pa = PerturbationAnalysis()
    m = pa.mask_percentile(ts, percentile=90)
    assert m.shape == ts.shape
    
    expected = np.array([[1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]])
    assert np.array_equal(m.T, expected)
    

def test_mask_randomize():
    pa = PerturbationAnalysis()
    # UTS
    x = ts
    m = pa.mask_randomize(x, percentile=90, delta=0.1)
    assert m.shape == x.shape
    
    n_steps, _ = x.shape
    n_offs = (m == 0).sum(axis=0)
    p_offs = n_offs/n_steps
    assert all(p_offs < 0.50)
    
    # MTS
    x = mts
    m = pa.mask_randomize(x, percentile=90, delta=0.1)
    assert m.shape == x.shape
    
    n_steps, _ = x.shape
    n_offs = (m == 0).sum(axis=0)
    p_offs = n_offs/n_steps
    assert all(p_offs < 0.50)