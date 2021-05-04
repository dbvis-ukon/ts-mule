"""Test module for replacement calculations."""
import numpy as np
from pert import repl

ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
                [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0]]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)

def test_repl_zeros():
    x = repl.zeros(ts)
    assert np.array_equal(x, np.zeros_like(ts))
    
    x = repl.zeros(mts)
    assert np.array_equal(x, np.zeros_like(mts))
    
def test_repl_local_mean():
    x = repl.local_mean(ts, segm_ts)
    x = x.round(1)
    expected = np.array(
                [[4.3, 4.3, 4.3, 5.5, 5.5, 5.5, 5.5, 5.5, 
                 5.5, 5.5, 5.5, 3.4, 3.4, 3.4, 3.4, 3.4]]
                ).T
    
    assert np.array_equal(x, expected)
    
    x = repl.local_mean(mts, segm_mts)
    x = x.round(1)
    expected = np.array([
        [4.3, 4.3, 4.3, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 3.4, 3.4, 3.4, 3.4, 3.4],
        [5. , 5. , 5. , 5. , 7. , 7. , 7. , 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2]]
        ).T
    
    assert np.array_equal(x, expected)
    
def test_global_mean():
    x = repl.global_mean(ts)
    x = x.round(1)
    expected = np.zeros_like(ts).astype(float)
    expected.fill(4.6)
    assert np.array_equal(x, expected)
    
    x = repl.global_mean(mts)
    x = x.round(1)
    expected = np.zeros_like(mts).astype(float)
    expected[:, 0].fill(4.6)
    expected[:, 1].fill(4.4)
    assert np.array_equal(x, expected)