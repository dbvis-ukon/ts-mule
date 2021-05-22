"""Test module for evaluation functions."""
import numpy as np
import pytest

from tsmule.xai.evaluation import PerturbationAnalysis
from tsmule.xai.lime import LimeTS

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
    n_offs = (np.ceil(n_offs * (1 + 0.1))).astype(int)

    p_offs = n_offs/n_steps
    assert all(p_offs < 0.50)


def test_analysis_relevance():
    X = [mts, mts, mts]
    y = [1, 2, 3]

    def predict_(x): return 1
    def predict_fn_x(X): return np.array([1 for x in X])
    def eval_fn(y1, y2): return 1

    explainer = LimeTS()
    relevance = [explainer.explain(x, predict_) for x in X]

    pa = PerturbationAnalysis()
    pa.analysis_relevance(X, y, relevance,
                          predict_fn=predict_fn_x,
                          eval_fn=eval_fn)
    keys = ['original', 'percentile', 'random']
    assert all(k in pa.insights.keys() for k in keys)
    assert all([pa.insights[k] == 1 for k in keys])


def test_analysis_relevance_mean():
    X = [mts, mts, mts]
    y = [1, 2, 3]

    def predict_(x):
        return 1

    def predict_fn_x(X):
        return np.array([1 for x in X])

    def eval_fn(y1, y2):
        return 1

    explainer = LimeTS()
    relevance = [explainer.explain(x, predict_) for x in X]

    pa = PerturbationAnalysis()
    pa.analysis_relevance(X, y, relevance,
                          predict_fn=predict_fn_x,
                          replace_method="local_mean",
                          eval_fn=eval_fn)
    keys = ['original', 'percentile', 'random']
    assert all(k in pa.insights.keys() for k in keys)
    assert all([pa.insights[k] == 1 for k in keys])

    pa = PerturbationAnalysis()
    pa.analysis_relevance(X, y, relevance,
                          predict_fn=predict_fn_x,
                          replace_method="global_mean",
                          eval_fn=eval_fn)
    keys = ['original', 'percentile', 'random']
    assert all(k in pa.insights.keys() for k in keys)
    assert all([pa.insights[k] == 1 for k in keys])


@pytest.mark.skip("Manuel Test")
def test_analysis_relevance_manual():
    import dill
    from sklearn import metrics
    from tensorflow import keras

    data_dir = "demo/beijing_air_2_5"
    cnn_model = keras.models.load_model(
        f'{data_dir}/beijing_air_2_5_cnn_model.h5')
    with open(f'{data_dir}/beijing_air_2_5_test_data.dill', 'rb') as f:
        dataset_test = dill.load(f)

    # Define a predict fn/model
    def predict_(x):
        if len(x.shape) == 2:
            predictions = cnn_model.predict(x[np.newaxis]).ravel()
        if len(x.shape) == 3:
            predictions = cnn_model.predict(X).ravel()
        return predictions

    # Get test set
    X = dataset_test[0][:10]
    y = dataset_test[1][:10]
    explainer = LimeTS()
    relevance = [explainer.explain(
        x, predict_, segmentation_method="slopes-max") for x in X]

    pa = PerturbationAnalysis(replace_method='zeros')
    scores = pa.analysis_relevance(X, y, relevance,
                                   predict_fn=cnn_model.predict,
                                   eval_fn=metrics.mean_squared_error)
    scores
