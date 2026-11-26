"""Test module for evaluation functions."""
import json
import warnings

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, r2_score

from tsmule.xai.evaluation import PerturbationAnalysis
from tsmule.xai.lime import LimeTS

ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([
    [1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
    [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0],
]).T


def test_mask_percentile():
    pa = PerturbationAnalysis()
    m = pa.mask_percentile(ts, percentile=90)
    assert m.shape == ts.shape

    expected = np.array([[1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]])
    assert np.array_equal(m.T, expected)

    # Constant relevance must not divide by zero or emit warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        m_const = pa.mask_percentile(np.ones((10, 2)), percentile=90)
    assert len(caught) == 0
    assert np.all(m_const == 1)

    # use_abs captures strong negative attributions
    neg = np.array([-10.0, 0.0, 0.0, 0.0, 1.0])
    m_abs = pa.mask_percentile(neg, percentile=75, use_abs=True)
    assert m_abs[0] == 0


def test_mask_randomize_and_delta_clamping():
    pa = PerturbationAnalysis()
    m = pa.mask_randomize(ts, percentile=90, delta=0.1, random_state=42)
    assert m.shape == ts.shape
    assert (m == 0).sum() / len(ts) < 0.5

    # Extreme delta values must clamp safely to [0, n_steps]
    m_all_off = pa.mask_randomize(ts, percentile=50, delta=10.0, random_state=0)
    assert np.all(m_all_off == 0)

    m_all_on = pa.mask_randomize(ts, percentile=90, delta=-1.5, random_state=0)
    assert np.all(m_all_on == 1)


def test_analysis_relevance_argument_order_and_to_json(tmp_path):
    X = np.array([mts, mts * 1.5, mts * 2.0])
    y = np.array([1.0, 2.0, 3.0])

    # Asymmetric metric: r2_score(y_true, y_pred) != r2_score(y_pred, y_true)
    def predict_fn_x(batch):
        return np.array([1.1, 2.0, 2.8])

    explainer = LimeTS(n_samples=40, random_state=42)
    relevance = [explainer.explain(x, lambda s: float(s.sum())) for x in X]

    pa = PerturbationAnalysis()
    insights = pa.analysis_relevance(
        X, y, relevance, predict_fn=predict_fn_x, eval_fn=r2_score, random_state=42
    )
    assert insights['original'] == pytest.approx(r2_score(y, predict_fn_x(X)))

    out_file = tmp_path / 'insights.json'
    pa.to_json(out_file)
    loaded = json.loads(out_file.read_text(encoding='utf-8'))
    assert set(loaded.keys()) == {'original', 'percentile', 'random'}


def test_analysis_relevance_mean_methods():
    X = np.array([mts, mts, mts])
    y = np.array([1.0, 2.0, 3.0])

    def predict_fn_x(batch):
        return np.ones(len(batch))

    explainer = LimeTS(n_samples=40, random_state=42)
    relevance = [explainer.explain(x, lambda s: 1.0) for x in X]

    for method in ('local_mean', 'global_mean'):
        pa = PerturbationAnalysis()
        pa.analysis_relevance(
            X,
            y,
            relevance,
            predict_fn=predict_fn_x,
            replace_method=method,
            eval_fn=lambda yt, yp: 1.0,
            random_state=42,
        )
        assert all(pa.insights[k] == 1.0 for k in ('original', 'percentile', 'random'))


def test_analysis_relevance_classification_faithfulness():
    rng = np.random.default_rng(123)
    X = rng.normal(0.0, 0.2, size=(20, 30))
    y = np.array([0] * 10 + [1] * 10)
    # Class 1 has a clear positive signal in [10:15]; class 0 has negative signal
    X[:10, 10:15] = -3.0
    X[10:, 10:15] = 3.0

    def clf_predict(batch):
        return (batch[:, 10:15].mean(axis=1) > 0.0).astype(int)

    # True relevance highlights [10:15]
    R = np.zeros_like(X)
    R[:, 10:15] = 1.0

    pa = PerturbationAnalysis()
    insights = pa.analysis_relevance(
        X,
        y,
        R,
        predict_fn=clf_predict,
        eval_fn=accuracy_score,
        replace_method='zeros',
        percentile=80,
        random_state=42,
    )
    assert insights['original'] == 1.0
    assert insights['percentile'] < insights['random']
    assert pa.is_valid(higher_is_better=True)
    assert pa.perturbation_score() > 1.0
