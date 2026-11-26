"""Test module for LIME explanations (forecasting and classification)."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from tsmule.sampling.perturb import Perturbation
from tsmule.sampling.segment import WindowSegmentation
from tsmule.xai.lime import Kernels, LimeBase, LimeTS

ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([
    [1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
    [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0],
]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)


def test_samples_unpack():
    sampler = Perturbation(random_state=42)

    samples = sampler.get_samples(ts, segm_ts, n_samples=20, random_state=42)
    new_x, z_prime, pi = list(zip(*samples))
    assert len(new_x) == len(z_prime) == len(pi) == 20
    assert new_x[0].shape == ts.shape
    assert isinstance(pi[0], float)
    assert z_prime[0].shape == np.unique(segm_ts).shape

    samples = sampler.get_samples(mts, segm_mts, n_samples=20, random_state=42)
    new_x, z_prime, pi = list(zip(*samples))
    assert len(new_x) == len(z_prime) == len(pi) == 20
    assert new_x[0].shape == mts.shape
    assert z_prime[0].shape == np.unique(segm_mts).shape


def test_base_explain():
    sampler = Perturbation(random_state=42)
    kernel = Ridge(alpha=0.01)

    def predict_fn(x):
        return 1.0

    lime = LimeBase()
    samples = sampler.get_samples(ts, segm_ts, n_samples=50, random_state=42)
    _kernel, score = lime._explain(samples, kernel, predict_fn)
    assert score == 1.0 and _kernel.coef_ is not None

    samples = sampler.get_samples(mts, segm_mts, n_samples=50, random_state=42)
    _kernel, score = lime._explain(samples, kernel, predict_fn)
    assert score == 1.0 and _kernel.coef_ is not None


def test_ts_explain_default_and_properties():
    # Default explain() without explicit segmentation_method must work out of the box
    lime_ts = LimeTS(n_samples=150, random_state=42)

    def predict_fn(x):
        return float(x.sum())

    coef = lime_ts.explain(mts, predict_fn)
    assert coef.shape == mts.shape
    assert np.all(coef.ravel() >= 0)
    assert lime_ts.coef is not None
    assert lime_ts.segment_coef is not None
    assert np.array_equal(lime_ts.coef, coef)
    assert lime_ts.target_class is None
    assert lime_ts.score > 0.8


def test_kernels_not_shared_across_instances():
    e1 = LimeTS(n_samples=40, random_state=1)
    e2 = LimeTS(n_samples=40, random_state=2)
    assert e1._kernel is not Kernels.Lasso
    assert e1._kernel is not e2._kernel

    e1.explain(mts, lambda x: float(x.sum()))
    assert not hasattr(Kernels.Lasso, 'coef_')
    assert not hasattr(e2._kernel, 'coef_')


def test_ts_explain_classification_1d_and_batch():
    rng = np.random.default_rng(7)
    x = rng.normal(1.0, 0.2, size=24)
    # Signal in indices [12:18] drives class 1 probability
    x[12:18] = 5.0

    def clf_predict_proba(batch):
        arr = np.asarray(batch, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        logits = arr[:, 12:18].mean(axis=1) - 1.5
        p1 = 1.0 / (1.0 + np.exp(-logits))
        probs = np.stack([1.0 - p1, p1], axis=1)
        return probs[0] if batch.ndim == 1 else probs

    explainer = LimeTS(
        kernel=Ridge(alpha=0.1),
        segmenter=WindowSegmentation(partitions=4, win_length=6),
        n_samples=200,
        random_state=42,
    )

    rel_auto = explainer.explain(x, clf_predict_proba, batch_predict=True)
    assert rel_auto.shape == x.shape
    assert explainer.target_class == 1
    assert np.argmax(rel_auto) in range(12, 18)

    rel_class0 = explainer.explain(x, clf_predict_proba, target_class=0, batch_predict=True)
    assert explainer.target_class == 0
    assert np.allclose(rel_class0, -rel_auto, atol=1e-6)


def test_ts_explain_multiclass_and_invalid_target():
    x = np.ones((20, 2))

    def multiclass_fn(batch):
        # 3-class probabilities for batch (N, 20, 2)
        s = batch[:, :10, 0].mean(axis=1)
        logits = np.stack([np.zeros_like(s), s, -s], axis=1)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    explainer = LimeTS(n_samples=60, win_length=4, random_state=0)
    rel = explainer.explain(x, multiclass_fn, target_class=2, batch_predict=True)
    assert rel.shape == x.shape
    assert explainer.target_class == 2

    with pytest.raises(IndexError):
        explainer.explain(x, multiclass_fn, target_class=5, batch_predict=True)
