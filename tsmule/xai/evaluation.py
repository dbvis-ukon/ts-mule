"""Evaluation module for explainers."""
import json

import numpy as np

from ..sampling import replace as repl


class PerturbationBase:
    """Base module for relevance analysis based on perturbation method."""

    def __init__(self) -> None:
        """Construct perturbation base module."""
        pass

    @staticmethod
    def mask_percentile(x, percentile=90.0, use_abs=False):
        """Create a binary keep/replace mask based on a relevance percentile.

        Args:
            x (ndarray): Relevance or coefficient array with shape
                ``(n_steps,)`` or ``(n_steps, n_features)``.
            percentile (float, optional): Percentile threshold above which points are replaced
                (set to 0). Defaults to 90.0.
            use_abs (bool, optional): If True, threshold on ``abs(x)`` so strong negative
                attributions are also treated as highly relevant. Defaults to False.

        Returns:
            ndarray: Integer mask of 0s (replace) and 1s (keep) with the same shape as ``x``.
        """
        x_arr = np.asarray(x, dtype=float)
        was_1d = x_arr.ndim == 1
        x_2d = x_arr.reshape(-1, 1) if was_1d else x_arr
        values = np.abs(x_2d) if use_abs else x_2d

        amin = np.min(values, axis=0)
        amax = np.max(values, axis=0)
        denom = amax - amin
        safe_denom = np.where(denom == 0, 1.0, denom)
        relevance_norm = np.where(denom == 0, 0.0, (values - amin) / safe_denom)

        thresh = np.percentile(relevance_norm, percentile, axis=0)
        high_rel = np.where(denom == 0, False, relevance_norm > thresh)

        # 1 = on/keep, 0 = off/replace
        m = (1 - high_rel.astype(int))
        return m.ravel() if was_1d else m

    @staticmethod
    def _randomize(m, delta=0.0, random_state=None):
        m_arr = np.asarray(m, dtype=int)
        was_1d = m_arr.ndim == 1
        m_2d = m_arr.reshape(-1, 1) if was_1d else m_arr
        n_steps, features = m_2d.shape
        rng = np.random.default_rng(random_state)

        n_offs = (m_2d == 0).sum(axis=0)
        n_offs = np.clip(np.ceil(n_offs * (1 + delta)).astype(int), 0, n_steps)
        n_ons = n_steps - n_offs

        random_mask = np.empty((n_steps, features), dtype=int)
        for i in range(features):
            col = np.concatenate(
                [
                    np.zeros(int(n_offs[i]), dtype=int),
                    np.ones(int(n_ons[i]), dtype=int),
                ]
            )
            rng.shuffle(col)
            random_mask[:, i] = col

        return random_mask.ravel() if was_1d else random_mask

    @classmethod
    def mask_randomize(cls, x, percentile=90, delta=0.0, use_abs=False, random_state=None):
        """Create mask based on percentile, then randomize masked positions per feature.

        Args:
            x (ndarray): Relevance or coefficient array.
            percentile (float, optional): Percentile of relevance to mask. Defaults to 90.
            delta (float, optional): Relative adjustment to number of masked points.
                Defaults to 0.0.
            use_abs (bool, optional): Whether to threshold on ``abs(x)``. Defaults to False.
            random_state (int or np.random.Generator, optional): Random seed or generator.

        Returns:
            ndarray: Randomized binary mask of 0s and 1s with the same shape as ``x``.
        """
        m = cls.mask_percentile(x, percentile, use_abs=use_abs)
        return cls._randomize(m, delta, random_state=random_state)

    @staticmethod
    def _perturb(x, m, replace_method='zeros'):
        repl_fn = getattr(repl, replace_method, None)
        if repl_fn is None or not callable(repl_fn) or replace_method.startswith('_'):
            raise ValueError(f"Unknown replace_method '{replace_method}'")
        x_arr = np.asarray(x, dtype=float)
        r = repl_fn(x_arr, m)
        assert x_arr.shape == m.shape == r.shape
        return x_arr * m + r * (1 - m)

    def perturb(
        self,
        X,
        R,
        replace_method='zeros',
        percentile=90,
        shuffle=False,
        delta=0.0,
        use_abs=False,
        random_state=None,
    ):
        """Perturb a sequence of time series according to relevance arrays.

        Args:
            X (Iterable): Instances with shape ``(n_steps,)`` or ``(n_steps, features)``.
            R (Iterable): Relevances matching each instance's shape.
            replace_method (str, optional): Replacement method name. Defaults to ``'zeros'``.
            percentile (float, optional): Relevance percentile threshold. Defaults to 90.
            shuffle (bool, optional): If True, randomize masked positions. Defaults to False.
            delta (float, optional): Relative adjustment for randomized mask count. Defaults to 0.0.
            use_abs (bool, optional): Whether to threshold on ``abs(r)``. Defaults to False.
            random_state (int or np.random.Generator, optional): Random seed or generator.

        Yields:
            ndarray: Perturbed time series instances.
        """
        rng = np.random.default_rng(random_state) if shuffle else None
        for x, r in zip(X, R):
            x_arr = np.asarray(x)
            r_arr = np.asarray(r)
            assert x_arr.shape == r_arr.shape, (
                f'Conflict in shape, instance x with shape {x_arr.shape} '
                f'while relevance r: {r_arr.shape}'
            )

            if shuffle:
                m = self.mask_randomize(
                    r_arr,
                    percentile=percentile,
                    delta=delta,
                    use_abs=use_abs,
                    random_state=rng,
                )
            else:
                m = self.mask_percentile(r_arr, percentile=percentile, use_abs=use_abs)
            yield self._perturb(x_arr, m, replace_method=replace_method)


class PerturbationAnalysis(PerturbationBase):
    """Module for relevance analysis based on perturbation method."""

    def __init__(self) -> None:
        """Construct analysis class for perturbation method."""
        super().__init__()
        self.insights = {}

    def add_insight(self, k, v):
        """Store a metric result into the insights dict.

        Args:
            k (str): Name of the evaluation condition.
            v (float): Metric value.
        """
        self.insights[k] = float(v)

    def to_json(self, file_path):
        """Dump insights dictionary to a JSON file."""
        serializable = {k: float(v) for k, v in self.insights.items()}
        with open(file_path, 'w', encoding='utf-8') as fh:
            json.dump(serializable, fh, indent=2)

    def is_valid(self, higher_is_better=False):
        """Check whether perturbation insights satisfy the faithfulness ordering.

        Args:
            higher_is_better (bool, optional): Set False for error metrics (e.g., MSE,
                where ``original <= random <= percentile`` indicates validity), or True for
                score/accuracy metrics (where ``original >= random >= percentile``).
                Defaults to False.

        Returns:
            bool: True if the required ordering holds.
        """
        required = ('original', 'random', 'percentile')
        if not all(k in self.insights for k in required):
            raise ValueError('insights must contain original, random, and percentile scores.')
        orig = self.insights['original']
        rand = self.insights['random']
        pert = self.insights['percentile']
        if higher_is_better:
            return bool(orig >= rand >= pert)
        return bool(orig <= rand <= pert)

    def perturbation_score(self):
        """Compute the relative perturbation ratio ``|pert - orig| / |rand - orig|``."""
        required = ('original', 'random', 'percentile')
        if not all(k in self.insights for k in required):
            raise ValueError('insights must contain original, random, and percentile scores.')
        orig = self.insights['original']
        rand = self.insights['random']
        pert = self.insights['percentile']
        denom = abs(rand - orig)
        num = abs(pert - orig)
        if denom == 0:
            return float('nan') if num == 0 else float('inf')
        return float(num / denom)

    @staticmethod
    def _format_predictions(preds):
        arr = np.asarray(preds)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
        return arr

    def analysis_relevance(
        self,
        X,
        y,
        R,
        predict_fn,
        eval_fn,
        replace_method='zeros',
        percentile=90,
        delta=0.0,
        use_abs=False,
        random_state=None,
    ):
        """Analysis of relevance proposed by Schlegel et al. [1]_.

        Perturbs ``X`` at high-relevance positions (``percentile``) and at an equal number of
        random positions (``random``), then compares ``eval_fn(y, predict_fn(...))`` across
        ``original``, ``percentile``, and ``random``.

        .. [1] Schlegel, U., Arnout, H., El-Assady, M., Oelke, D., & Keim, D. A. (2019).
            Towards A Rigorous Evaluation Of XAI Methods On Time Series.
            In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW),
            pp. 4321-4325.

        Args:
            X (Iterable): Instances with shape ``(n_steps,)`` or ``(n_steps, features)``.
            y (Iterable): True targets or labels for ``X``.
            R (Iterable): Relevances matching each instance's shape.
            predict_fn (callable): Callable taking a batch of instances and returning predictions.
            eval_fn (callable): Metric function with signature ``eval_fn(y_true, y_pred)``.
            replace_method (str, optional): Replacement method. Defaults to ``'zeros'``.
            percentile (float, optional): Relevance percentile threshold. Defaults to 90.
            delta (float, optional): Relative adjustment for random mask size. Defaults to 0.0.
            use_abs (bool, optional): Whether to threshold on ``abs(r)``. Defaults to False.
            random_state (int or np.random.Generator, optional): Random seed for reproducibility.

        Returns:
            dict: Scores for ``'original'``, ``'percentile'``, and ``'random'``.
        """
        self.insights = {}
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        X_percentile = np.asarray(
            list(
                self.perturb(
                    X_arr,
                    R,
                    replace_method=replace_method,
                    percentile=percentile,
                    use_abs=use_abs,
                )
            ),
            dtype=float,
        )

        X_random = np.asarray(
            list(
                self.perturb(
                    X_arr,
                    R,
                    replace_method=replace_method,
                    percentile=percentile,
                    shuffle=True,
                    delta=delta,
                    use_abs=use_abs,
                    random_state=random_state,
                )
            ),
            dtype=float,
        )

        y_pred_orig = self._format_predictions(predict_fn(X_arr))
        self.add_insight('original', eval_fn(y_arr, y_pred_orig))

        y_pred_pert = self._format_predictions(predict_fn(X_percentile))
        self.add_insight('percentile', eval_fn(y_arr, y_pred_pert))

        y_pred_rand = self._format_predictions(predict_fn(X_random))
        self.add_insight('random', eval_fn(y_arr, y_pred_rand))

        return self.insights
