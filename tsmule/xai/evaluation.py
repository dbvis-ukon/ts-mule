"""Evaluation module for explainers."""
from functools import partial
import numpy as np

from ..sampling import replace as repl


class PerturbationBase:
    """Base module for relevance analysis based on perturbation method."""


    def __init__(self) -> None:
        """Construct of perturbation base module."""
        pass


    @staticmethod
    def mask_percentile(x, percentile=90.):
        """Create a mask based on percentile for the relevance.

        Args:
            x (ndarray): Relevance or coefficient.
            percentile (float, optional): Percentile of relevance to mask. Defaults to 90.

        Returns:
            ndarray: an ndarray mask of 0s and 1s. Same shape with x.
        """
        # 1/on/keep and 0/off/disabled
        # n_steps, features = x

        # normalized relevance
        amin = partial(np.min, axis=0)
        amax = partial(np.max, axis=0)
        relevance_norm = (x - amin(x)) / (amax(x) - (amin(x)))

        # get points > percentile 90, which are being perturbed
        p90 = np.percentile(relevance_norm, percentile, axis=0)
        m = (relevance_norm > p90)

        # reverse to have 1 = on, 0 = off
        m = 1 - m

        return m


    @staticmethod
    def _randomize(m, delta=0.0):
        # Random the masked percentile-90.
        # m = mask_percentile(x)
        m = np.array(m)     # copy
        n_steps, features = m.shape

        # Get number of off-relevance per feature
        n_offs = (m == 0).sum(axis=0)

        # Increase/decrease number of off-relevance with delta
        #   Notice, n_offs is a vector of all features
        n_offs = (np.ceil(n_offs * (1 + delta))).astype(int)
        n_ons = (n_steps - n_offs).astype(int)

        # Get probability of disabled relevance
        random_mask = []
        for i in range(features):
            t = np.concatenate([np.zeros(n_offs[i]), np.ones(n_ons[i])])
            random_mask.append(t)
        random_mask = np.stack(random_mask, axis=1).astype(int)

        assert m.shape == random_mask.shape

        # inplace shuffle for each feature
        _ = np.apply_along_axis(np.random.shuffle, axis=0, arr=random_mask)

        return random_mask


    @classmethod
    def mask_randomize(cls, x, percentile=90, delta=0.0):
        """Create mask based on percentile, then randomize the number of masked ones.

        Args:
            x (ndarray): Relevance or coefficient.
            percentile (float, optional): Percentile of relevance to mask. Defaults to 90.
            delta (float, optional): Specifiy when you want to increase or decrease the number of masked one.
                Defaults to 0.0.

        Returns:
            ndarray: an randomized mask of 0s and 1s. Same shape with x.
        """
        m = cls.mask_percentile(x, percentile)
        m = cls._randomize(m, delta)
        return m


    @staticmethod
    def _perturb(x, m, replace_method='zeros'):
        repl_fn = getattr(repl, replace_method)
        r = repl_fn(x, m)
        assert x.shape == m.shape == r.shape
        z = x * m + r * (1 - m)
        return z


    def perturb(self, X, R, replace_method="zeros", percentile=90, shuffle=False, delta=0.0):
        """Perturb list of time series.

        Args:
            X (Iterable): Instances with shape (n_steps, features).
            R (Iterable): Relevances for each instance with shape (n_steps, features).
            replace_method (str, optional): method to replace disabled. Defaults to "zeros".
            shuffle (bool, optional): If true, then random. Defaults to False.
                The relevance is randomized based on number of disabled relevance.
            delta (float, optional): Increase/decrease the number of disabled relevance.
                Defaults to 0.0.

        Yields:
            ndarray: multiple perturbed instances
        """
        for x, r in zip(X, R):
            assert x.shape == r.shape, \
                f"Conflict in shape, instance x with shape {x.shape} while relevance r: {r.shape}"

            # Get mask based on relevance
            if shuffle:
                m = self.mask_randomize(r, percentile, delta)
            else:
                m = self.mask_percentile(r, percentile)
            yield self._perturb(x, m, replace_method=replace_method)


class PerturbationAnalysis(PerturbationBase):
    """Module for relevance analysis based on perturbation method."""


    def __init__(self,) -> None:
        """Construct analysis class for perturbation method."""
        super().__init__()

        self.insights = dict()


    def add_insight(self, k, v):
        """Store the result to the insights dict.

        Args:
            k (str): name of the insight/evalutation
            v (float): value/score of the evaluation.
        """
        self.insights.update({k: v})


    def to_json(self, file_path):
        """Dumpy insights to json file."""
        pass


    def analysis_relevance(self, X, y, R,
                           predict_fn, eval_fn,
                           replace_method='zeros', percentile=90, delta=0.0):
        """Analysis of relevance, proposed by Schlegel et al.[1].

        The analysis perturb the test set X based on its relevance. Then the new generated test sets,
        including `original`, perturbed `percentile`, and perturbed `random` test sets.
        These test sets are evaluated and compared each other.

        If error(`original`) <= error(`random`) <= error(`percentile`), then the explanation is valid.

        .. [1] Schlegel, U., Arnout, H., El-Assady, M., Oelke, D., & Keim, D. A. (2019).
            Towards A Rigorous Evaluation Of XAI Methods On Time Series.
            In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW).
            (pp. 4321-4325).

        Args:
            X (Iterable): Instances with shape (n_steps, features).
            y (Iterable): True values of input X.
            R (Iterable): Relevances for each instance with shape (n_steps, features).
            predict_fn ([type]): [description]
            eval_fn ([type]): [description]
            replace_method (str, optional): method to replace disabled. Defaults to "zeros".
            shuffle (bool, optional): If true, then random. Defaults to False.
                The relevance is randomized based on number of disabled relevance.
            percentile (int, optional): [description]. Defaults to 90.
            delta (float, optional): Increase/decrease the number of disabled relevance.
                Defaults to 0.0.

        Returns:
            dict: Scores of 'original', 'percentile' and 'random' test.
        """
        # Perturb instance based on percentile
        X_percentile = self.perturb(X, R,
                                    replace_method=replace_method,
                                    percentile=percentile,
                                    )
        X_percentile = np.array(list(X_percentile))

        X_random = self.perturb(X, R,
                                replace_method=replace_method,
                                percentile=percentile,
                                shuffle=True,
                                delta=delta
                                )
        X_random = np.array(list(X_random))

        # Score for original
        y_pred = predict_fn(X).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('original', score)

        # Score for Percentile
        y_pred = predict_fn(X_percentile).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('percentile', score)

        # Score for random
        y_pred = predict_fn(X_random).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('random', score)

        return self.insights
