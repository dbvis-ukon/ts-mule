"""Evaluation for different explainers."""
from functools import partial
import numpy as np

from ..sampling import replace as repl

class PerturbationBase:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def mask_percentile(x, percentile=90):
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
        m = cls.mask_percentile(x, percentile)
        m = cls._randomize(m, delta)
        return m


    @staticmethod
    def _perturb(x, m, replace_method='zeros'):
        """Perturb an instance x, given a mask. 

        Args:
            x (ndarray): time series with shape (n_steps, features)
            m (ndarray): a masking of 1s and 0s of x in shape of (n_steps, features)
            replace_method (str, optional): Replacement methods to replace disabled points. Defaults to 'zeros'.
                - All methods is built-in functions in sampling.replace module.

        Returns:
            ndarray: a new perturbed instance of x.
        """
        repl_fn = getattr(repl, replace_method)
        r = repl_fn(x)
        assert x.shape == m.shape == r.shape
        z = x * m + r * (1 - m)
        return z


    def perturb(self, X, R, replace_method="zeros", percentile=90, shuffle=False, delta=0.0):
        """Perturb list of time series

        Args:
            X (Iterable): Instances with shape (n_steps, features). 
            R (Iterable): Relevances for each instance with shape (n_steps, features).
            replace_method (str, optional): method to replace disabled. Defaults to "zeros".
            shuffle (bool, optional): If true, then random. Defaults to False.
                The relevance is randomized based on number of disabled relevance. 
            delta (float, optional): Increase/decrease the number of disabled relevance. Defaults to 0.0.

        Yields:
            [ndarray]: multiple perturbed instances
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
    def __init__(self, percentile=90, delta=0.0, replace_method='zeros') -> None:
        super().__init__()
        
        self.insights = dict()
        self.percentile = percentile
        self.delta = delta
        self.repl_method = replace_method
        
        
    def add_insight(self, k, v):
        self.insights.update({k: v})


    def to_json(self, file_path):
        pass
    
    
    def analysis_relevance(self, X, y, R, 
                           predict_fn, eval_fn, 
                           replace_method='zeros', percentile=90, delta=0.0):
        
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
        self.add_insight(f'percentile', score)
        
        # Score for random
        y_pred = predict_fn(X_random).ravel()
        score = eval_fn(y_pred, y)
        self.add_insight('random', score)
        
        return self.insights
        
