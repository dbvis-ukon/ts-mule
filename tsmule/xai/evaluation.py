""" Evaluation for different explainers. """
import pandas as pd
from .lime import LIMEAbstract
from biokit.viz import corrplot


def corr_matrix(coef_or_models, names=None, **corr_kwargs):
    assert isinstance(coef_or_models, list)
    if all([isinstance(m, LIMEAbstract) for m in coef_or_models]):
        coef = [m.coef for m in coef_or_models]
    else:
        coef = coef_or_models

    if names is None:
        names = [i for i in range(len(coef))]

    assert len(names) == len(coef), \
        f'Not matching length of names and coef_or_models, {len(names)} and {len(coef)}'

    df = pd.DataFrame({n: c for n, c in zip(names, coef)})
    return df.corr(**corr_kwargs)


def plot_corr(df_corr, method='square'):
    c = corrplot.Corrplot(df_corr)
    c.plot(method=method, shrink=.9, rotation=45)
    # c.plot(method=method, fontsize=8, colorbar=False)
