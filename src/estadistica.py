from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramer_v(s1: pd.Series, s2: pd.Series) -> float:
    d = pd.concat([s1, s2], axis=1).dropna()
    if d.shape[0] == 0:
        return 0.0
    tab = pd.crosstab(d.iloc[:, 0], d.iloc[:, 1])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(tab, correction=False)
    n = tab.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def correlation_ratio(categorias: pd.Series, valores: pd.Series) -> float:
    d = pd.concat([categorias, valores], axis=1).dropna()
    if d.shape[0] == 0 or d.iloc[:, 0].nunique() < 2:
        return 0.0
    cats = d.iloc[:, 0].astype(object)
    vals = d.iloc[:, 1].astype(float)
    media_global = vals.mean()
    grupos = vals.groupby(cats).agg(["size", "mean"])
    ss_between = float((grupos["size"] * (grupos["mean"] - media_global) ** 2).sum())
    ss_total = float(((vals - media_global) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0
