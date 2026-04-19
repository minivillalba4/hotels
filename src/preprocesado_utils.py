import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WinsorizerPercentil(BaseEstimator, TransformerMixin):
    def __init__(self, columnas=None, p_inferior=0.01, p_superior=0.99):
        self.columnas = columnas
        self.p_inferior = p_inferior
        self.p_superior = p_superior

    def fit(self, X, y=None):
        cols = list(self.columnas) if self.columnas else []
        self.limites_ = {}
        for col in cols:
            valores = np.asarray(X[col], dtype=float)
            self.limites_[col] = (
                float(np.quantile(valores, self.p_inferior)),
                float(np.quantile(valores, self.p_superior)),
            )
        return self

    def transform(self, X):
        Xt = X.copy()
        for col, (q_bajo, q_alto) in self.limites_.items():
            Xt[col] = np.clip(Xt[col].to_numpy(dtype=float), q_bajo, q_alto)
        return Xt

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)
