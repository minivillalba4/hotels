from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

from src.ml.shap_utils import nombre_legible
from src.ui.tema import BRAND


def bar_importance_plotly(mean_abs_shap: pd.Series, top: int = 20) -> go.Figure:
    datos = mean_abs_shap.head(top).iloc[::-1]
    etiquetas = [nombre_legible(f) for f in datos.index]
    fig = px.bar(
        x=datos.values,
        y=etiquetas,
        orientation="h",
        labels={"x": "Importancia media |SHAP|", "y": "Variable"},
        color_discrete_sequence=[BRAND["primary"]],
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>|SHAP| medio: %{x:.4f}<extra></extra>")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def waterfall_plotly(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    expected_value: float,
    top: int = 15,
) -> go.Figure:
    contribuciones = pd.Series(shap_row, index=feature_values.index)
    orden = contribuciones.abs().sort_values(ascending=False).head(top).index
    restante = contribuciones.drop(orden).sum()
    etiquetas: list[str] = []
    valores: list[float] = []
    hover_vals: list[str] = []
    for f in orden:
        etiquetas.append(f"{nombre_legible(f)} = {feature_values[f]:.4g}")
        valores.append(float(contribuciones[f]))
        hover_vals.append(f"{nombre_legible(f)}<br>valor: {feature_values[f]:.4g}<br>contribución: {contribuciones[f]:+.4f}")
    if abs(restante) > 1e-9:
        etiquetas.append(f"Otras {len(contribuciones) - top} variables")
        valores.append(float(restante))
        hover_vals.append(f"Suma del resto de variables<br>contribución: {restante:+.4f}")
    colores = [BRAND["highlight"] if v > 0 else BRAND["primary"] for v in valores]
    fig = go.Figure(
        go.Bar(
            x=valores,
            y=etiquetas,
            orientation="h",
            marker_color=colores,
            text=[f"{v:+.3f}" for v in valores],
            textposition="outside",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_vals,
        )
    )
    fig.add_vline(x=0, line_color=BRAND["muted"], line_dash="dash", line_width=1)
    fig.update_layout(
        xaxis_title=f"Contribución SHAP (base value = {expected_value:+.4f})",
        yaxis=dict(categoryorder="total ascending"),
        showlegend=False,
        margin=dict(l=10, r=40, t=10, b=10),
    )
    return fig


def beeswarm_mpl(shap_values: np.ndarray, X_sample: pd.DataFrame, max_display: int = 20) -> plt.Figure:
    X_es = X_sample.copy()
    X_es.columns = [nombre_legible(c) for c in X_es.columns]
    shap.summary_plot(
        shap_values,
        X_es,
        max_display=max_display,
        show=False,
        color_bar_label="Valor de la variable",
    )
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    return fig
