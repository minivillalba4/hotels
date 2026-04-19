from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.configurar_logging import get_logger
from src.ui.tema import BRAND

log = get_logger()


ESTADO_COLORES = {"Canceladas": BRAND["highlight"], "Confirmadas": BRAND["primary"]}
ESTADO_ORDEN = ["Confirmadas", "Canceladas"]

BOXPLOT_MARGIN = dict(l=48, r=80, t=48, b=150)
BOXPLOT_ANN_YSHIFT = -60
BOXPLOT_ANN_ANGLE = -30

_HEATMAP_TICKANGLE = -35
_HEATMAP_TEXTFONT = dict(family="IBM Plex Mono, monospace", size=11)


def asignar_estado(flags: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(flags, "Canceladas", "Confirmadas"),
        index=flags.index,
        name="estado",
    )


def forzar_object_si_cat(serie: pd.Series) -> pd.Series:
    if isinstance(serie.dtype, pd.CategoricalDtype):
        return serie.astype(object)
    return serie


def aplicar_estilo_heatmap(fig: go.Figure) -> None:
    fig.update_xaxes(tickangle=_HEATMAP_TICKANGLE)
    fig.update_traces(textfont=_HEATMAP_TEXTFONT)


def mostrar_grafico(fig: go.Figure, *, key: str | None = None) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


@contextmanager
def chart_block(label: str):
    try:
        yield
    except Exception as e:
        log.exception("Error en gráfico '%s'", label)
        st.error(f"Error generando '{label}' ({type(e).__name__}).")


def fmt_es(x: float, decimales: int = 2) -> str:
    if pd.isna(x):
        return "—"
    if float(x).is_integer():
        s = f"{int(x):,}"
    else:
        s = f"{x:,.{decimales}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def stats_boxplot(serie: pd.Series) -> list[tuple[str, float]]:
    s = serie.dropna()
    if s.empty:
        return []
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    mn, mx = float(s.min()), float(s.max())
    return [
        ("Mín", mn),
        ("Bigote inf.", max(mn, q1 - 1.5 * iqr)),
        ("Q1", q1),
        ("Mediana", float(s.median())),
        ("Q3", q3),
        ("Bigote sup.", min(mx, q3 + 1.5 * iqr)),
        ("Máx", mx),
    ]


def aplicar_boxplot_es(fig: go.Figure, serie: pd.Series, etiqueta: str) -> go.Figure:
    stats = stats_boxplot(serie)
    if not stats:
        return fig
    vals = [v for _, v in stats]
    mn, mx = min(vals), max(vals)
    pad = max((mx - mn) * 0.07, 1.0)
    hover_lines = "<br>".join(f"{lbl}: {fmt_es(v)}" for lbl, v in stats)
    fig.update_traces(
        hovertemplate=f"<b>{etiqueta}</b><br>{hover_lines}<extra></extra>",
    )
    for lbl, val in stats:
        fig.add_annotation(
            x=val, y=0, xref="x", yref="paper",
            yshift=BOXPLOT_ANN_YSHIFT,
            text=f"{lbl}: {fmt_es(val)}",
            showarrow=False, textangle=BOXPLOT_ANN_ANGLE,
            font=dict(size=10, color=BRAND["muted"]),
            xanchor="center", yanchor="top",
        )
    fig.update_layout(showlegend=False, margin=BOXPLOT_MARGIN, xaxis_title=None)
    fig.update_xaxes(range=[mn - pad, mx + pad])
    return fig
