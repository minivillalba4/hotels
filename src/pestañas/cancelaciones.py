import pandas as pd
import streamlit as st
import plotly.express as px

from src.configurar_logging import get_logger
from src.kpis import tasa_cancelacion, tasa_por
from src.traducciones import MESES_ORDEN_ES
from src.ui.tema import BRAND, eyebrow, lead
from src.ui.graficos import (
    ESTADO_COLORES,
    asignar_estado,
    chart_block,
    fmt_es,
    mostrar_grafico,
)

log = get_logger()

_NBINS_LEAD_TIME = 60


def _barra_tasa(
    df: pd.DataFrame,
    col: str,
    label_x: str,
    paleta: list[str],
    color_col: str | None = None,
) -> None:
    datos = tasa_por(df, col)
    fig = px.bar(
        datos,
        x=col, y="tasa_cancelacion",
        text="tasa_cancelacion",
        color=color_col,
        color_discrete_sequence=paleta,
        labels={"tasa_cancelacion": "% canceladas", col: label_x},
    )
    fig.update_traces(texttemplate="%{text:.1f}%")
    mostrar_grafico(fig)


def render(df: pd.DataFrame) -> None:
    log.info("Render Cancelaciones inicio | df shape=%s", df.shape)
    eyebrow("02 / CANCELACIONES")
    st.header("Análisis de cancelaciones")
    lead("Qué reservas se caen, cuándo se caen, y qué variables anticipan el riesgo de cancelación.")

    tasa = tasa_cancelacion(df)
    canceladas = int(df["is_canceled"].sum())
    no_canceladas = int((~df["is_canceled"]).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Tasa global de cancelación", f"{fmt_es(tasa)}%")
    c2.metric("Reservas canceladas", fmt_es(canceladas))
    c3.metric("Reservas confirmadas", fmt_es(no_canceladas))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Por tipo de hotel")
        with chart_block("Por tipo de hotel"):
            _barra_tasa(df, "hotel", "Tipo de hotel",
                        paleta=[BRAND["primary"], BRAND["accent"]],
                        color_col="hotel")
    with col2:
        st.subheader("Por tipo de depósito")
        with chart_block("Por tipo de depósito"):
            _barra_tasa(df, "deposit_type", "Tipo de depósito",
                        paleta=[BRAND["highlight"]])

    st.subheader("Por segmento de mercado")
    with chart_block("Por segmento de mercado"):
        _barra_tasa(df, "market_segment", "Segmento de mercado",
                    paleta=[BRAND["accent"]])

    st.subheader("Días de antelación: canceladas vs confirmadas")
    with chart_block("Días de antelación"):
        df_plot = df.assign(estado=asignar_estado(df["is_canceled"]))
        fig = px.histogram(
            df_plot, x="lead_time", color="estado",
            nbins=_NBINS_LEAD_TIME, barmode="stack",
            labels={"lead_time": "Días entre reserva y llegada", "estado": "Estado"},
            color_discrete_map=ESTADO_COLORES,
        )
        fig.update_yaxes(title="Reservas")
        mostrar_grafico(fig)

    st.subheader("Mapa de calor año × mes (tasa de cancelación)")
    heat = (
        df.groupby(["arrival_date_year", "arrival_date_month"], observed=True)["is_canceled"]
        .mean()
        .reset_index()
    )
    heat["is_canceled"] = (heat["is_canceled"] * 100).round(2)
    pivot = heat.pivot(
        index="arrival_date_year",
        columns="arrival_date_month",
        values="is_canceled",
    ).reindex(columns=MESES_ORDEN_ES)

    if pivot.empty or pivot.isna().all(axis=None):
        st.info("Sin datos para el mapa de calor.")
    else:
        with chart_block("Mapa de calor año × mes"):
            meses_con_datos = [m for m in pivot.columns if not pivot[m].isna().all()]
            pivot.index = pivot.index.astype(str)
            fig = px.imshow(
                pivot, text_auto=".1f", aspect="auto",
                color_continuous_scale="Blues",
                labels={"color": "% cancelación"},
            )
            fig.update_yaxes(type="category", title="Año de llegada")
            fig.update_xaxes(title="Mes de llegada")
            if meses_con_datos:
                st.caption(f"Meses con datos: {meses_con_datos[0]} – {meses_con_datos[-1]}.")
            mostrar_grafico(fig)

    log.info("Render Cancelaciones OK")
