import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import AÑO_REFERENCIA
from src.configurar_logging import get_logger
from src.traducciones import MESES_ORDEN_ES
from src.ui.tema import BRAND, eyebrow, lead
from src.ui.graficos import chart_block, mostrar_grafico

log = get_logger()


def render(df: pd.DataFrame) -> None:
    log.info("Render Estacionalidad inicio | df shape=%s", df.shape)
    eyebrow("03 / ESTACIONALIDAD")
    st.header("Estacionalidad de la demanda")
    lead(f"Cómo se distribuye la demanda a lo largo del año {AÑO_REFERENCIA} y cómo evoluciona el precio medio por noche.")

    df_año = df[df["arrival_date_year"] == AÑO_REFERENCIA]
    if df_año.empty:
        st.warning(f"No hay datos para el año de referencia ({AÑO_REFERENCIA}). Ajusta el dataset o la constante AÑO_REFERENCIA.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reservas por mes")
        with chart_block("Reservas por mes"):
            por_mes = (
                df_año.groupby("arrival_date_month", observed=True)
                .size()
                .reindex(MESES_ORDEN_ES)
                .reset_index(name="reservas")
            )
            fig = px.line(
                por_mes, x="arrival_date_month", y="reservas",
                markers=True,
                labels={"arrival_date_month": "Mes de llegada", "reservas": "Reservas"},
            )
            fig.update_traces(line_color=BRAND["primary"])
            mostrar_grafico(fig)

    with col2:
        st.subheader("Precio medio por noche según el mes")
        with chart_block("Precio medio por noche según el mes"):
            adr_mes = (
                df_año.groupby("arrival_date_month", observed=True)["adr"]
                .mean()
                .reindex(MESES_ORDEN_ES)
                .reset_index()
            )
            fig = px.line(
                adr_mes, x="arrival_date_month", y="adr",
                markers=True,
                labels={"arrival_date_month": "Mes de llegada", "adr": "Precio medio por noche (€)"},
            )
            fig.update_traces(line_color=BRAND["accent"])
            mostrar_grafico(fig)

    st.subheader("Reservas por semana del año")
    with chart_block("Reservas por semana"):
        por_semana = (
            df_año.groupby("arrival_date_week_number").size().reset_index(name="reservas")
        )
        fig = px.bar(
            por_semana, x="arrival_date_week_number", y="reservas",
            labels={"arrival_date_week_number": "Semana del año", "reservas": "Reservas"},
        )
        fig.update_traces(marker_color=BRAND["primary"])
        fig.update_xaxes(type="linear", dtick=4)
        mostrar_grafico(fig)

    log.info("Render Estacionalidad OK")
