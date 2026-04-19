import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

from src.configurar_logging import get_logger
from src.traducciones import NOMBRE_A_ISO3
from src.agrupacion_categorias import conteo
from src.ui.tema import BRAND, eyebrow, lead
from src.ui.graficos import chart_block, mostrar_grafico

log = get_logger()


def render(df: pd.DataFrame) -> None:
    log.info("Render Geografia inicio | df shape=%s", df.shape)
    eyebrow("04 / GEOGRAFÍA")
    st.header("Geografía y clientes")
    lead("Procedencia de los huéspedes y perfil de la clientela por tipo y canal de distribución.")

    por_pais = conteo(df["country"], columna_recuento="reservas")
    mapa_df = por_pais.assign(iso3=por_pais["country"].map(NOMBRE_A_ISO3))
    sin_iso3 = mapa_df.loc[mapa_df["iso3"].isna(), "country"].tolist()
    if sin_iso3:
        log.info("Geografia: países sin ISO3 (no se muestran en el mapa): %s", sin_iso3)
    mapa_df = mapa_df.dropna(subset=["iso3"])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Top 10 países de origen")
        with chart_block("Top 10 países"):
            top10 = por_pais.head(10).sort_values("reservas")
            fig = px.bar(
                top10, x="reservas", y="country", orientation="h",
                text="reservas",
                labels={"reservas": "Reservas", "country": "País"},
                color_discrete_sequence=[BRAND["primary"]],
            )
            mostrar_grafico(fig)

    with col2:
        st.subheader("Reservas por país (mapa mundial)")
        with chart_block("Mapa de países"):
            mapa = mapa_df.assign(log_reservas=np.log10(mapa_df["reservas"]))
            fig = px.choropleth(
                mapa,
                locations="iso3",
                locationmode="ISO-3",
                color="log_reservas",
                color_continuous_scale="Blues",
                hover_name="country",
                hover_data={"reservas": True, "log_reservas": False, "iso3": False},
                labels={"log_reservas": "log₁₀ reservas", "reservas": "Reservas"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            mostrar_grafico(fig)

    st.divider()
    st.subheader("Perfil de los clientes")

    with chart_block("Métricas de perfil"):
        pct_repetidores = df["is_repeated_guest"].mean() * 100
        media_peticiones = df["total_of_special_requests"].mean()
        media_cambios = df["booking_changes"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("% de huéspedes repetidores", f"{pct_repetidores:.2f}%")
        c2.metric("Media de peticiones especiales", f"{media_peticiones:.2f}")
        c3.metric("Cambios medios sobre la reserva", f"{media_cambios:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tipo de cliente")
        with chart_block("Tipo de cliente"):
            cust = conteo(df["customer_type"], columna_recuento="reservas")
            fig = px.pie(
                cust, names="customer_type", values="reservas", hole=0.4,
                labels={"customer_type": "Tipo de cliente", "reservas": "Reservas"},
                color_discrete_sequence=[BRAND["primary"], BRAND["accent"], BRAND["highlight"], BRAND["neutral"]],
            )
            mostrar_grafico(fig)

    with col2:
        st.subheader("Canal de distribución")
        with chart_block("Canal de distribución"):
            canal = conteo(df["distribution_channel"], columna_recuento="reservas")
            fig = px.bar(
                canal, x="distribution_channel", y="reservas",
                text="reservas",
                labels={"distribution_channel": "Canal de distribución", "reservas": "Reservas"},
                color_discrete_sequence=[BRAND["accent"]],
            )
            mostrar_grafico(fig)

    log.info("Render Geografia OK")
