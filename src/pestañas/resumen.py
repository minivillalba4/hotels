import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import AÑO_REFERENCIA
from src.configurar_logging import get_logger
from src.kpis import kpis_globales
from src.traducciones import LABELS_ES
from src.ui.contacto import render_hero
from src.ui.tema import BRAND, kpi_grid, lead, section_meta
from src.ui.graficos import chart_block, fmt_es, mostrar_grafico

log = get_logger()


def render(df: pd.DataFrame) -> None:
    log.info("Render Resumen inicio | df shape=%s", df.shape)
    k = kpis_globales(df)

    render_hero()

    lead(
        "Este dashboard es un caso de estudio del portfolio: análisis exploratorio, "
        "modelo de cancelaciones con LightGBM e interpretabilidad SHAP sobre datos "
        "reales de reservas hoteleras."
    )

    section_meta(
        f"Datos · {k['año_min']}–{k['año_max']}",
        f"{fmt_es(k['total_reservas'])} reservas",
    )

    kpi_grid([
        ("Reservas totales", fmt_es(k["total_reservas"])),
        ("Hoteles", str(k["n_hoteles"])),
        ("Rango de años", f"{k['año_min']}–{k['año_max']}"),
        ("Tasa de cancelación", f"{k['tasa_cancelacion']:.1f}", "%"),
        ("Precio medio por noche", fmt_es(k["adr_medio"]), "€"),
        ("Países de origen", f"{k['n_paises']}"),
    ])

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Reservas por año")
        with chart_block("Reservas por año"):
            por_año = (
                df.groupby("arrival_date_year").size().reset_index(name="reservas")
            )
            fig = px.bar(
                por_año,
                x="arrival_date_year",
                y="reservas",
                text_auto=True,
                labels={"arrival_date_year": "Año de llegada", "reservas": "Reservas"},
            )
            fig.update_traces(marker_color=BRAND["primary"])
            fig.update_xaxes(type="category")
            mostrar_grafico(fig)

    with col_right:
        st.subheader("Datos destacados")

        for etq, col in (
            ("Hotel más reservado", "hotel"),
            (f"Mes pico ({AÑO_REFERENCIA})", "arrival_date_month"),
            ("País con más reservas", "country"),
        ):
            with chart_block(etq):
                if col not in df.columns:
                    st.markdown(f"**{etq}**  \nColumna ausente")
                    continue
                datos = df[df["arrival_date_year"] == AÑO_REFERENCIA] if col == "arrival_date_month" else df
                conteos = datos[col].value_counts(dropna=True)
                if conteos.empty:
                    st.markdown(f"**{etq}**  \nSin datos")
                    continue
                top = conteos.idxmax()
                n_top = int(conteos.max())
                st.markdown(f"**{etq}**  \n{top} - {fmt_es(n_top)} reservas")

    with st.expander("Vista previa del conjunto de datos"):
        st.caption("Primeros 20 registros del dataset completo")
        preview = df.head(20).rename(columns=LABELS_ES)
        st.dataframe(preview, width="stretch")

    log.info("Render Resumen OK")
