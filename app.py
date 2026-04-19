from pathlib import Path

import streamlit as st

from src.configurar_logging import get_logger
from src.carga_datos import cargar_datos
from src.pestañas import (
    resumen,
    cancelaciones,
    estacionalidad,
    geografia,
    analisis_exploratorio,
    simulador,
    interpretabilidad,
)
from src.ui.contacto import CONTACTO, render_footer
from src.ui.tema import inject_theme

log = get_logger()

LOGO_PATH = str(Path(__file__).parent / "docs" / "Booking.com_logo.svg.png")

_autor = CONTACTO.get("nombre") or ""
_page_title = f"Hotel Bookings · {_autor}" if _autor else "Hotel Bookings Dashboard"

st.set_page_config(
    page_title=_page_title,
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.logo(LOGO_PATH, size="large")
inject_theme()

log.info("App iniciada")

try:
    df = cargar_datos()
    log.info("Datos cargados: %d filas x %d columnas", df.shape[0], df.shape[1])
except Exception:
    log.exception("Error cargando datos")
    st.error("No se pudieron cargar los datos. Revisa logs/app.log.")
    st.stop()


def _render(nombre, fn):
    try:
        fn(df)
    except Exception:
        log.exception("Error renderizando pestaña '%s'", nombre)
        st.error(f"Error en pestaña '{nombre}'. Revisa logs/app.log.")


tab_home, tab_canc, tab_estac, tab_geo, tab_analysis, tab_sim, tab_ml = st.tabs([
    "Inicio",
    "Cancelaciones",
    "Estacionalidad",
    "Geografía y clientes",
    "Análisis exploratorio",
    "Simulador",
    "Interpretabilidad",
])

with tab_home:
    _render("Inicio", resumen.render)
with tab_canc:
    _render("Cancelaciones", cancelaciones.render)
with tab_estac:
    _render("Estacionalidad", estacionalidad.render)
with tab_geo:
    _render("Geografía y clientes", geografia.render)
with tab_analysis:
    _render("Análisis exploratorio", analisis_exploratorio.render)
with tab_sim:
    _render("Simulador", simulador.render)
with tab_ml:
    _render("Interpretabilidad", interpretabilidad.render)

render_footer()
