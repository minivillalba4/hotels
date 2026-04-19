from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.carga_datos import cargar_datos_raw
from src.configurar_logging import get_logger
from src.ml.shap_utils import cargar_modelo, cargar_preprocesador
from src.ml.simulador import (
    CUSTOMER_TYPES,
    DEPOSITOS,
    DISTRIBUTION_CHANNELS,
    HOTELES,
    MARKET_SEGMENTS,
    MEALS,
    PAIS_OTRO,
    ROOM_TYPES,
    construir_presets,
    defaults_desde_df,
    paises_top,
    percentiles_desde_df,
    predecir,
    validar_inputs,
)
from src.traducciones import PAISES_ES, VALORES_ES
from src.ui.graficos import chart_block, fmt_es, mostrar_grafico
from src.ui.tema import BRAND, eyebrow, lead

log = get_logger()


_PREFIJO = "sim_"

_CAMPOS_PRESET: list[str] = [
    "hotel", "stays_in_weekend_nights", "stays_in_week_nights",
    "reserved_room_type", "misma_habitacion", "assigned_room_type",
    "deposit_type", "meal", "adults", "children", "babies",
    "country", "customer_type", "market_segment", "distribution_channel",
    "is_repeated_guest", "has_company",
    "adr", "lead_time", "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
    "previous_cancellations", "previous_bookings_not_canceled",
]


@st.cache_data(show_spinner=False)
def _defaults_cached(df_raw: pd.DataFrame) -> dict:
    return defaults_desde_df(df_raw)


@st.cache_data(show_spinner=False)
def _percentiles_cached(df_raw: pd.DataFrame) -> dict:
    return percentiles_desde_df(df_raw)


@st.cache_data(show_spinner=False)
def _presets_cached(df_raw: pd.DataFrame) -> dict[str, dict]:
    return construir_presets(df_raw)


@st.cache_data(show_spinner=False)
def _paises_cached(df_raw: pd.DataFrame) -> list[str]:
    return paises_top(df_raw, n=30)


def _k(nombre: str) -> str:
    return f"{_PREFIJO}{nombre}"


def _inicializar_estado(defaults: dict) -> None:
    for campo in _CAMPOS_PRESET:
        clave = _k(campo)
        if clave not in st.session_state:
            st.session_state[clave] = defaults[campo]
    if _k("fecha_llegada") not in st.session_state:
        st.session_state[_k("fecha_llegada")] = defaults["fecha_llegada"]


def _aplicar_preset(preset: dict, nombre: str) -> None:
    for campo in _CAMPOS_PRESET:
        if campo in preset:
            st.session_state[_k(campo)] = preset[campo]
    st.session_state["sim_ultimo_preset"] = nombre


def _etiqueta_pais(iso: str) -> str:
    if iso == PAIS_OTRO:
        return "Otro (fuera del top 30)"
    return f"{PAISES_ES.get(iso, iso)} ({iso})"


def _etiqueta_habitacion(cod: str) -> str:
    return f"Tipo {cod}"


def _valor_es(col: str, valor: str) -> str:
    return VALORES_ES.get(col, {}).get(valor, valor)


def _bullet_chart(proba: float) -> go.Figure:
    pct = proba * 100
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=pct,
        number={
            "suffix": " %",
            "valueformat": ".1f",
            "font": {"size": 30, "color": BRAND["ink"]},
        },
        gauge={
            "shape": "bullet",
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 30, 50, 80, 100],
                "tickfont": {"color": BRAND["muted"], "size": 11},
                "ticksuffix": " %",
            },
            "bar": {"color": BRAND["ink"], "thickness": 0.45},
            "steps": [
                {"range": [0, 30], "color": "#cfe4c9"},
                {"range": [30, 50], "color": "#f6e0a6"},
                {"range": [50, 100], "color": "#e8b8b3"},
            ],
            "threshold": {
                "line": {"color": BRAND["ink"], "width": 2},
                "thickness": 0.85,
                "value": 50,
            },
            "borderwidth": 0,
        },
        domain={"x": [0.0, 1.0], "y": [0.25, 0.75]},
    ))
    fig.add_annotation(
        x=0.08, y=0.95, xref="paper", yref="paper",
        text="<b>BAJO</b>", showarrow=False,
        font={"size": 10, "color": BRAND["muted"], "family": "IBM Plex Mono, monospace"},
    )
    fig.add_annotation(
        x=0.27, y=0.95, xref="paper", yref="paper",
        text="<b>MEDIO</b>", showarrow=False,
        font={"size": 10, "color": BRAND["muted"], "family": "IBM Plex Mono, monospace"},
    )
    fig.add_annotation(
        x=0.58, y=0.95, xref="paper", yref="paper",
        text="<b>ALTO</b>", showarrow=False,
        font={"size": 10, "color": BRAND["muted"], "family": "IBM Plex Mono, monospace"},
    )
    fig.update_layout(height=150, margin=dict(l=20, r=30, t=30, b=20))
    return fig


def render(df: pd.DataFrame) -> None:
    log.info("Render Simulador inicio | df shape=%s", df.shape)
    eyebrow("06 / SIMULADOR")
    st.header("Simulador de cancelación")
    lead(
        "Ajusta los parámetros de una reserva hipotética y observa cómo el modelo "
        "LightGBM estima la probabilidad de cancelación en tiempo real."
    )

    try:
        df_raw = cargar_datos_raw()
        modelo = cargar_modelo()
        preprocesador, feature_names = cargar_preprocesador()
    except Exception:
        log.exception("Error cargando modelo/preprocesador/datos raw para simulador")
        st.error("No se pudieron cargar modelo o preprocesador. Revisa logs/app.log.")
        return

    defaults = _defaults_cached(df_raw)
    percentiles = _percentiles_cached(df_raw)
    presets = _presets_cached(df_raw)
    paises_disponibles = _paises_cached(df_raw)

    _inicializar_estado(defaults)

    col_p1, col_p2, col_p3, _sp = st.columns([1, 1, 1, 2])
    if col_p1.button("Caso típico", use_container_width=True, key="sim_btn_tipico"):
        _aplicar_preset(presets["tipico"], "típico")
        st.rerun()
    if col_p2.button("Reserva de riesgo", use_container_width=True, key="sim_btn_riesgo"):
        _aplicar_preset(presets["riesgo"], "de riesgo")
        st.rerun()
    if col_p3.button("Reserva segura", use_container_width=True, key="sim_btn_segura"):
        _aplicar_preset(presets["seguro"], "segura")
        st.rerun()

    ultimo = st.session_state.get("sim_ultimo_preset")
    if ultimo:
        st.info(
            f"Aplicado preset **{ultimo}**: "
            f"depósito {_valor_es('deposit_type', st.session_state[_k('deposit_type')])} · "
            f"antelación {st.session_state[_k('lead_time')]} días · "
            f"ADR {fmt_es(st.session_state[_k('adr')], 0)} € · "
            f"canal {_valor_es('distribution_channel', st.session_state[_k('distribution_channel')])} · "
            f"cancelaciones previas {st.session_state[_k('previous_cancellations')]}."
        )

    col_izq, col_der = st.columns([2, 3], gap="large")

    with col_izq:
        with st.expander("Reserva", expanded=True):
            st.selectbox(
                "Tipo de hotel",
                options=HOTELES,
                format_func=lambda v: _valor_es("hotel", v),
                key=_k("hotel"),
            )
            st.date_input(
                "Fecha de llegada",
                key=_k("fecha_llegada"),
                min_value=date(2014, 1, 1),
                max_value=date(2030, 12, 31),
            )
            st.slider("Noches entre semana", 0, 14, key=_k("stays_in_week_nights"))
            st.slider("Noches de fin de semana", 0, 6, key=_k("stays_in_weekend_nights"))
            st.selectbox(
                "Habitación reservada",
                options=ROOM_TYPES,
                format_func=_etiqueta_habitacion,
                key=_k("reserved_room_type"),
            )
            st.checkbox(
                "Asignar la misma habitación que la reservada",
                key=_k("misma_habitacion"),
            )
            if not st.session_state[_k("misma_habitacion")]:
                st.selectbox(
                    "Habitación asignada",
                    options=ROOM_TYPES,
                    format_func=_etiqueta_habitacion,
                    key=_k("assigned_room_type"),
                )
            st.selectbox(
                "Tipo de depósito",
                options=DEPOSITOS,
                format_func=lambda v: _valor_es("deposit_type", v),
                key=_k("deposit_type"),
            )
            st.selectbox(
                "Régimen de comidas",
                options=MEALS,
                format_func=lambda v: _valor_es("meal", v),
                key=_k("meal"),
            )

        with st.expander("Cliente"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Adultos", 0, 10, key=_k("adults"))
            with c2:
                st.number_input("Niños", 0, 10, key=_k("children"))
            with c3:
                st.number_input("Bebés", 0, 10, key=_k("babies"))

            opciones_pais = paises_disponibles + [PAIS_OTRO]
            st.selectbox(
                "País",
                options=opciones_pais,
                format_func=_etiqueta_pais,
                key=_k("country"),
            )
            st.selectbox(
                "Tipo de cliente",
                options=CUSTOMER_TYPES,
                format_func=lambda v: _valor_es("customer_type", v),
                key=_k("customer_type"),
            )
            st.selectbox(
                "Segmento de mercado",
                options=MARKET_SEGMENTS,
                format_func=lambda v: _valor_es("market_segment", v),
                key=_k("market_segment"),
            )
            st.selectbox(
                "Canal de distribución",
                options=DISTRIBUTION_CHANNELS,
                format_func=lambda v: _valor_es("distribution_channel", v),
                key=_k("distribution_channel"),
            )
            st.toggle("Cliente recurrente", key=_k("is_repeated_guest"))
            st.toggle("Reserva asociada a empresa", key=_k("has_company"))

        with st.expander("Contexto económico y operativo"):
            st.slider(
                "Precio medio por noche (ADR) €",
                0.0, 500.0, step=1.0, key=_k("adr"),
            )
            st.slider(
                "Antelación (lead time) en días",
                0, 700, step=1, key=_k("lead_time"),
            )
            st.number_input("Cambios sobre la reserva", 0, 30, key=_k("booking_changes"))
            st.number_input("Días en lista de espera", 0, 400, key=_k("days_in_waiting_list"))
            st.number_input("Plazas de parking solicitadas", 0, 3, key=_k("required_car_parking_spaces"))
            st.number_input("Peticiones especiales", 0, 5, key=_k("total_of_special_requests"))

        with st.expander("Historial de la reserva"):
            st.caption(
                "Estos valores describen el historial registrado en la reserva, "
                "no una verificación de identidad de un cliente específico."
            )
            st.number_input(
                "Cancelaciones previas",
                0, 100, key=_k("previous_cancellations"),
            )
            st.number_input(
                "Reservas previas no canceladas",
                0, 100, key=_k("previous_bookings_not_canceled"),
            )

    inputs = {campo: st.session_state[_k(campo)] for campo in _CAMPOS_PRESET}
    inputs["fecha_llegada"] = st.session_state[_k("fecha_llegada")]

    if inputs["fecha_llegada"] is None:
        with col_der:
            st.warning("Selecciona una fecha de llegada para calcular la predicción.")
        return

    try:
        proba, _fila_enc = predecir(inputs, modelo, preprocesador, feature_names)
    except Exception:
        log.exception("Error en predecir() del simulador")
        with col_der:
            st.error("Fallo al calcular la predicción. Revisa logs/app.log.")
        return

    clase = "Cancelada" if proba >= 0.5 else "Confirmada"

    with col_der:
        with chart_block("Bullet de probabilidad"):
            mostrar_grafico(_bullet_chart(proba), key="sim_bullet")

        m1, m2 = st.columns(2)
        m1.metric("Predicción del modelo", clase)
        m2.metric(
            "Probabilidad de cancelación",
            f"{proba * 100:.1f} %".replace(".", ","),
        )

        avisos = validar_inputs(inputs, percentiles, paises_disponibles)
        for aviso in avisos:
            st.warning(aviso)

        st.caption(
            "Modelo: LightGBM · predicción reactiva sobre el preprocesador serializado. "
            "Para la descomposición por factores (SHAP) ver la pestaña Interpretabilidad."
        )

    log.info(
        "Render Simulador OK | proba=%.4f clase=%s preset=%s",
        proba, clase, st.session_state.get("sim_ultimo_preset"),
    )
