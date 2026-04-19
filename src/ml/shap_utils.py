from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.configurar_logging import get_logger

log = get_logger()

RUTA_RAIZ = Path(__file__).resolve().parents[2]
RUTA_MODELO = RUTA_RAIZ / "models" / "clasificador.joblib"
RUTA_PREPROCESADOR = RUTA_RAIZ / "models" / "encoders" / "preprocesador.joblib"
RUTA_FEATURE_NAMES = RUTA_RAIZ / "models" / "encoders" / "feature_names.json"
RUTA_X_TEST = RUTA_RAIZ / "Data" / "ml" / "X_test.csv"
RUTA_Y_TEST = RUTA_RAIZ / "Data" / "ml" / "y_test.csv"

SEED_SAMPLE = 42


FEATURES_ES: dict[str, str] = {
    "lead_time": "Antelación (días)",
    "adr": "Tarifa media diaria (€)",
    "total_of_special_requests": "Solicitudes especiales",
    "previous_cancellations": "Cancelaciones previas",
    "previous_bookings_not_canceled": "Reservas previas OK",
    "booking_changes": "Cambios en la reserva",
    "required_car_parking_spaces": "Plazas de parking",
    "days_in_waiting_list": "Días en lista de espera",
    "total_nights": "Noches totales",
    "total_guests": "Huéspedes totales",
    "stays_in_weekend_nights": "Noches fin de semana",
    "stays_in_week_nights": "Noches entre semana",
    "adults": "Adultos",
    "children": "Niños",
    "babies": "Bebés",
    "arrival_date_month": "Mes de llegada",
    "arrival_date_year": "Año de llegada",
    "arrival_date_week_number": "Semana del año",
    "arrival_date_day_of_month": "Día del mes",
    "day_of_week": "Día de la semana",
    "is_weekend": "¿Fin de semana?",
    "month_sin": "Mes (sin)",
    "month_cos": "Mes (cos)",
    "dow_sin": "Día semana (sin)",
    "dow_cos": "Día semana (cos)",
    "country": "País",
    "agent": "Agencia (ID)",
    "is_repeated_guest": "Huésped repetidor",
    "has_company": "Asociado a empresa",
    "hotel_Resort Hotel": "Hotel vacacional",
    "deposit_type_Non Refund": "Depósito no reembolsable",
    "deposit_type_Refundable": "Depósito reembolsable",
    "customer_type_Group": "Cliente: grupo",
    "customer_type_Transient": "Cliente: individual",
    "customer_type_Transient-Party": "Cliente: ind. en grupo",
    "market_segment_Online TA": "Segmento: agencia online",
    "market_segment_Offline TA/TO": "Segmento: agencia física",
    "market_segment_Direct": "Segmento: directo",
    "market_segment_Corporate": "Segmento: corporativo",
    "market_segment_Groups": "Segmento: grupos",
    "market_segment_Complementary": "Segmento: cortesía",
    "market_segment_Undefined": "Segmento: sin especificar",
    "distribution_channel_Direct": "Canal: directo",
    "distribution_channel_TA/TO": "Canal: agencias",
    "distribution_channel_GDS": "Canal: GDS",
    "distribution_channel_Undefined": "Canal: sin especificar",
    "meal_BB": "Régimen: desayuno",
    "meal_HB": "Régimen: media pensión",
    "meal_FB": "Régimen: pensión completa",
    "meal_SC": "Régimen: sin comidas",
    "meal_Undefined": "Régimen: sin especificar",
}


def nombre_legible(feature: str) -> str:
    return FEATURES_ES.get(feature, feature)


@st.cache_resource(show_spinner=False)
def cargar_modelo() -> Any:
    log.info("Cargando modelo desde %s", RUTA_MODELO)
    modelo = joblib.load(RUTA_MODELO)
    log.info("Modelo cargado | tipo=%s", type(modelo).__name__)
    return modelo


@st.cache_resource(show_spinner=False)
def _cargar_preprocesador_joblib() -> Any:
    log.info("Cargando preprocesador desde %s", RUTA_PREPROCESADOR)
    return joblib.load(RUTA_PREPROCESADOR)


def cargar_preprocesador() -> Tuple[Any, list[str]]:
    preprocesador = _cargar_preprocesador_joblib()
    with open(RUTA_FEATURE_NAMES, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    log.info("Preprocesador listo | n_features=%d", len(feature_names))
    return preprocesador, feature_names


@st.cache_data(show_spinner=False)
def cargar_sample_test(n: int = 2000) -> Tuple[pd.DataFrame, pd.Series]:
    log.info("Cargando sample X_test | n=%d", n)
    X = pd.read_csv(RUTA_X_TEST)
    y = pd.read_csv(RUTA_Y_TEST)["is_canceled"]
    if len(X) > n:
        idx = X.sample(n=n, random_state=SEED_SAMPLE).index
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)
    log.info("Sample preparado | X=%s y=%s", X.shape, y.shape)
    return X, y


@st.cache_resource(show_spinner=False)
def _explainer_de(_modelo: Any) -> shap.TreeExplainer:
    log.info("Construyendo TreeExplainer")
    return shap.TreeExplainer(_modelo)


def obtener_explainer(_modelo: Any) -> shap.TreeExplainer:
    return _explainer_de(_modelo)


@st.cache_data(show_spinner=False)
def calcular_shap_global(
    _modelo: Any, X_sample: pd.DataFrame
) -> Tuple[np.ndarray, float, pd.Series]:
    log.info("Calculando SHAP global | sample=%s", X_sample.shape)
    explainer = _explainer_de(_modelo)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        ev_arr = np.asarray(expected_value).ravel()
        expected_value = float(ev_arr[1] if ev_arr.size >= 2 else ev_arr[-1])
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_sample.columns)
    mean_abs = mean_abs.sort_values(ascending=False)
    log.info("SHAP global listo | top feature=%s valor=%.4f", mean_abs.index[0], float(mean_abs.iloc[0]))
    return sv, float(expected_value), mean_abs


def explicar_reserva(_explainer: Any, X_row: pd.DataFrame) -> np.ndarray:
    sv = _explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv)[0]
