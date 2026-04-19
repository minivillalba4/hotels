from pathlib import Path

import pandas as pd
import streamlit as st

from src.configurar_logging import get_logger
from src.traducciones import MESES_ORDEN_ES, traducir_mes, traducir_valores

log = get_logger()

CSV_PATH: Path = Path(__file__).resolve().parents[1] / "Data" / "clean" / "df_eda.csv"

_COLS_BOOL: tuple[str, ...] = ("is_canceled", "is_repeated_guest")


def _castear_booleanos(df: pd.DataFrame) -> pd.DataFrame:
    for col in _COLS_BOOL:
        if col in df.columns and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(bool)
    return df


def _normalizar_meses(df: pd.DataFrame) -> pd.DataFrame:
    if "arrival_date_month" not in df.columns:
        log.warning("cargar_datos: falta columna 'arrival_date_month'")
        return df
    serie = traducir_mes(df["arrival_date_month"])
    meses_no_esperados = set(serie.dropna().unique()).difference(MESES_ORDEN_ES)
    if meses_no_esperados:
        log.warning("cargar_datos: meses no reconocidos en arrival_date_month: %s", meses_no_esperados)
    df["arrival_date_month"] = pd.Categorical(serie, categories=MESES_ORDEN_ES, ordered=True)
    return df


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    log.debug("cargar_datos: leyendo %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.debug("cargar_datos: CSV cargado shape=%s", df.shape)

    df = _castear_booleanos(df)
    df = traducir_valores(df)
    df = _normalizar_meses(df)
    return df


@st.cache_data
def cargar_datos_raw() -> pd.DataFrame:
    log.debug("cargar_datos_raw: leyendo %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.debug("cargar_datos_raw: CSV cargado shape=%s", df.shape)
    df = _castear_booleanos(df)
    return df
