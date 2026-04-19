import pandas as pd

from src.config import MIN_RESERVAS_AGRUPACION
from src.configurar_logging import get_logger

log = get_logger()


def tasa_cancelacion(df: pd.DataFrame) -> float:
    return float(df["is_canceled"].mean() * 100)


def tasa_por(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tabla = (
        df.groupby(col, observed=True)["is_canceled"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "tasa_cancelacion", "size": "reservas"})
    )
    tabla = tabla[tabla["reservas"] >= MIN_RESERVAS_AGRUPACION]
    tabla["tasa_cancelacion"] = (tabla["tasa_cancelacion"] * 100).round(2)
    return tabla.sort_values("reservas", ascending=False)


def kpis_globales(df: pd.DataFrame) -> dict:
    requeridas = {"arrival_date_year", "is_canceled", "adr", "country", "hotel"}
    faltan = requeridas - set(df.columns)
    if faltan:
        log.error("kpis_globales: faltan columnas %s", faltan)
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltan)}")
    if df.empty:
        raise ValueError("El DataFrame está vacío (0 filas).")

    años = df["arrival_date_year"]
    kpis = {
        "total_reservas": int(len(df)),
        "n_hoteles": int(df["hotel"].nunique()),
        "año_min": int(años.min()),
        "año_max": int(años.max()),
        "tasa_cancelacion": tasa_cancelacion(df),
        "adr_medio": float(df["adr"].mean()),
        "n_paises": int(df["country"].nunique(dropna=True)),
    }
    log.debug("kpis_globales: %s", kpis)
    return kpis
