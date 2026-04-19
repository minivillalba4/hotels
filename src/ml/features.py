from __future__ import annotations

import numpy as np
import pandas as pd

from src.configurar_logging import get_logger

log = get_logger()

_COLS_LEAKAGE = ("reservation_status", "reservation_status_date")
_ORDEN_MESES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def construir_features_modelo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in _COLS_LEAKAGE:
        if col in out.columns:
            out = out.drop(columns=col)
    if "is_canceled" in out.columns:
        out = out.drop(columns="is_canceled")

    if "company" in out.columns:
        out["has_company"] = (out["company"] != 0).astype("int8")
        out = out.drop(columns="company")
    else:
        log.warning("construir_features_modelo: falta 'company'; has_company=0 por defecto")
        out["has_company"] = np.int8(0)

    if "is_repeated_guest" in out.columns:
        out["is_repeated_guest"] = out["is_repeated_guest"].astype("bool")

    meses_a_num = {m: i for i, m in enumerate(_ORDEN_MESES, start=1)}
    valores = out["arrival_date_month"].dropna().unique()
    invalidos = sorted(set(valores) - set(meses_a_num))
    if invalidos:
        raise ValueError(
            "construir_features_modelo: valores de 'arrival_date_month' no reconocidos "
            f"(se esperan meses en inglés): {invalidos[:5]}"
        )
    mes_num = out["arrival_date_month"].map(meses_a_num).astype("int64")
    arrival_date = pd.to_datetime({
        "year":  out["arrival_date_year"],
        "month": mes_num,
        "day":   out["arrival_date_day_of_month"],
    })
    out["day_of_week"]  = arrival_date.dt.dayofweek.astype("int8")
    out["is_weekend"]   = out["day_of_week"].isin([5, 6]).astype("int8")
    out["month_sin"]    = np.sin(2 * np.pi * mes_num / 12)
    out["month_cos"]    = np.cos(2 * np.pi * mes_num / 12)
    out["dow_sin"]      = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"]      = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["total_guests"] = out["adults"] + out["children"] + out["babies"]

    return out
