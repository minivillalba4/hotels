from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.configurar_logging import get_logger
from src.ml.features import construir_features_modelo

log = get_logger()


_MESES_EN: list[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


HOTELES: list[str] = ["City Hotel", "Resort Hotel"]
DEPOSITOS: list[str] = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMER_TYPES: list[str] = ["Transient", "Transient-Party", "Contract", "Group"]
MEALS: list[str] = ["BB", "HB", "FB", "SC", "Undefined"]
MARKET_SEGMENTS: list[str] = [
    "Online TA", "Offline TA/TO", "Direct", "Corporate",
    "Complementary", "Groups", "Aviation", "Undefined",
]
DISTRIBUTION_CHANNELS: list[str] = ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"]
ROOM_TYPES: list[str] = list("ABCDEFGHIKLP")

PAIS_OTRO = "Otro"

# Columnas que esperamos construir para alimentar construir_features_modelo.
_COLS_RAW: list[str] = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "assigned_room_type", "booking_changes",
    "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces",
    "total_of_special_requests",
]


def _modal(serie: pd.Series, fallback: Any = None) -> Any:
    modas = serie.dropna().mode()
    if modas.empty:
        return fallback
    return modas.iloc[0]


def defaults_desde_df(df_raw: pd.DataFrame) -> dict:
    # Construye los defaults del simulador a partir del df sin traducir.
    log.info("Calculando defaults del simulador | df_raw shape=%s", df_raw.shape)

    hoy = date.today()
    try:
        año_mediana = int(df_raw["arrival_date_year"].median())
    except Exception:
        año_mediana = hoy.year

    fecha_default = date(año_mediana, 6, 15)

    defaults = {
        "hotel": _modal(df_raw["hotel"], HOTELES[0]),
        "fecha_llegada": fecha_default,
        "stays_in_weekend_nights": int(df_raw["stays_in_weekend_nights"].median()),
        "stays_in_week_nights": int(df_raw["stays_in_week_nights"].median()),
        "reserved_room_type": _modal(df_raw["reserved_room_type"], "A"),
        "misma_habitacion": True,
        "assigned_room_type": _modal(df_raw["assigned_room_type"], "A"),
        "deposit_type": _modal(df_raw["deposit_type"], "No Deposit"),
        "meal": _modal(df_raw["meal"], "BB"),
        "adults": int(df_raw["adults"].median()) or 2,
        "children": int(df_raw["children"].median()),
        "babies": int(df_raw["babies"].median()),
        "country": _modal(df_raw["country"], "PRT"),
        "customer_type": _modal(df_raw["customer_type"], "Transient"),
        "market_segment": _modal(df_raw["market_segment"], "Online TA"),
        "distribution_channel": _modal(df_raw["distribution_channel"], "TA/TO"),
        "is_repeated_guest": False,
        "has_company": False,
        "adr": float(df_raw["adr"].median()),
        "lead_time": int(df_raw["lead_time"].median()),
        "booking_changes": 0,
        "days_in_waiting_list": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
    }
    return defaults


def percentiles_desde_df(df_raw: pd.DataFrame) -> dict:
    # Percentiles p01/p99 usados para avisar de extrapolación.
    return {
        "adr_p99": float(df_raw["adr"].quantile(0.99)),
        "adr_p01": float(df_raw["adr"].quantile(0.01)),
        "lead_time_p99": float(df_raw["lead_time"].quantile(0.99)),
        "prev_canc_p99": float(df_raw["previous_cancellations"].quantile(0.99)),
        "prev_ok_p99": float(df_raw["previous_bookings_not_canceled"].quantile(0.99)),
        "booking_changes_p99": float(df_raw["booking_changes"].quantile(0.99)),
        "waiting_p99": float(df_raw["days_in_waiting_list"].quantile(0.99)),
    }


def paises_top(df_raw: pd.DataFrame, n: int = 30) -> list[str]:
    return df_raw["country"].value_counts().head(n).index.tolist()


def construir_presets(df_raw: pd.DataFrame) -> dict[str, dict]:
    # Tres perfiles calculados desde el dataset para no hardcodear.
    base = defaults_desde_df(df_raw)

    canceladas = df_raw[df_raw["is_canceled"] == True]  # noqa: E712
    confirmadas = df_raw[df_raw["is_canceled"] == False]  # noqa: E712

    riesgo = dict(base)
    if not canceladas.empty:
        riesgo.update({
            "deposit_type": "Non Refund" if "Non Refund" in canceladas["deposit_type"].values else base["deposit_type"],
            "lead_time": int(canceladas["lead_time"].quantile(0.75)),
            "adr": float(canceladas["adr"].quantile(0.75)),
            "previous_cancellations": int(max(1, canceladas["previous_cancellations"].quantile(0.90))),
            "booking_changes": 0,
            "total_of_special_requests": 0,
            "required_car_parking_spaces": 0,
            "is_repeated_guest": False,
            "market_segment": _modal(canceladas["market_segment"], base["market_segment"]),
            "distribution_channel": _modal(canceladas["distribution_channel"], base["distribution_channel"]),
        })

    seguro = dict(base)
    if not confirmadas.empty:
        seguro.update({
            "deposit_type": "No Deposit",
            "lead_time": int(confirmadas["lead_time"].quantile(0.25)),
            "adr": float(confirmadas["adr"].median()),
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": int(max(1, confirmadas["previous_bookings_not_canceled"].quantile(0.75))),
            "booking_changes": int(confirmadas["booking_changes"].quantile(0.75)) or 1,
            "total_of_special_requests": int(max(1, confirmadas["total_of_special_requests"].quantile(0.75))),
            "required_car_parking_spaces": 1,
            "is_repeated_guest": True,
        })

    return {
        "tipico": base,
        "riesgo": riesgo,
        "seguro": seguro,
    }


def validar_inputs(inputs: dict, percentiles: dict, paises_disponibles: list[str]) -> list[str]:
    # Devuelve una lista de mensajes legibles para mostrar como warnings.
    avisos: list[str] = []
    adr = float(inputs.get("adr", 0.0))
    lead = int(inputs.get("lead_time", 0))
    noches = int(inputs.get("stays_in_weekend_nights", 0)) + int(inputs.get("stays_in_week_nights", 0))
    adultos = int(inputs.get("adults", 0))
    pais = str(inputs.get("country", ""))

    if adr > percentiles["adr_p99"]:
        avisos.append(
            f"El ADR introducido ({adr:.0f} €) supera el percentil 99 del dataset "
            f"({percentiles['adr_p99']:.0f} €). El modelo extrapola y la predicción es menos fiable."
        )
    if lead > percentiles["lead_time_p99"]:
        avisos.append(
            f"La antelación ({lead} días) supera el percentil 99 del dataset "
            f"({percentiles['lead_time_p99']:.0f} días)."
        )
    if noches == 0:
        avisos.append("El total de noches es 0. Es una combinación poco representada y la predicción puede no reflejar una reserva real.")
    if adultos == 0:
        avisos.append("La reserva tiene 0 adultos. El modelo rara vez verá este caso durante el entrenamiento.")
    if pais == PAIS_OTRO or pais not in paises_disponibles:
        avisos.append(
            "País fuera de los 30 más frecuentes del dataset: el codificador lo trata como "
            "categoría nueva y la predicción es menos fiable para ese mercado concreto."
        )
    return avisos


def _fila_raw(inputs: dict) -> pd.DataFrame:
    fecha: date = inputs["fecha_llegada"]
    mes_en = _MESES_EN[fecha.month - 1]
    semana = int(fecha.isocalendar().week)

    reserved = inputs["reserved_room_type"]
    assigned = reserved if inputs.get("misma_habitacion", True) else inputs.get("assigned_room_type", reserved)

    pais = inputs["country"]

    fila = {
        "hotel": inputs["hotel"],
        "lead_time": int(inputs["lead_time"]),
        "arrival_date_year": int(fecha.year),
        "arrival_date_month": mes_en,
        "arrival_date_week_number": semana,
        "arrival_date_day_of_month": int(fecha.day),
        "stays_in_weekend_nights": int(inputs["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(inputs["stays_in_week_nights"]),
        "adults": int(inputs["adults"]),
        "children": int(inputs["children"]),
        "babies": int(inputs["babies"]),
        "meal": inputs["meal"],
        "country": pais,
        "market_segment": inputs["market_segment"],
        "distribution_channel": inputs["distribution_channel"],
        "is_repeated_guest": bool(inputs["is_repeated_guest"]),
        "previous_cancellations": int(inputs["previous_cancellations"]),
        "previous_bookings_not_canceled": int(inputs["previous_bookings_not_canceled"]),
        "reserved_room_type": reserved,
        "assigned_room_type": assigned,
        "booking_changes": int(inputs["booking_changes"]),
        "deposit_type": inputs["deposit_type"],
        "agent": 0,
        "company": 1 if inputs.get("has_company", False) else 0,
        "days_in_waiting_list": int(inputs["days_in_waiting_list"]),
        "customer_type": inputs["customer_type"],
        "adr": float(inputs["adr"]),
        "required_car_parking_spaces": int(inputs["required_car_parking_spaces"]),
        "total_of_special_requests": int(inputs["total_of_special_requests"]),
    }
    return pd.DataFrame([fila], columns=_COLS_RAW)


def predecir(
    inputs: dict,
    modelo: Any,
    preprocesador: Any,
    feature_names: list[str],
) -> tuple[float, pd.DataFrame]:
    fila_raw = _fila_raw(inputs)
    fila_modelo = construir_features_modelo(fila_raw)
    arr = preprocesador.transform(fila_modelo)
    fila_enc = pd.DataFrame(np.asarray(arr, dtype="float64"), columns=feature_names)
    proba = float(modelo.predict_proba(fila_enc)[0, 1])
    log.debug(
        "predecir | proba=%.4f hotel=%s lead=%s adr=%.1f pais=%s",
        proba, inputs.get("hotel"), inputs.get("lead_time"),
        float(inputs.get("adr", 0)), inputs.get("country"),
    )
    return proba, fila_enc
