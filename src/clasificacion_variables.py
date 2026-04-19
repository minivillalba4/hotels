import pandas as pd

from src.config import UMBRAL_CATEGORICA
from src.configurar_logging import get_logger

log = get_logger()

COLS_ID = {"agent", "company"}

COLS_CAT_FORZADAS = {
    "previous_cancellations",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "babies",
    "children",
    "adults",
    "booking_changes",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_bookings_not_canceled",
}


def clasificar_variable(serie: pd.Series, nombre: str, umbral: int = UMBRAL_CATEGORICA) -> str:
    if nombre in COLS_ID:
        return "id"
    if "date" in nombre and not pd.api.types.is_numeric_dtype(serie):
        return "fecha"
    if nombre in COLS_CAT_FORZADAS:
        return "cat_baja" if serie.nunique(dropna=True) <= umbral else "cat_alta"
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        return "num_discreta" if serie.nunique(dropna=True) <= umbral else "num_continua"
    return "cat_baja" if serie.nunique(dropna=True) <= umbral else "cat_alta"
