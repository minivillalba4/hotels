import pandas as pd

from src.config import TOP_N_DEFAULT


def conteo(serie: pd.Series, columna_recuento: str = "recuento") -> pd.DataFrame:
    out = serie.value_counts(dropna=True).reset_index()
    out.columns = [serie.name, columna_recuento]
    return out


def top_n_serie(serie: pd.Series, n: int = TOP_N_DEFAULT) -> pd.Series:
    top = serie.value_counts(dropna=True).head(n).index
    return serie.where(serie.isin(top), other="Otros")


def top_n_tabla(serie: pd.Series, n: int = TOP_N_DEFAULT) -> pd.DataFrame:
    conteos = serie.value_counts(dropna=True)
    otros_total = int(conteos.iloc[n:].sum())
    cabeza = conteos.head(n)
    if otros_total > 0:
        cabeza = pd.concat([cabeza, pd.Series({"Otros": otros_total})])
    out = cabeza.reset_index()
    out.columns = [serie.name, "recuento"]
    total = out["recuento"].sum()
    out["%"] = (out["recuento"] / total * 100).round(2)
    return out
