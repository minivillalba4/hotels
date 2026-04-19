from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.carga_datos import cargar_datos_raw
from src.configurar_logging import get_logger
from src.ml.features import construir_features_modelo
from src.ml.graficos_shap import (
    bar_importance_plotly,
    beeswarm_mpl,
    waterfall_plotly,
)
from src.ml.shap_utils import (
    cargar_modelo,
    cargar_preprocesador,
    cargar_sample_test,
    calcular_shap_global,
    explicar_reserva,
    obtener_explainer,
)
from src.ui.graficos import chart_block, mostrar_grafico
from src.ui.tema import eyebrow, lead

log = get_logger()


def render(df: pd.DataFrame) -> None:
    log.info("Render Interpretabilidad inicio | df shape=%s", df.shape)
    eyebrow("07 / INTERPRETABILIDAD")

    try:
        modelo = cargar_modelo()
        preprocesador, feature_names = cargar_preprocesador()
        X_sample, _ = cargar_sample_test(2000)
    except Exception:
        log.exception("Error cargando artefactos ML")
        st.error("No se pudieron cargar modelo/preprocesador/sample. Revisa logs/app.log.")
        return

    with st.spinner("Calculando SHAP sobre el sample..."):
        try:
            shap_values, expected_value, mean_abs = calcular_shap_global(modelo, X_sample)
            explainer = obtener_explainer(modelo)
        except Exception:
            log.exception("Error calculando SHAP global")
            st.error("Fallo al calcular SHAP. Revisa logs/app.log.")
            return


    tab_global, tab_ind = st.tabs(["Global", "Individual"])

    with tab_global:
        st.subheader("Importancia media |SHAP| por variable (top 20)")
        with chart_block("Importancia SHAP global"):
            mostrar_grafico(bar_importance_plotly(mean_abs, top=20), key="shap_bar")
        st.caption("Cuánto mueve en promedio cada variable la predicción del modelo, en magnitud absoluta.")

        st.subheader("Beeswarm: distribución de SHAP por variable")
        with chart_block("Beeswarm SHAP"):
            fig_bee = beeswarm_mpl(shap_values, X_sample, max_display=20)
            st.pyplot(fig_bee, use_container_width=True)
        st.caption("Cada punto es una reserva del sample. Rojo = valor alto de la variable; azul = valor bajo. A la derecha del cero la variable empuja hacia cancelación; a la izquierda, hacia confirmación.")

    with tab_ind:
        st.subheader("Explica una reserva concreta")

        hoteles = ["Todos"] + sorted(df["hotel"].dropna().unique().tolist())
        depositos = ["Todos"] + sorted(df["deposit_type"].dropna().unique().tolist())
        top_paises = df["country"].value_counts().head(20).index.tolist()
        paises = ["Todos"] + top_paises
        estados = ["Ambas", "Solo canceladas", "Solo confirmadas"]

        f1, f2, f3, f4 = st.columns(4)
        hotel_sel = f1.selectbox("Hotel", hoteles, key="ind_hotel")
        dep_sel = f2.selectbox("Depósito", depositos, key="ind_dep")
        pais_sel = f3.selectbox("País (top 20)", paises, key="ind_pais")
        estado_sel = f4.selectbox("Estado real", estados, key="ind_estado")

        lt_min, lt_max = int(df["lead_time"].min()), int(df["lead_time"].max())
        lead_rango = st.slider(
            "Rango de antelación (días)",
            min_value=lt_min, max_value=lt_max,
            value=(lt_min, lt_max), key="ind_lead",
        )

        filtradas = _reservas_filtradas(
            df, hotel_sel, dep_sel, pais_sel, estado_sel, lead_rango[0], lead_rango[1],
        )

        if filtradas.empty:
            log.warning(
                "Filtros sin resultados | hotel=%s dep=%s pais=%s estado=%s lead=%s",
                hotel_sel, dep_sel, pais_sel, estado_sel, lead_rango,
            )
            st.info("Sin reservas con esos filtros. Ajusta los criterios.")
            return

        st.caption(f"Coincidencias: {len(filtradas):,}".replace(",", ".") + " reservas.")

        opciones_res = filtradas.head(200)
        idx_a_etiqueta = {
            idx: f"#{idx} — {row.country} · {row.hotel} · lead {int(row.lead_time)} · ADR {row.adr:.0f}"
            for idx, row in opciones_res.iterrows()
        }
        idx_sel = st.selectbox(
            "Reserva a explicar (máx. 200 primeras coincidencias)",
            options=list(opciones_res.index),
            format_func=idx_a_etiqueta.get,
            key="ind_reserva",
        )

        try:
            df_raw = cargar_datos_raw()
            fila_raw = df_raw.loc[[idx_sel]]
            fila_modelo = construir_features_modelo(fila_raw)
            fila_enc_arr = preprocesador.transform(fila_modelo)
        except KeyError:
            log.exception("Índice %s no encontrado en df_raw", idx_sel)
            st.error("No se pudo reconstruir la reserva. Revisa logs/app.log.")
            return
        except Exception:
            log.exception("Fallo aplicando el preprocesador a la reserva idx=%s", idx_sel)
            st.error("Fallo reconstruyendo la reserva para el modelo. Revisa logs/app.log.")
            return

        fila_enc = pd.DataFrame(np.asarray(fila_enc_arr, dtype="float64"), columns=feature_names, index=fila_modelo.index)

        proba = float(modelo.predict_proba(fila_enc)[0, 1])
        pred_clase = "Cancelada" if proba >= 0.5 else "Confirmada"
        real_clase = "Cancelada" if bool(df.loc[idx_sel, "is_canceled"]) else "Confirmada"

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidad de cancelación", f"{proba * 100:.2f} %".replace(".", ","))
        m2.metric("Predicción", pred_clase)
        m3.metric("Clase real", real_clase)

        log.info("Reserva individual explicada | idx=%s prob=%.4f pred=%s real=%s", idx_sel, proba, pred_clase, real_clase)

        sv_row = explicar_reserva(explainer, fila_enc)
        with chart_block("Waterfall SHAP de la reserva"):
            mostrar_grafico(
                waterfall_plotly(sv_row, fila_enc.iloc[0], expected_value, top=15),
                key=f"shap_wf_{idx_sel}",
            )
        st.caption(
            "Barras naranjas: variables que empujan hacia cancelación en esta reserva. "
            "Barras azules: variables que empujan hacia confirmación. La suma, partiendo del base value, es el logit de la predicción."
        )

    log.info("Render Interpretabilidad OK")


def _reservas_filtradas(
    df: pd.DataFrame,
    hotel: str,
    deposito: str,
    pais: str,
    estado: str,
    lead_min: int,
    lead_max: int,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if hotel != "Todos":
        mask &= df["hotel"] == hotel
    if deposito != "Todos":
        mask &= df["deposit_type"] == deposito
    if pais != "Todos":
        mask &= df["country"] == pais
    if estado == "Solo canceladas":
        mask &= df["is_canceled"].astype(bool)
    elif estado == "Solo confirmadas":
        mask &= ~df["is_canceled"].astype(bool)
    mask &= df["lead_time"].between(lead_min, lead_max)
    return df.loc[mask]
