import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.agrupacion_categorias import top_n_serie, top_n_tabla
from src.clasificacion_variables import clasificar_variable
from src.config import UMBRAL_CATEGORICA
from src.configurar_logging import get_logger
from src.estadistica import correlation_ratio, cramer_v
from src.kpis import tasa_cancelacion
from src.traducciones import etiqueta
from src.ui.tema import (
    BRAND,
    ESCALA_DIVERGENTE,
    ESCALA_SECUENCIAL_ACCENT,
    ESCALA_SECUENCIAL_PRIMARY,
    eyebrow,
    lead,
)
from src.ui.graficos import (
    ESTADO_COLORES,
    ESTADO_ORDEN,
    aplicar_boxplot_es,
    aplicar_estilo_heatmap,
    asignar_estado,
    chart_block,
    fmt_es,
    forzar_object_si_cat,
    mostrar_grafico,
)

log = get_logger()

TIPOS_NUMERICOS = {"num_continua", "num_discreta"}
TIPOS_CATEGORICOS = {"cat_baja", "cat_alta"}

TARGET = "is_canceled"
COLS_EXCLUIDAS_BI = {"is_canceled", "reservation_status", "reservation_status_date"}
COLS_EXCLUIDAS_MULTI = {"reservation_status", "reservation_status_date"}

_TOGGLE_IDS_LABEL = "Mostrar identificadores y fechas"
_TOGGLE_IDS_HELP = "ID de agencia, ID de empresa y fechas se ocultan por defecto."

_NBINS_HIST = 50
_HIST_BI_OPACITY = 0.6
_SCATTER_OPACITY = 0.45
_VIOLIN_OPACITY = 0.75
_VIOLIN_WIDTH = 0.85
_VIOLIN_MARKER_SIZE = 11

_HEATMAP_HEIGHT_MIN = 420
_HEATMAP_ROW_PX = 28
_HEATMAP_OVERHEAD_PX = 180


def _reset_key_si_invalido(key: str, opciones: list[str]) -> None:
    if st.session_state.get(key) not in opciones:
        st.session_state.pop(key, None)


def _filtrar_cols_por_tipo(tipos: dict[str, str], tipo_filtro: str) -> list[str]:
    if tipo_filtro == "Numéricas":
        return [c for c, t in tipos.items() if t in TIPOS_NUMERICOS]
    if tipo_filtro == "Categóricas":
        return [c for c, t in tipos.items() if t in TIPOS_CATEGORICOS]
    return list(tipos.keys())


def _caption_n_cols(n: int) -> str:
    s = "s" if n != 1 else ""
    return f"{n} variable{s} disponible{s}"


@st.cache_data(show_spinner=False)
def _tipos_columnas(
    columnas: tuple[str, ...], _df: pd.DataFrame, incluir_ids_fechas: bool
) -> dict[str, str]:
    tipos = {c: clasificar_variable(_df[c], c) for c in columnas}
    if not incluir_ids_fechas:
        tipos = {c: t for c, t in tipos.items() if t not in {"id", "fecha"}}
    return tipos


def _cabecera(serie: pd.Series) -> None:
    n = len(serie)
    n_nulos = int(serie.isna().sum())
    n_unicos = int(serie.nunique(dropna=True))

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{n:,}")
    c2.metric("Valores únicos", f"{n_unicos:,}")
    c3.metric("Valores nulos", f"{n_nulos:,}")


def _plot_histograma(serie: pd.Series, nombre: str) -> None:
    etq = etiqueta(nombre)
    df_plot = pd.DataFrame({nombre: serie})
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_plot, x=nombre, nbins=_NBINS_HIST, title=f"Histograma: {etq}",
                           labels={nombre: etq},
                           color_discrete_sequence=[BRAND["primary"]])
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Frecuencia")
        mostrar_grafico(fig, key=f"uni_hist_{nombre}")
    with col2:
        fig = px.box(df_plot, x=nombre, title=f"Diagrama de caja: {etq}",
                     labels={nombre: etq},
                     color_discrete_sequence=[BRAND["primary"]])
        aplicar_boxplot_es(fig, serie, etq)
        mostrar_grafico(fig, key=f"uni_box_{nombre}")


def _plot_countplot(serie: pd.Series, nombre: str, top_n: int | None = None) -> None:
    etq = etiqueta(nombre)
    if top_n and serie.nunique(dropna=True) > top_n:
        tabla = top_n_tabla(serie, n=top_n)
        total = serie.nunique(dropna=True)
        st.caption(f"Mostrando los {top_n} valores más frecuentes de {total} categorías (el resto se agrupa en 'Otros').")
    else:
        tabla = serie.value_counts(dropna=True).reset_index()
        tabla.columns = [nombre, "recuento"]

    fig = px.bar(
        tabla, x="recuento", y=nombre, orientation="h",
        text="recuento",
        labels={nombre: etq, "recuento": "Recuento"},
        color_discrete_sequence=[BRAND["primary"]],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    mostrar_grafico(fig, key=f"uni_count_{nombre}")


@st.fragment
def _render_univariable(df: pd.DataFrame) -> None:
    incluir = st.toggle(
        _TOGGLE_IDS_LABEL, value=False, key="uni_incluir_ids",
        help=_TOGGLE_IDS_HELP,
    )

    c1, c2 = st.columns([1, 3])
    tipo_filtro = c1.radio(
        "Tipo de variable",
        ["Todas", "Numéricas", "Categóricas"],
        key="uni_tipo_filtro",
    )
    tipos = _tipos_columnas(tuple(df.columns), df, incluir_ids_fechas=incluir)
    cols = _filtrar_cols_por_tipo(tipos, tipo_filtro)

    c1.caption(_caption_n_cols(len(cols)))

    if not cols:
        st.info("No hay variables de este tipo. Ajusta el filtro.")
        return
    _reset_key_si_invalido("uni_columna", cols)
    columna = c2.selectbox(
        "Variable a analizar", cols, key="uni_columna",
        format_func=etiqueta,
    )
    if not columna:
        return

    serie = df[columna]
    tipo = tipos[columna]
    log.info("Univariable: columna=%s tipo=%s", columna, tipo)

    _cabecera(serie)
    st.divider()

    with chart_block(f"Análisis univariable: {etiqueta(columna)}"):
        if tipo == "num_continua":
            _plot_histograma(serie.dropna(), columna)
        elif tipo in {"num_discreta", "cat_baja"}:
            _plot_countplot(serie, columna)
        elif tipo == "cat_alta":
            _plot_countplot(serie, columna, top_n=UMBRAL_CATEGORICA)
        else:
            _plot_countplot(serie.astype(str), columna, top_n=UMBRAL_CATEGORICA)


def _con_estado(df: pd.DataFrame, col: str) -> pd.DataFrame:
    datos = df[[col, TARGET]].dropna(subset=[col]).copy()
    datos["estado"] = asignar_estado(datos[TARGET])
    return datos


def _plot_histograma_bi(df: pd.DataFrame, col: str) -> None:
    etq = etiqueta(col)
    datos = _con_estado(df, col)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            datos, x=col, color="estado", nbins=_NBINS_HIST,
            barmode="overlay", opacity=_HIST_BI_OPACITY,
            title=f"Histograma: {etq}",
            labels={col: etq, "estado": "Estado"},
            color_discrete_map=ESTADO_COLORES,
            category_orders={"estado": ESTADO_ORDEN},
        )
        fig.update_yaxes(title="Frecuencia")
        fig.update_layout(legend_title_text="")
        mostrar_grafico(fig, key=f"bi_hist_{col}")
    with col2:
        fig = px.box(
            datos, x="estado", y=col, color="estado",
            title=f"Diagrama de caja: {etq}",
            labels={col: etq, "estado": "Estado"},
            color_discrete_map=ESTADO_COLORES,
            category_orders={"estado": ESTADO_ORDEN},
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        fig.update_traces(
            hovertemplate=f"{etq}: %{{y:.2f}}<extra>%{{x}}</extra>",
        )
        mostrar_grafico(fig, key=f"bi_box_{col}")


def _plot_countplot_bi(df: pd.DataFrame, col: str, top_n: int | None = None) -> None:
    etq = etiqueta(col)
    datos = _con_estado(df, col)
    datos[col] = forzar_object_si_cat(datos[col])

    if top_n and datos[col].nunique(dropna=True) > top_n:
        total = datos[col].nunique(dropna=True)
        datos[col] = top_n_serie(datos[col], n=top_n)
        st.caption(f"Mostrando los {top_n} valores más frecuentes de {total} categorías (el resto se agrupa en 'Otros').")

    agrupado = (
        datos.groupby([col, "estado"], observed=True)
        .size()
        .reset_index(name="recuento")
    )

    fig = px.bar(
        agrupado, x="recuento", y=col, color="estado", orientation="h",
        labels={col: etq, "recuento": "Recuento", "estado": "Estado"},
        color_discrete_map=ESTADO_COLORES,
        category_orders={"estado": ESTADO_ORDEN},
        barmode="stack",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title_text="")
    mostrar_grafico(fig, key=f"bi_count_{col}")


@st.fragment
def _render_bivariable(df: pd.DataFrame) -> None:
    if TARGET not in df.columns:
        st.error(f"Falta la columna objetivo '{TARGET}' en el dataset.")
        return

    incluir = st.toggle(
        _TOGGLE_IDS_LABEL, value=False, key="bi_incluir_ids",
        help=_TOGGLE_IDS_HELP,
    )

    c1, c2 = st.columns([1, 3])
    tipo_filtro = c1.radio(
        "Tipo de variable",
        ["Todas", "Numéricas", "Categóricas"],
        key="bi_tipo_filtro",
    )
    tipos = {
        c: t
        for c, t in _tipos_columnas(tuple(df.columns), df, incluir_ids_fechas=incluir).items()
        if c not in COLS_EXCLUIDAS_BI
    }
    cols = _filtrar_cols_por_tipo(tipos, tipo_filtro)

    c1.caption(_caption_n_cols(len(cols)))

    if not cols:
        st.info("No hay variables de este tipo. Ajusta el filtro.")
        return
    _reset_key_si_invalido("bi_columna", cols)
    columna = c2.selectbox(
        "Variable a comparar con Cancelaciones", cols, key="bi_columna",
        format_func=etiqueta,
    )
    if not columna:
        return

    serie = df[columna]
    tipo = tipos[columna]
    log.info("Bivariable: columna=%s tipo=%s target=%s", columna, tipo, TARGET)

    _cabecera(serie)
    tasa = tasa_cancelacion(df)
    st.caption(f"Tasa global de cancelación: **{fmt_es(tasa, 1)}%** (referencia para comparar).")
    st.divider()

    with chart_block(f"Análisis bivariable: {etiqueta(columna)} vs {etiqueta(TARGET)}"):
        if tipo == "num_continua":
            _plot_histograma_bi(df, columna)
        elif tipo in {"num_discreta", "cat_baja"}:
            _plot_countplot_bi(df, columna)
        elif tipo == "cat_alta":
            _plot_countplot_bi(df, columna, top_n=UMBRAL_CATEGORICA)
        else:
            df_str = df.assign(**{columna: df[columna].astype(str)})
            _plot_countplot_bi(df_str, columna, top_n=UMBRAL_CATEGORICA)


@st.cache_data(show_spinner="Calculando matriz de correlación (Spearman)…")
def _matriz_spearman(_df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    return _df[list(cols)].corr(method="spearman")


@st.cache_data(show_spinner="Calculando matriz V de Cramér…")
def _matriz_cramer(_df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    cols = list(cols)
    data = {}
    for c in cols:
        s = _df[c]
        data[c] = top_n_serie(s, n=UMBRAL_CATEGORICA) if s.nunique(dropna=True) > UMBRAL_CATEGORICA else s
    M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            v = cramer_v(data[a], data[b])
            M.iat[i, j] = v
            M.iat[j, i] = v
    return M


@st.cache_data(show_spinner="Calculando ratio de correlación (η)…")
def _matriz_eta(_df: pd.DataFrame, cols_num: tuple, cols_cat: tuple) -> pd.DataFrame:
    cols_num, cols_cat = list(cols_num), list(cols_cat)
    M = pd.DataFrame(index=cols_num, columns=cols_cat, dtype=float)
    for num in cols_num:
        for cat in cols_cat:
            M.loc[num, cat] = correlation_ratio(_df[cat], _df[num])
    return M


def _plot_matriz(M: pd.DataFrame, titulo: str, key: str, *, zmin=None, zmax=None,
                 colorscale=None, leyenda_color: str = "Valor") -> None:
    M_disp = M.copy()
    M_disp.index = [etiqueta(c) for c in M.index]
    M_disp.columns = [etiqueta(c) for c in M.columns]
    fig = px.imshow(
        M_disp,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=colorscale,
        zmin=zmin, zmax=zmax,
        labels=dict(color=leyenda_color),
        title=titulo,
    )
    fig.update_layout(
        height=max(_HEATMAP_HEIGHT_MIN, _HEATMAP_ROW_PX * len(M_disp.index) + _HEATMAP_OVERHEAD_PX)
    )
    aplicar_estilo_heatmap(fig)
    mostrar_grafico(fig, key=key)


def _plot_scatter_par(df: pd.DataFrame, x: str, y: str) -> None:
    datos = df[[x, y]].dropna()
    rho = float(datos[[x, y]].corr(method="spearman").iat[0, 1])
    fig = px.scatter(
        datos, x=x, y=y,
        labels={x: etiqueta(x), y: etiqueta(y)},
        title=f"Dispersión: {etiqueta(x)} vs {etiqueta(y)}",
        color_discrete_sequence=[BRAND["primary"]],
        opacity=_SCATTER_OPACITY,
        render_mode="webgl",
        trendline="ols",
        trendline_color_override=BRAND["highlight"],
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"ρ Spearman = {fmt_es(rho, 3)}",
        showarrow=False, bgcolor=BRAND["overlay"],
        bordercolor=BRAND["primary"], borderwidth=1,
        font=dict(size=13, color=BRAND["primary"]),
        xanchor="left", yanchor="top",
    )
    mostrar_grafico(fig, key=f"multi_scatter_{x}_{y}")


def _plot_violin_par(df: pd.DataFrame, col_num: str, col_cat: str,
                     top_n: int | None = None) -> None:
    datos = df[[col_num, col_cat]].dropna().copy()
    datos[col_cat] = forzar_object_si_cat(datos[col_cat])
    if top_n and datos[col_cat].nunique() > top_n:
        total_cat = datos[col_cat].nunique()
        datos[col_cat] = top_n_serie(datos[col_cat], n=top_n)
        st.caption(
            f"Mostrando los {top_n} valores más frecuentes de {total_cat} categorías "
            "(el resto se agrupa en 'Otros')."
        )
    eta = correlation_ratio(datos[col_cat], datos[col_num])
    resumen = (
        datos.groupby(col_cat, observed=True)[col_num]
        .agg(n="count", media="mean", mediana="median", std="std")
        .sort_values("media")
    )
    orden = resumen.index.tolist()
    media_global = float(datos[col_num].mean())

    fig = px.violin(
        datos, x=col_cat, y=col_num,
        labels={col_num: etiqueta(col_num), col_cat: etiqueta(col_cat)},
        title=f"Distribución de {etiqueta(col_num)} por {etiqueta(col_cat)}",
        color_discrete_sequence=[BRAND["primary"]],
        category_orders={col_cat: orden},
        box=True, points=False,
    )
    fig.update_traces(meanline_visible=True, opacity=_VIOLIN_OPACITY, width=_VIOLIN_WIDTH)
    fig.add_scatter(
        x=orden,
        y=[resumen.loc[c, "media"] for c in orden],
        mode="markers",
        marker=dict(
            size=_VIOLIN_MARKER_SIZE, color=BRAND["highlight"],
            line=dict(color=BRAND["ink"], width=1.2),
            symbol="diamond",
        ),
        name="Media",
        hovertemplate="Media: %{y:.2f}<extra></extra>",
        showlegend=False,
    )
    fig.add_hline(
        y=media_global, line_dash="dot", line_color=BRAND["muted"],
        annotation_text=f"Media global: {fmt_es(media_global, 2)}",
        annotation_position="top right",
        annotation_font_color=BRAND["muted"],
    )
    fig.update_layout(showlegend=False)
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"η = {fmt_es(eta, 3)}",
        showarrow=False, bgcolor=BRAND["overlay"],
        bordercolor=BRAND["primary"], borderwidth=1,
        font=dict(size=13, color=BRAND["primary"]),
        xanchor="left", yanchor="top",
    )
    mostrar_grafico(fig, key=f"multi_violin_{col_num}_{col_cat}")

    tabla = resumen.reset_index().rename(columns={col_cat: etiqueta(col_cat)})
    tabla["desv. de la media global"] = tabla["media"] - media_global
    tabla = tabla[[etiqueta(col_cat), "n", "media", "mediana", "std",
                   "desv. de la media global"]]
    st.dataframe(
        tabla,
        width="stretch",
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn("n", format="%d"),
            "media": st.column_config.NumberColumn("media", format="%.2f"),
            "mediana": st.column_config.NumberColumn("mediana", format="%.2f"),
            "std": st.column_config.NumberColumn("std", format="%.2f"),
            "desv. de la media global": st.column_config.NumberColumn(
                "desv. media global", format="%.2f"
            ),
        },
    )


def _plot_contingencia(df: pd.DataFrame, a: str, b: str,
                       top_n: int | None = None) -> None:
    datos = df[[a, b]].dropna().copy()
    for c in (a, b):
        datos[c] = forzar_object_si_cat(datos[c])
        if top_n and datos[c].nunique() > top_n:
            total_c = datos[c].nunique()
            datos[c] = top_n_serie(datos[c], n=top_n)
            st.caption(
                f"{etiqueta(c)}: mostrando los {top_n} valores más frecuentes "
                f"de {total_c} categorías (el resto en 'Otros')."
            )

    tab_abs = pd.crosstab(datos[a], datos[b])
    tab_pct = pd.crosstab(datos[a], datos[b], normalize="index") * 100
    v = cramer_v(datos[a], datos[b])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(
            tab_abs,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=ESCALA_SECUENCIAL_PRIMARY,
            labels=dict(x=etiqueta(b), y=etiqueta(a), color="Recuento"),
            title="Tabla de contingencia (recuentos)",
        )
        aplicar_estilo_heatmap(fig)
        mostrar_grafico(fig, key=f"multi_cont_abs_{a}_{b}")
    with col2:
        fig = px.imshow(
            tab_pct,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale=ESCALA_SECUENCIAL_ACCENT,
            labels=dict(x=etiqueta(b), y=etiqueta(a), color="% por fila"),
            title="% por fila (condicionada a la fila)",
            zmin=0, zmax=100,
        )
        aplicar_estilo_heatmap(fig)
        mostrar_grafico(fig, key=f"multi_cont_pct_{a}_{b}")

    st.caption(
        f"**V de Cramér = {fmt_es(v, 3)}** "
        "(0 = sin asociación, 1 = asociación perfecta)."
    )


def _cols_multi(df: pd.DataFrame, incluir_ids: bool, tipos_validos: set) -> list[str]:
    tipos = _tipos_columnas(tuple(df.columns), df, incluir_ids_fechas=incluir_ids)
    return [
        c for c, t in tipos.items()
        if c not in COLS_EXCLUIDAS_MULTI and t in tipos_validos
    ]


@st.fragment
def _render_multi_numericas(df: pd.DataFrame) -> None:
    incluir = st.toggle(
        _TOGGLE_IDS_LABEL, value=False, key="multi_num_ids",
        help=_TOGGLE_IDS_HELP,
    )
    cols = _cols_multi(df, incluir, TIPOS_NUMERICOS)
    if len(cols) < 2:
        st.info("Se necesitan al menos 2 variables numéricas.")
        return

    log.info("Multi num-num: %d variables", len(cols))
    st.caption(
        f"{len(cols)} variables numéricas. Se usa **correlación de Spearman** "
        "(basada en rangos: captura relaciones monótonas y es robusta a valores atípicos)."
    )
    M = _matriz_spearman(df, tuple(cols))
    with chart_block("Matriz de correlación (Spearman)"):
        _plot_matriz(
            M, "Matriz de correlación (Spearman)", "multi_num_heatmap",
            zmin=-1, zmax=1, colorscale=ESCALA_DIVERGENTE, leyenda_color="ρ",
        )

    st.divider()
    st.markdown("**Detalle de un par de variables**")
    _reset_key_si_invalido("multi_num_x", cols)
    c1, c2 = st.columns(2)
    x = c1.selectbox("Variable X", cols, key="multi_num_x", format_func=etiqueta)
    opts_y = [c for c in cols if c != x]
    _reset_key_si_invalido("multi_num_y", opts_y)
    y = c2.selectbox("Variable Y", opts_y, key="multi_num_y", format_func=etiqueta)
    if x and y:
        log.info("Multivariable num-num detalle: %s vs %s", x, y)
        with chart_block(f"Dispersión: {etiqueta(x)} vs {etiqueta(y)}"):
            _plot_scatter_par(df, x, y)


@st.fragment
def _render_multi_categoricas(df: pd.DataFrame) -> None:
    incluir = st.toggle(
        _TOGGLE_IDS_LABEL, value=False, key="multi_cat_ids",
        help=_TOGGLE_IDS_HELP,
    )
    cols = _cols_multi(df, incluir, TIPOS_CATEGORICOS)
    if len(cols) < 2:
        st.info("Se necesitan al menos 2 variables categóricas.")
        return

    log.info("Multi cat-cat: %d variables", len(cols))
    st.caption(
        f"{len(cols)} variables categóricas. **V de Cramér** con corrección de sesgo "
        f"(0 = sin asociación, 1 = asociación perfecta). Las variables con más de "
        f"{UMBRAL_CATEGORICA} categorías se agrupan (top {UMBRAL_CATEGORICA} + «Otros»)."
    )
    M = _matriz_cramer(df, tuple(cols))
    with chart_block("Matriz V de Cramér"):
        _plot_matriz(
            M, "Matriz V de Cramér", "multi_cat_heatmap",
            zmin=0, zmax=1, colorscale=ESCALA_SECUENCIAL_PRIMARY, leyenda_color="V",
        )

    st.divider()
    st.markdown("**Tabla de contingencia de un par**")
    _reset_key_si_invalido("multi_cat_a", cols)
    c1, c2 = st.columns(2)
    a = c1.selectbox("Variable A", cols, key="multi_cat_a", format_func=etiqueta)
    opts_b = [c for c in cols if c != a]
    _reset_key_si_invalido("multi_cat_b", opts_b)
    b = c2.selectbox("Variable B", opts_b, key="multi_cat_b", format_func=etiqueta)
    if a and b:
        log.info("Multivariable cat-cat detalle: %s vs %s", a, b)
        with chart_block(f"Contingencia: {etiqueta(a)} vs {etiqueta(b)}"):
            _plot_contingencia(df, a, b, top_n=UMBRAL_CATEGORICA)


@st.fragment
def _render_multi_mixtas(df: pd.DataFrame) -> None:
    incluir = st.toggle(
        _TOGGLE_IDS_LABEL, value=False, key="multi_mix_ids",
        help=_TOGGLE_IDS_HELP,
    )
    cols_num = _cols_multi(df, incluir, TIPOS_NUMERICOS)
    cols_cat = _cols_multi(df, incluir, TIPOS_CATEGORICOS)
    if not cols_num or not cols_cat:
        st.info("Se necesita al menos 1 variable numérica y 1 categórica.")
        return

    log.info("Multi num-cat: %d num × %d cat", len(cols_num), len(cols_cat))
    st.caption(
        f"{len(cols_num)} numéricas × {len(cols_cat)} categóricas. "
        "**Ratio de correlación (η)**: proporción de varianza de la numérica que "
        "explica la categórica (0 = ninguna, 1 = totalmente)."
    )
    M = _matriz_eta(df, tuple(cols_num), tuple(cols_cat))
    with chart_block("Matriz η (numéricas × categóricas)"):
        _plot_matriz(
            M, "Matriz η (numéricas × categóricas)", "multi_mix_heatmap",
            zmin=0, zmax=1, colorscale=ESCALA_SECUENCIAL_ACCENT, leyenda_color="η",
        )

    st.divider()
    st.markdown("**Detalle de un par**")
    _reset_key_si_invalido("multi_mix_num", cols_num)
    _reset_key_si_invalido("multi_mix_cat", cols_cat)
    c1, c2 = st.columns(2)
    num = c1.selectbox("Variable numérica", cols_num, key="multi_mix_num",
                       format_func=etiqueta)
    cat = c2.selectbox("Variable categórica", cols_cat, key="multi_mix_cat",
                       format_func=etiqueta)
    if num and cat:
        log.info("Multivariable num-cat detalle: %s vs %s", num, cat)
        with chart_block(f"Distribución: {etiqueta(num)} por {etiqueta(cat)}"):
            _plot_violin_par(df, num, cat, top_n=UMBRAL_CATEGORICA)


def _render_multivariable(df: pd.DataFrame) -> None:
    st.caption(
        "Exploración simultánea de varias variables. Cada subpestaña muestra una "
        "**matriz global** y un **detalle** del par seleccionado."
    )
    tab_n, tab_c, tab_m = st.tabs(["Numéricas", "Categóricas", "Mixtas"])
    with tab_n:
        _render_multi_numericas(df)
    with tab_c:
        _render_multi_categoricas(df)
    with tab_m:
        _render_multi_mixtas(df)


def render(df: pd.DataFrame) -> None:
    log.info("Render AnalisisExploratorio inicio | df shape=%s", df.shape)
    eyebrow("05 / ANÁLISIS EXPLORATORIO")
    st.header("Análisis exploratorio del dataset")
    lead("Exploración univariable, bivariable frente al target de cancelación y multivariable (correlaciones y asociaciones entre todas las variables).")
    tab_uni, tab_bi, tab_multi = st.tabs(
        ["Una variable", "Dos variables", "Multivariable"]
    )
    with tab_uni:
        _render_univariable(df)
    with tab_bi:
        _render_bivariable(df)
    with tab_multi:
        _render_multivariable(df)
    log.info("Render AnalisisExploratorio OK")
