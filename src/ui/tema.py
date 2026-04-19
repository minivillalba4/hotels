from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Iterable

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

from src.configurar_logging import get_logger

log = get_logger()

_CONFIG_PATH = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"

_TEMA_DEFECTO: dict[str, str] = {
    "primaryColor": "#003580",
    "linkColor": "#009fe3",
    "textColor": "#0B1F3A",
    "backgroundColor": "#FAFAF7",
    "secondaryBackgroundColor": "#FFFFFF",
    "borderColor": "#E3E7EE",
}


def _leer_tema_streamlit() -> dict[str, str]:
    try:
        with _CONFIG_PATH.open("rb") as f:
            cargado = tomllib.load(f).get("theme", {})
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError) as exc:
        log.warning("No se pudo leer %s (%s); usando tema por defecto.", _CONFIG_PATH, exc)
        return dict(_TEMA_DEFECTO)
    return {**_TEMA_DEFECTO, **cargado}


_STREAMLIT_THEME = _leer_tema_streamlit()


BRAND: dict[str, str] = {
    "primary":   _STREAMLIT_THEME["primaryColor"],
    "accent":    _STREAMLIT_THEME["linkColor"],
    "ink":       _STREAMLIT_THEME["textColor"],
    "paper":     _STREAMLIT_THEME["backgroundColor"],
    "card":      _STREAMLIT_THEME["secondaryBackgroundColor"],
    "rule":      _STREAMLIT_THEME["borderColor"],
    "highlight": "#feba02",
    "muted":     "#5A6A83",
    "neutral":   "#666666",
    "grid":      "#EDEFF3",
    "overlay":   "rgba(255,255,255,0.85)",
    "on_dark":   "#FFFFFF",
}


ESCALA_DIVERGENTE: list[list] = [
    [0.0, BRAND["highlight"]],
    [0.5, BRAND["paper"]],
    [1.0, BRAND["primary"]],
]
ESCALA_SECUENCIAL_PRIMARY: list[list] = [
    [0.0, BRAND["paper"]],
    [1.0, BRAND["primary"]],
]
ESCALA_SECUENCIAL_ACCENT: list[list] = [
    [0.0, BRAND["paper"]],
    [1.0, BRAND["accent"]],
]


_COLORWAY = [
    BRAND["primary"],
    BRAND["accent"],
    BRAND["highlight"],
    "#4E7FBD",
    "#7BB7E0",
    "#C99A2E",
    "#1F3C6A",
]


_CSS: str = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

:root {{
    --brand-primary: {BRAND["primary"]};
    --brand-accent: {BRAND["accent"]};
    --brand-highlight: {BRAND["highlight"]};
    --brand-ink: {BRAND["ink"]};
    --brand-muted: {BRAND["muted"]};
    --brand-paper: {BRAND["paper"]};
    --brand-card: {BRAND["card"]};
    --brand-rule: {BRAND["rule"]};
}}

html, body {{
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--brand-ink);
    font-feature-settings: "ss01", "cv05";
}}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] button,
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] select,
[data-testid="stAppViewContainer"] textarea {{
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stHeading"] h4 {{
    font-family: 'Source Serif 4', 'Source Serif Pro', Georgia, serif;
    font-weight: 500;
    letter-spacing: -0.015em;
    color: var(--brand-ink);
    font-style: normal;
}}

h1 {{ font-size: 2.4rem; line-height: 1.08; font-weight: 600; letter-spacing: -0.025em; }}
h2 {{ font-size: 1.65rem; line-height: 1.2; font-weight: 500; }}
h3 {{ font-size: 1.25rem; line-height: 1.25; font-weight: 600; letter-spacing: -0.01em; }}
h4 {{ font-size: 1.02rem; line-height: 1.3; font-weight: 600; }}

[data-testid="stAppViewContainer"] {{
    background: {BRAND["paper"]};
}}

.block-container {{
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}}

/* ----- Hero de contacto (Inicio) ---------------------------------- */
.hero-contacto {{
    display: flex;
    gap: 1.75rem;
    align-items: center;
    padding: 1.5rem 0 1.75rem;
    border-bottom: 1px solid var(--brand-rule);
    margin-bottom: 1.25rem;
}}

.hero-contacto__media {{
    flex: 0 0 auto;
}}

.hero-contacto__foto {{
    width: 140px;
    height: 140px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--brand-rule);
    background: var(--brand-card);
    display: block;
}}

.hero-contacto__foto--placeholder {{
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-accent) 100%);
}}

.hero-contacto__body {{
    flex: 1 1 auto;
    min-width: 0;
}}

.hero-contacto__nombre {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.6rem;
    line-height: 1.05;
    font-weight: 600;
    letter-spacing: -0.025em;
    color: var(--brand-ink);
    margin: 0 0 0.35rem;
}}

.hero-contacto__rol {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--brand-primary);
    font-weight: 600;
    margin: 0 0 0.6rem;
}}

.hero-contacto__tagline {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    color: var(--brand-muted);
    max-width: 62ch;
    line-height: 1.5;
    margin: 0 0 1rem;
}}

.hero-contacto__acciones {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 0.6rem;
}}

.hero-contacto__btn {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 1rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: var(--brand-primary);
    color: var(--brand-card) !important;
    border: 1px solid var(--brand-primary);
    border-radius: 2px;
    text-decoration: none !important;
    transition: background 160ms ease, border-color 160ms ease, color 160ms ease;
}}

.hero-contacto__btn:hover {{
    background: var(--brand-accent);
    border-color: var(--brand-accent);
    color: var(--brand-card) !important;
}}

.hero-contacto__btn:focus-visible {{
    outline: 2px solid var(--brand-highlight);
    outline-offset: 2px;
}}

.hero-contacto__btn--ghost {{
    background: transparent;
    color: var(--brand-primary) !important;
    border-color: var(--brand-rule);
}}

.hero-contacto__btn--ghost:hover {{
    background: var(--brand-card);
    color: var(--brand-accent) !important;
    border-color: var(--brand-accent);
}}

.hero-contacto__email {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--brand-muted);
}}

.hero-contacto__email code {{
    background: var(--brand-card);
    border: 1px solid var(--brand-rule);
    padding: 0.15rem 0.45rem;
    border-radius: 2px;
    user-select: all;
    color: var(--brand-ink);
}}

@media (max-width: 720px) {{
    .hero-contacto {{
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }}
    .hero-contacto__foto {{ width: 96px; height: 96px; }}
    .hero-contacto__nombre {{ font-size: 2rem; }}
}}

/* ----- Footer de contacto (sticky) -------------------------------- */
.footer-contacto {{
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 50;
    padding: 0.55rem 1.25rem;
    background: rgba(250, 250, 247, 0.92);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border-top: 1px solid var(--brand-rule);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--brand-muted);
    text-align: center;
    line-height: 1.4;
}}

.footer-contacto a, .footer-contacto a:visited {{
    color: var(--brand-primary);
    border-bottom: none;
    margin: 0 0.35rem;
}}

.footer-contacto a:hover {{
    color: var(--brand-accent);
}}

.footer-contacto__sep {{
    color: var(--brand-rule);
    margin: 0 0.15rem;
}}

@media (max-width: 600px) {{
    .footer-contacto {{
        font-size: 0.72rem;
        padding: 0.45rem 0.75rem;
    }}
}}

.stCaption, [data-testid="stCaptionContainer"], .st-emotion-cache-1h9usn1 {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    color: var(--brand-muted);
    font-weight: 400;
}}

/* ----- KPI GRID (custom, replaces st.metric) ------------------------ */
.kpi-strip {{
    margin: 1.25rem 0 1.75rem;
    border-top: 1px solid var(--brand-ink);
    border-bottom: 1px solid var(--brand-rule);
    background: var(--brand-card);
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0;
}}

@media (max-width: 1180px) {{
    .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}
@media (max-width: 720px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

.kpi-cell {{
    position: relative;
    padding: 1.15rem 1.35rem 1.25rem;
    border-left: 1px solid var(--brand-rule);
    min-width: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 108px;
}}

.kpi-cell:first-child {{ border-left: none; }}

@media (max-width: 1180px) {{
    .kpi-cell:nth-child(3n+1) {{ border-left: none; }}
    .kpi-cell:nth-child(n+4) {{ border-top: 1px solid var(--brand-rule); }}
}}
@media (max-width: 720px) {{
    .kpi-cell:nth-child(odd) {{ border-left: none; }}
    .kpi-cell:nth-child(n+3) {{ border-top: 1px solid var(--brand-rule); }}
}}

.kpi-label {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--brand-muted);
    font-weight: 600;
    line-height: 1.35;
    white-space: normal;
    margin-bottom: 0.85rem;
}}

.kpi-value {{
    font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace;
    font-size: 1.75rem;
    font-weight: 400;
    color: var(--brand-ink);
    letter-spacing: -0.02em;
    line-height: 1.05;
    font-variant-numeric: tabular-nums lining-nums;
    font-feature-settings: "tnum", "lnum", "zero";
    white-space: nowrap;
    overflow: visible;
}}

.kpi-value .unit {{
    font-size: 0.72em;
    color: var(--brand-muted);
    margin-left: 0.15em;
    font-weight: 400;
}}

.kpi-cell::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 0;
    height: 2px;
    background: var(--brand-primary);
    transition: width 240ms ease;
}}

.kpi-cell:hover::before {{ width: 100%; }}

/* ----- Native st.metric fallback (in case used elsewhere) ----------- */
[data-testid="stMetric"] {{
    background: var(--brand-card);
    border: 1px solid var(--brand-rule);
    padding: 1rem 1.1rem;
    position: relative;
}}

[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--brand-muted);
    font-weight: 600;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.3;
    min-height: 2.4em;
}}

[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {{
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}}

[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 400;
    font-variant-numeric: tabular-nums;
    color: var(--brand-ink);
    letter-spacing: -0.02em;
    white-space: nowrap !important;
    overflow: visible !important;
}}

[data-testid="stMetricValue"] > div {{
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}}

/* ----- Tabs -------------------------------------------------------- */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 0;
    border-bottom: 1px solid var(--brand-rule);
}}

[data-testid="stTabs"] [role="tab"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.1rem;
    border-bottom: 2px solid transparent;
    background: transparent;
    color: var(--brand-muted);
    transition: color 160ms ease, border-color 160ms ease;
}}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: var(--brand-ink);
    border-bottom-color: var(--brand-ink);
    font-weight: 600;
}}

/* ----- Dividers --------------------------------------------------- */
hr, [data-testid="stDivider"] {{
    border: none;
    border-top: 1px solid var(--brand-rule);
    margin: 1.75rem 0;
    background: none;
    height: 0;
}}

/* ----- Editorial bits --------------------------------------------- */
.eyebrow {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--brand-muted);
    margin-bottom: 0.5rem;
    font-weight: 500;
}}

.eyebrow::before {{
    content: "— ";
    color: var(--brand-primary);
    margin-right: 0.15em;
}}

.lead {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 0.98rem;
    color: var(--brand-muted);
    max-width: 68ch;
    margin: 0.4rem 0 1.35rem;
    line-height: 1.55;
}}

.section-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: var(--brand-muted);
    text-transform: uppercase;
    padding: 0.25rem 0;
    border-bottom: 1px solid var(--brand-rule);
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
}}

/* ----- Links ------------------------------------------------------- */
a, a:visited {{
    color: var(--brand-primary);
    text-decoration: none;
    border-bottom: 1px solid var(--brand-rule);
}}

a:hover {{
    color: var(--brand-accent);
    border-bottom-color: var(--brand-accent);
}}

/* ----- Tables / dataframes --------------------------------------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid var(--brand-rule);
    border-radius: 0;
    font-variant-numeric: tabular-nums;
}}

/* ----- Expander --------------------------------------------------- */
[data-testid="stExpander"] {{
    border: 1px solid var(--brand-rule);
    border-radius: 0;
    background: var(--brand-card);
}}

[data-testid="stExpander"] details > summary {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--brand-ink);
    padding: 0.75rem 1rem;
    letter-spacing: 0.01em;
}}

[data-testid="stExpander"] summary p {{ margin: 0; }}

/* Force Material Symbols font on icon elements (fixes _arrow_right leak) */
[data-testid="stExpander"] details > summary svg,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
span[data-testid*="Icon"],
span[class*="material-symbols"],
span[class*="material-icons"],
[class*="material-symbols"],
[class*="material-icons"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp,
.material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-feature-settings: 'liga';
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    -webkit-font-feature-settings: 'liga';
    text-transform: none !important;
    letter-spacing: normal !important;
}}

/* ----- Sub-headers for sections ---------------------------------- */
[data-testid="stAppViewContainer"] h3 {{
    margin-top: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--brand-rule);
}}
</style>
"""


def _register_plotly_template() -> None:
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="IBM Plex Sans, -apple-system, Segoe UI, sans-serif",
                size=12,
                color=BRAND["ink"],
            ),
            title=dict(
                font=dict(
                    family="Source Serif 4, Georgia, serif",
                    size=18,
                    color=BRAND["ink"],
                ),
                x=0,
                xanchor="left",
                pad=dict(l=4, t=8, b=12),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=_COLORWAY,
            margin=dict(l=48, r=24, t=48, b=48),
            xaxis=dict(
                gridcolor=BRAND["grid"],
                linecolor=BRAND["rule"],
                zerolinecolor=BRAND["grid"],
                tickfont=dict(family="IBM Plex Mono, monospace", color=BRAND["muted"], size=11),
                title=dict(font=dict(color=BRAND["muted"], size=12)),
            ),
            yaxis=dict(
                gridcolor=BRAND["grid"],
                linecolor=BRAND["rule"],
                zerolinecolor=BRAND["grid"],
                tickfont=dict(family="IBM Plex Mono, monospace", color=BRAND["muted"], size=11),
                title=dict(font=dict(color=BRAND["muted"], size=12)),
            ),
            legend=dict(
                bgcolor=BRAND["overlay"],
                bordercolor=BRAND["rule"],
                borderwidth=1,
                font=dict(family="IBM Plex Sans, sans-serif", size=11, color=BRAND["ink"]),
            ),
            hoverlabel=dict(
                bgcolor=BRAND["ink"],
                bordercolor=BRAND["ink"],
                font=dict(family="IBM Plex Mono, monospace", color=BRAND["on_dark"], size=12),
            ),
        )
    )
    pio.templates["hotel_brand"] = template
    pio.templates.default = "plotly+hotel_brand"


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly_template()


def eyebrow(text: str) -> None:
    st.markdown(f'<span class="eyebrow">{text}</span>', unsafe_allow_html=True)


def lead(text: str) -> None:
    st.markdown(f'<p class="lead">{text}</p>', unsafe_allow_html=True)


def section_meta(left: str, right: str = "") -> None:
    st.markdown(
        f'<div class="section-meta"><span>{left}</span><span>{right}</span></div>',
        unsafe_allow_html=True,
    )


def kpi_grid(items: Iterable[tuple[str, str] | tuple[str, str, str]]) -> None:
    cells = []
    for item in items:
        if len(item) == 3:
            label, value, unit = item
            unit_html = f'<span class="unit">{unit}</span>' if unit else ""
        elif len(item) == 2:
            label, value = item
            unit_html = ""
        else:
            raise ValueError(
                f"kpi_grid: cada item debe tener 2 o 3 elementos, recibido {len(item)}."
            )
        cells.append(
            f'<div class="kpi-cell">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}{unit_html}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="kpi-strip"><div class="kpi-grid">{"".join(cells)}</div></div>',
        unsafe_allow_html=True,
    )
