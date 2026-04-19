from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.configurar_logging import get_logger
from src.traducciones import LABELS_ES

log = get_logger()


CONTACTO: dict[str, str | int] = {
    "nombre": "Ismael Villalba Serrano",
    "rol_clave": "contacto_rol",
    "tagline_clave": "contacto_tagline",
    "linkedin": "https://www.linkedin.com/in/ismael-villalba-serrano-209440229/",
    "email": "villalbaserranoismael@gmail.com",
    "foto": "docs/1758611341302.jpeg",
    "año": 2026,
}


def _texto(clave_clave: str, por_defecto: str) -> str:
    clave = CONTACTO.get(clave_clave)
    if isinstance(clave, str):
        return LABELS_ES.get(clave, por_defecto)
    return por_defecto


def _foto_data_uri() -> str | None:
    ruta_rel = CONTACTO.get("foto")
    if not isinstance(ruta_rel, str) or not ruta_rel:
        return None
    ruta = Path(__file__).resolve().parents[2] / ruta_rel
    if not ruta.exists():
        return None
    import base64
    import mimetypes

    tipo, _ = mimetypes.guess_type(str(ruta))
    if tipo is None:
        tipo = "image/jpeg"
    b64 = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f"data:{tipo};base64,{b64}"


def render_hero() -> None:
    nombre = CONTACTO.get("nombre")
    if not isinstance(nombre, str) or not nombre.strip():
        return

    rol = _texto("rol_clave", "Analista de datos")
    tagline = _texto(
        "tagline_clave",
        "Machine Learning aplicado al negocio hotelero",
    )
    linkedin = CONTACTO.get("linkedin") or ""
    email = CONTACTO.get("email") or ""
    foto_uri = _foto_data_uri()

    lbl_linkedin = LABELS_ES.get("contacto_btn_linkedin", "LinkedIn")
    lbl_email = LABELS_ES.get("contacto_btn_email", "Email")

    foto_html = (
        f'<img class="hero-contacto__foto" src="{foto_uri}" alt="{escape(nombre)}" />'
        if foto_uri
        else '<div class="hero-contacto__foto hero-contacto__foto--placeholder" aria-hidden="true"></div>'
    )

    botones = []
    if linkedin:
        botones.append(
            f'<a class="hero-contacto__btn" href="{escape(str(linkedin))}" '
            f'target="_blank" rel="noopener">{escape(lbl_linkedin)}</a>'
        )
    if email:
        mailto = f"mailto:{email}?subject=Contacto%20desde%20portfolio"
        botones.append(
            f'<a class="hero-contacto__btn hero-contacto__btn--ghost" '
            f'href="{escape(mailto)}">{escape(lbl_email)}</a>'
        )

    email_copia = (
        f'<div class="hero-contacto__email" aria-label="Email">'
        f'<code>{escape(str(email))}</code></div>'
        if email
        else ""
    )

    st.markdown(
        f"""
<section class="hero-contacto" aria-label="Presentación">
  <div class="hero-contacto__media">{foto_html}</div>
  <div class="hero-contacto__body">
    <h1 class="hero-contacto__nombre">{escape(nombre)}</h1>
    <p class="hero-contacto__rol">{escape(rol)}</p>
    <p class="hero-contacto__tagline">{escape(tagline)}</p>
    <div class="hero-contacto__acciones">{"".join(botones)}</div>
    {email_copia}
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    nombre = CONTACTO.get("nombre")
    if not isinstance(nombre, str) or not nombre.strip():
        return

    año = CONTACTO.get("año", 2026)
    email = CONTACTO.get("email") or ""
    linkedin = CONTACTO.get("linkedin") or ""

    partes = [f'© {año} {escape(nombre)}']
    if linkedin:
        partes.append(
            f'<a href="{escape(str(linkedin))}" target="_blank" rel="noopener">LinkedIn</a>'
        )
    if email:
        partes.append(
            f'<a href="mailto:{escape(str(email))}">{escape(str(email))}</a>'
        )

    separador = '<span class="footer-contacto__sep">·</span>'
    cuerpo = separador.join(partes)
    st.markdown(
        f'<footer class="footer-contacto" role="contentinfo">{cuerpo}</footer>',
        unsafe_allow_html=True,
    )
