from __future__ import annotations

import pandas as pd


LABELS_ES: dict[str, str] = {
    "hotel": "Tipo de hotel",
    "is_canceled": "¿Cancelada?",
    "lead_time": "Días de antelación",
    "arrival_date_year": "Año de llegada",
    "arrival_date_month": "Mes de llegada",
    "arrival_date_week_number": "Semana del año",
    "arrival_date_day_of_month": "Día del mes",
    "stays_in_weekend_nights": "Noches de fin de semana",
    "stays_in_week_nights": "Noches entre semana",
    "adults": "Adultos",
    "children": "Niños",
    "babies": "Bebés",
    "meal": "Régimen de comidas",
    "country": "País",
    "market_segment": "Segmento de mercado",
    "distribution_channel": "Canal de distribución",
    "is_repeated_guest": "¿Huésped repetidor?",
    "previous_cancellations": "Cancelaciones previas",
    "previous_bookings_not_canceled": "Reservas previas no canceladas",
    "reserved_room_type": "Habitación reservada",
    "assigned_room_type": "Habitación asignada",
    "booking_changes": "Cambios sobre la reserva",
    "deposit_type": "Tipo de depósito",
    "agent": "ID de agencia",
    "company": "ID de empresa",
    "days_in_waiting_list": "Días en lista de espera",
    "customer_type": "Tipo de cliente",
    "adr": "Precio medio por noche (€)",
    "required_car_parking_spaces": "Plazas de parking solicitadas",
    "total_of_special_requests": "Peticiones especiales",
    "reservation_status": "Estado final de la reserva",
    "reservation_status_date": "Fecha del estado",
    "contacto_rol": "Data Science",
    "contacto_tagline": "Machine Learning aplicado al negocio hotelero",
    "contacto_btn_linkedin": "LinkedIn",
    "contacto_btn_email": "Email",
}


_MESES_EN_A_ES: dict[str, str] = {
    "January": "Enero", "February": "Febrero", "March": "Marzo",
    "April": "Abril", "May": "Mayo", "June": "Junio",
    "July": "Julio", "August": "Agosto", "September": "Septiembre",
    "October": "Octubre", "November": "Noviembre", "December": "Diciembre",
}

MESES_ORDEN_ES: list[str] = list(_MESES_EN_A_ES.values())


PAISES_ES: dict[str, str] = {
    "ABW": "Aruba", "AGO": "Angola", "AIA": "Anguila", "ALB": "Albania",
    "AND": "Andorra", "ARE": "Emiratos Árabes Unidos", "ARG": "Argentina",
    "ARM": "Armenia", "ASM": "Samoa Americana", "ATA": "Antártida",
    "ATF": "Territorios Australes Franceses", "AUS": "Australia", "AUT": "Austria",
    "AZE": "Azerbaiyán", "BDI": "Burundi", "BEL": "Bélgica", "BEN": "Benín",
    "BFA": "Burkina Faso", "BGD": "Bangladés", "BGR": "Bulgaria", "BHR": "Baréin",
    "BHS": "Bahamas", "BIH": "Bosnia y Herzegovina", "BLR": "Bielorrusia",
    "BOL": "Bolivia", "BRA": "Brasil", "BRB": "Barbados", "BWA": "Botsuana",
    "CAF": "República Centroafricana", "CHE": "Suiza", "CHL": "Chile",
    "CHN": "China", "CIV": "Costa de Marfil", "CMR": "Camerún",
    "COL": "Colombia", "COM": "Comoras", "CPV": "Cabo Verde", "CRI": "Costa Rica",
    "CUB": "Cuba", "CYM": "Islas Caimán", "CYP": "Chipre", "CZE": "Chequia",
    "DEU": "Alemania", "DJI": "Yibuti", "DMA": "Dominica", "DNK": "Dinamarca",
    "DOM": "República Dominicana", "DZA": "Argelia", "ECU": "Ecuador",
    "EGY": "Egipto", "ESP": "España", "EST": "Estonia", "ETH": "Etiopía",
    "FIN": "Finlandia", "FJI": "Fiyi", "FRA": "Francia", "FRO": "Islas Feroe",
    "GAB": "Gabón", "GBR": "Reino Unido", "GEO": "Georgia", "GGY": "Guernesey",
    "GHA": "Ghana", "GIB": "Gibraltar", "GLP": "Guadalupe", "GNB": "Guinea-Bisáu",
    "GRC": "Grecia", "GTM": "Guatemala", "GUY": "Guyana",
    "HKG": "Hong Kong (China)", "HND": "Honduras", "HRV": "Croacia",
    "HUN": "Hungría", "IDN": "Indonesia", "IMN": "Isla de Man", "IND": "India",
    "IRL": "Irlanda", "IRN": "Irán", "IRQ": "Irak", "ISL": "Islandia",
    "ISR": "Israel", "ITA": "Italia", "JAM": "Jamaica", "JEY": "Jersey",
    "JOR": "Jordania", "JPN": "Japón", "KAZ": "Kazajistán", "KEN": "Kenia",
    "KHM": "Camboya", "KIR": "Kiribati", "KNA": "San Cristóbal y Nieves",
    "KOR": "Corea del Sur", "KWT": "Kuwait", "LAO": "Laos", "LBN": "Líbano",
    "LBY": "Libia", "LCA": "Santa Lucía", "LIE": "Liechtenstein",
    "LKA": "Sri Lanka", "LTU": "Lituania", "LUX": "Luxemburgo", "LVA": "Letonia",
    "MAC": "Macao (China)", "MAR": "Marruecos", "MCO": "Mónaco",
    "MDG": "Madagascar", "MDV": "Maldivas", "MEX": "México",
    "MKD": "Macedonia del Norte", "MLI": "Mali", "MLT": "Malta",
    "MMR": "Myanmar (Birmania)", "MNE": "Montenegro", "MOZ": "Mozambique",
    "MRT": "Mauritania", "MUS": "Mauricio", "MWI": "Malaui", "MYS": "Malasia",
    "MYT": "Mayotte", "NAM": "Namibia", "NCL": "Nueva Caledonia", "NGA": "Nigeria",
    "NIC": "Nicaragua", "NLD": "Países Bajos", "NOR": "Noruega", "NPL": "Nepal",
    "NZL": "Nueva Zelanda", "OMN": "Omán", "PAK": "Pakistán", "PAN": "Panamá",
    "PER": "Perú", "PHL": "Filipinas", "PLW": "Palaos", "POL": "Polonia",
    "PRI": "Puerto Rico", "PRT": "Portugal", "PRY": "Paraguay",
    "PYF": "Polinesia Francesa", "QAT": "Catar", "ROU": "Rumanía", "RUS": "Rusia",
    "RWA": "Ruanda", "SAU": "Arabia Saudí", "SDN": "Sudán", "SEN": "Senegal",
    "SGP": "Singapur", "SLE": "Sierra Leona", "SLV": "El Salvador",
    "SMR": "San Marino", "SRB": "Serbia", "STP": "Santo Tomé y Príncipe",
    "SUR": "Surinam", "SVK": "Eslovaquia", "SVN": "Eslovenia", "SWE": "Suecia",
    "SYC": "Seychelles", "SYR": "Siria", "TGO": "Togo", "THA": "Tailandia",
    "TJK": "Tayikistán", "TMP": "Timor Oriental", "TUN": "Túnez", "TUR": "Turquía",
    "TWN": "Taiwán", "TZA": "Tanzania", "UGA": "Uganda", "UKR": "Ucrania",
    "UMI": "Islas menores alejadas de EE. UU.", "URY": "Uruguay",
    "USA": "Estados Unidos", "UZB": "Uzbekistán", "VEN": "Venezuela",
    "VGB": "Islas Vírgenes Británicas", "VNM": "Vietnam", "ZAF": "Sudáfrica",
    "ZMB": "Zambia", "ZWE": "Zimbabue", "desconocido": "Desconocido",
}


_TIPOS_HABITACION: dict[str, str] = {
    c: f"Tipo {c}" for c in "ABCDEFGHIKLP"
}


NOMBRE_A_ISO3: dict[str, str] = {
    nombre: iso for iso, nombre in PAISES_ES.items() if len(iso) == 3
}


_NORMALIZACION_PAISES: dict[str, str] = {"CN": "CHN"}


VALORES_ES: dict[str, dict] = {
    "hotel": {
        "City Hotel": "Hotel urbano",
        "Resort Hotel": "Hotel vacacional",
    },
    "deposit_type": {
        "No Deposit": "Sin depósito",
        "Non Refund": "Sin reembolso",
        "Refundable": "Reembolsable",
    },
    "customer_type": {
        "Transient": "Individual",
        "Transient-Party": "Individual en grupo",
        "Contract": "Contrato",
        "Group": "Grupo",
    },
    "meal": {
        "BB": "Solo desayuno",
        "HB": "Media pensión",
        "FB": "Pensión completa",
        "SC": "Sin comidas",
        "Undefined": "Sin especificar",
    },
    "market_segment": {
        "Online TA": "Agencia online",
        "Offline TA/TO": "Agencia física",
        "Direct": "Reserva directa",
        "Corporate": "Corporativo",
        "Complementary": "Cortesía",
        "Groups": "Grupos",
        "Aviation": "Aerolíneas",
        "Undefined": "Sin especificar",
    },
    "distribution_channel": {
        "Direct": "Reserva directa",
        "Corporate": "Corporativo",
        "TA/TO": "Agencias",
        "GDS": "GDS central",
        "Undefined": "Sin especificar",
    },
    "reservation_status": {
        "Canceled": "Cancelada",
        "Check-Out": "Completada",
        "No-Show": "No presentada",
    },
    "country": PAISES_ES,
    "reserved_room_type": _TIPOS_HABITACION,
    "assigned_room_type": _TIPOS_HABITACION,
}


def etiqueta(col: str) -> str:
    return LABELS_ES.get(col, col)


def traducir_mes(serie: pd.Series) -> pd.Series:
    if isinstance(serie.dtype, pd.CategoricalDtype):
        serie = serie.astype(object)
    return serie.replace(_MESES_EN_A_ES)


def traducir_valores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "country" in df.columns:
        df["country"] = df["country"].replace(_NORMALIZACION_PAISES)
    for col, mapping in VALORES_ES.items():
        if col not in df.columns:
            continue
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        df[col] = s.map(mapping).fillna(s)
    return df
