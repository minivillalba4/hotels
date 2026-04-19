# Hotel Bookings

Dashboard interactivo de análisis de reservas hoteleras y predicción de cancelaciones.

La aplicación explora el histórico combinado de un hotel urbano y otro vacacional, identifica los principales factores asociados a la cancelación y permite estimar, para cualquier reserva nueva, la probabilidad de que termine cancelándose.

## Funcionalidades

- **Inicio**: resumen ejecutivo e indicadores agregados del periodo.
- **Cancelaciones**: tasa global, evolución temporal y segmentación por canal.
- **Estacionalidad**: patrones mensuales y semanales de ocupación y cancelación.
- **Geografía y clientes**: distribución geográfica y perfil del huésped.
- **Análisis exploratorio**: estadística descriptiva, correlaciones y tests de independencia.
- **Simulador**: predicción individual de cancelación a partir de los parámetros de una reserva.
- **Interpretabilidad**: explicaciones SHAP globales y locales del clasificador.

## Stack

- Python 3.11
- Streamlit
- LightGBM, scikit-learn, SHAP
- pandas, NumPy, Plotly, Matplotlib

## Requisitos

- Python 3.11 o superior
- Dependencias declaradas en `requirements.txt`

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

## Estructura del proyecto

```
hotel_bookings/
├── app.py                # Punto de entrada
├── src/
│   ├── pestañas/         # Vistas del dashboard
│   ├── ml/               # Modelos e interpretabilidad
│   └── ui/               # Tema y componentes visuales
├── Data/                 # Datos de entrada y conjuntos de ML
├── models/               # Clasificador y codificadores serializados
└── docs/                 # Recursos gráficos y documentación
```

## Datos

Conjunto público *Hotel Booking Demand* (Antonio, Almeida y Nunes, 2019): 119 390 reservas con 32 variables correspondientes al periodo 2015–2017.
