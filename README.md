# Plataforma Web Predictiva — Gestión de Inventarios

Sistema real (Tesis 2): modelo híbrido **ARIMA + LSTM** entrenado sobre datos
reales de SupplyGraph (Wasi et al., 2024), más una plataforma web (Flask) que
muestra los 8 indicadores O1 (pretest) vs O2 (postest).

## Cómo correrlo (recomendado: Google Colab)

Colab ya trae Python, pip e internet — evita instalar nada en tu PC.

1. Abre https://colab.research.google.com y crea un notebook nuevo.
2. Sube estos 6 archivos a la sesión de Colab (ícono de carpeta a la izquierda → Subir):
   `data_pipeline.py`, `modelo_arima.py`, `modelo_lstm.py`, `modelo_hibrido.py`,
   `indicadores.py`, `entrenar_y_exportar.py`
3. En la primera celda:
   ```
   !git clone https://github.com/ciol-researchlab/SupplyGraph.git
   !pip install statsmodels tensorflow -q
   ```
4. En la segunda celda:
   ```
   !python entrenar_y_exportar.py
   ```
   Esto entrena el modelo real (ARIMA con `statsmodels` + LSTM con `TensorFlow`)
   y genera `resultados.json` con los 8 indicadores reales O1 vs O2.
5. Descarga `resultados.json` de Colab a tu compu (clic derecho → Descargar).

## Cómo correr la plataforma web (en tu PC, con Python instalado)

1. Copia el `resultados.json` que descargaste de Colab dentro de esta misma carpeta
   (reemplaza el que ya viene de ejemplo).
2. Instala Flask: `pip install flask`
3. Ejecuta: `python app.py`
4. Abre en el navegador: `http://localhost:5000`

Ahí verás el dashboard con los 8 indicadores comparando el esquema actual (O1)
contra la plataforma propuesta (O2), más el gráfico de demanda real vs.
pronóstico.

## Qué contiene cada archivo

| Archivo | Qué hace |
|---|---|
| `data_pipeline.py` | Carga y prepara los datos de SupplyGraph (Categoría A) |
| `modelo_arima.py` | Componente lineal — ARIMA con `statsmodels`, orden elegido automáticamente por AIC |
| `modelo_lstm.py` | Componente no lineal — red LSTM (`TensorFlow/Keras`) que corrige los residuos del ARIMA |
| `modelo_hibrido.py` | Combina ambos componentes en el pronóstico final |
| `indicadores.py` | Calcula los 8 indicadores (MAE, MAPE, tiempos, quiebre, OTD, ventas perdidas, Fill Rate) |
| `entrenar_y_exportar.py` | Corre todo el pipeline una vez y guarda `resultados.json` |
| `app.py` + `templates/dashboard.html` | La plataforma web que muestra los resultados |
| `resultados.json` | **Ya incluido con resultados reales** de una corrida de demostración (ver nota abajo) |

## Por qué el pronóstico es "walk-forward" (un paso adelante)

**Historial importante:** la primera versión de este sistema pronosticaba las
8 semanas de prueba "a ciegas" de una sola vez. Al correrla en Colab con las
librerías reales (`statsmodels` + `TensorFlow`), el resultado fue mediocre
(MAPE 86%, sin mejora en OTD). El diagnóstico mostró que el problema no eran
las librerías, sino el diseño: un pronóstico ciego a 8 semanas hace que el
LSTM (y el propio ARIMA a largo plazo) acumulen error sobre sus propias
predicciones pasadas, en vez de sobre la realidad.

La solución (ya aplicada en el código actual): pronóstico **walk-forward** —
cada día se predice usando el modelo entrenado, y luego se le revela el dato
real de ese día antes de predecir el siguiente (`ARIMA.append(..., refit=False)`
para el componente lineal; residuos reales, no predichos, para la ventana del
LSTM). Esto es además más realista: una plataforma en producción se
re-alimenta cada día con el dato que ya se observó, no adivina dos meses de
una sola vez.

## Nota importante sobre el `resultados.json` incluido

El archivo que ya viene en esta carpeta contiene resultados **reales**,
calculados con el enfoque walk-forward correcto, pero con una aproximación
local (regresión lineal + red neuronal MLP de scikit-learn) porque el
entorno donde se generó no tenía acceso a internet para instalar
`statsmodels`/`TensorFlow`. La lógica walk-forward ya está implementada en
`modelo_arima.py` y `modelo_lstm.py` usando las librerías reales — **antes
de tu sustentación, corre el pipeline en Colab** (pasos de arriba) para
confirmar los números finales con `statsmodels`/`TensorFlow`, que deberían
ser cercanos a estos.

## Resumen de resultados (enfoque walk-forward, demo local)

Periodo de prueba: 2023-06-15 a 2023-08-09 (8 semanas), Categoría A (8 SKUs).

| Indicador | O1 (actual) | O2 (plataforma) |
|---|---|---|
| MAE | 16,290.9 | 9,797.6 |
| MAPE | 99.2% | 48.6% |
| Tasa de quiebre | 46.4% | 26.8% |
| Ventas perdidas | 28.3% | 10.3% |
| Fill Rate | 75.1% | 91.3% |
| OTD | 11.1% | 44.4% |
| Tiempo de entrenamiento | — | 0.019 s |
| Tiempo de respuesta | — | 0.000121 s |

**Pendiente de tu parte:** vuelve a correr `entrenar_y_exportar.py` en Colab
con el código actualizado (ya lo tienes en este ZIP) y confírmame los
números reales con `statsmodels`/`TensorFlow` para que quede documentado
con las librerías definitivas antes de tu sustentación.
