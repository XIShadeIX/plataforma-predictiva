# Plataforma predictiva ARIMA–LSTM y fichas de registro

Aplicación Flask para Python local en Windows / VS Code. El motor predictivo
usa ARIMA de statsmodels y LSTM de TensorFlow. La logística se calcula ahora
exclusivamente de pedidos y entregas registrados en SupplyGraph.

## Ejecutar en Windows (PowerShell)

Usar Python 3.12 de 64 bits. Dentro de la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe entrenar_y_exportar.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

Abrir http://127.0.0.1:5000. No hace falta activar el entorno virtual. En VS Code,
seleccionar `.venv` como intérprete. Probado en Linux/Python 3.12.14 en esta sesión;
no se afirma haber probado Windows. No requiere comandos Unix, GPU ni Colab.
Para servir solo Flask basta `requirements.txt`; el entrenamiento requiere
`requirements-train.txt`. `ejecutar_todo.py` llama al mismo pipeline oficial.

## Fuentes y cruce

Se conservan copias de `Sales Order.csv` y `Delivery To distributor.csv` en
`data/source`. Sus URLs y SHA-256 aparecen en `resultados.json`. Se valida:
fechas diarias únicas, continuidad, valores numéricos finitos y no negativos,
idénticas fechas en ambas fuentes y presencia de los ocho SKUs seleccionados.
No se rellenan entregas ausentes con ceros ni se descartan cruces fallidos.

Se suman unidades por semana lunes–domingo (W-SUN). Se excluyen los extremos
incompletos: 01/01/2023 (1 día) y 13/08/2023 (3 días). El join explícito utiliza
semana y SKU: 31 semanas completas, 8 SKUs y 248 registros. Los ocho productos
se seleccionan por volumen en entrenamiento para no utilizar información futura;
esto por sí solo no certifica la clasificación ABC/Pareto del documento.

## Ficha 1 · Precisión

MAE = promedio de errores absolutos. MAPE = promedio de errores porcentuales
absolutos. Demanda cero se excluye de MAPE y se registra su cantidad; si no hay
valores positivos se devuelve null. WAPE se conserva como complemento.
O1 es la media móvil de cuatro semanas, actualizada paso a paso. O2 es ARIMA–LSTM.
Se usan 23 semanas de entrenamiento, 8 de prueba y semilla 42. El período de
prueba termina entre 18/06/2023 y 06/08/2023. La ventana LSTM de 14 semanas deja
nueve secuencias para entrenar: muestra exploratoria, sin garantía de mejora.

La nueva ejecución mantiene MAE O1=52 293,01 / O2=70 483,85, MAPE 49,77% / 61,47%
y WAPE 31,72% / 42,75%. El híbrido sigue siendo peor que la media móvil en ese test.
La logística observada no se usa para alterar estos resultados predictivos.

## Ficha 3 · Fórmulas acordadas

Para cada semana y SKU, P son las unidades de Sales Order y E las unidades de
Delivery To distributor de esa misma semana:

- Pedidos con quiebre Q = max(P − E, 0).
- Tasa de quiebre = Q / P × 100.
- OTD = E / P × 100.
- Ventas perdidas = Q / P × 100.
- Fill Rate = unidades despachadas / unidades solicitadas × 100 = E / P × 100.

Sin pedidos (P=0), los cuatro porcentajes son null y la interfaz muestra N/D;
se conservan las cantidades. OTD y Fill Rate NO se recortan a 100%.
Los campos se expresan en unidades físicas, no en número de órdenes. El nombre
«entregas a tiempo» identifica las entregas del período conforme a la fórmula
acordada: el dataset no permite verificar fechas prometidas. Ventas perdidas y
quiebre son indicadores de déficit según esta convención, no eventos comerciales
individuales confirmados. No se utiliza el pronóstico en ninguna de esas fórmulas.

La tabla de portafolio primero suma P y E de los ocho SKUs por semana y luego
calcula Q. En esa vista un excedente de un SKU puede compensar el faltante de
otro. El detalle por SKU conserva los déficits individuales para su inspección.

Las tarjetas resumen las ocho semanas del test. Se calcula OTD/Fill Rate como
ΣE/ΣP y quiebre/ventas perdidas como ΣQ_semanal/ΣP (NO promedio simple de porcentajes).
Se evita compensar un déficit con excedentes de semanas posteriores. Por eso
quiebre + OTD no necesariamente suma 100% en el resumen del período.

Son resultados históricos observados, independientes del algoritmo. No se
presentan falsamente como resultados O2 producidos por el modelo: entrenar
un pronóstico no cambia las entregas que ya ocurrieron. Para comparar políticas
O1/O2 logísticas se necesitaría definir y validar una simulación adicional.

## Ficha 2 · Tiempos separados

- `before_request` inicia el reloj monotónico y registra fecha/hora UTC.
- `after_request` mide después de ejecutar la ruta y construir el HTML/JSON.
  Incluye carga de archivos, cómputo y serialización. También registra errores
  HTTP, como 404. Excluye red/navegador y la escritura del propio registro;
  no pretende medir consumo diferido de una respuesta streaming.
- Cabeceras: `X-Response-Time-Seconds`, `X-Request-ID` y `Server-Timing`.
  `X-Audit-Recorded` informa si el registro se pudo guardar.
- Cada petición se guarda en `auditoria/peticiones_http.jsonl` con identificador,
  método, ruta, estado, inicio, fin y duración en segundos. El dashboard muestra
  la medición real de la petición que obtuvo `/api/data`, no un dato pregrabado.
- El entrenamiento mide exclusivamente selección/ajuste ARIMA y entrenamiento
  LSTM; no incluye descarga, backtest, generación de reportes ni peticiones HTTP.
  Se registran inicio/fin UTC y segundos/minutos por cada SKU y por el modelo
  independiente del portafolio (nueve modelos). El total es su suma.
- El tiempo del backtest por paso se conserva con ese nombre, separado del
  tiempo de respuesta HTTP; no se muestra como respuesta del servidor.

## Exportaciones y auditoría

`resultados.json` contiene predicciones, métricas, logística observada, fuentes,
hashes, versiones y las mediciones de entrenamiento de la última corrida.
Los CSV son exportaciones de los mismos registros:

| Archivo | Contenido |
|---|---|
| `auditoria/ficha3_semanal_sku.csv` | 248 filas por semana y SKU |
| `auditoria/ficha3_semanal_portafolio.csv` | 31 filas semanales del agregado |
| `auditoria/ficha2_entrenamiento.csv` | 9 mediciones de la última ejecución |
| `auditoria/entrenamientos.jsonl` | Historial acumulado por corrida y modelo |
| `auditoria/peticiones_http.jsonl` | Historial acumulado de peticiones Flask |

Las tres fichas CSV también se descargan desde el dashboard. Los timestamps de
las ejecuciones corresponden al momento real de ejecución, no a 2023 (fecha de
los datos). El pronóstico siguiente es relativo al cierre del histórico, no a la
semana actual. Las curvas y tarjetas del simulador inferior siguen siendo ejemplos
ilustrativos, señalados como tales. No se envían pedidos reales a proveedores.

El JSON se reemplaza atómicamente después del entrenamiento exitoso. Las descargas
CSV web se generan del JSON vigente; los CSV locales se regeneran al entrenar.
El archivo de entrada original se conserva en `auditoria/resultados_recibidos.json`.

## GitHub

Copiar los archivos del paquete al repositorio existente conservando `.git`.
Revisar los cambios antes de publicar. No se modificó el despliegue remoto.

```powershell
git status
git diff
git add README.md .gitignore app.py data_pipeline.py indicadores.py modelo_hibrido.py entrenar_y_exportar.py templates/dashboard.html tests resultados.json data/source auditoria
git commit -m "Cruza pedidos y entregas y registra tiempos HTTP y entrenamiento"
git push origin main
```

Los logs vivos HTTP y de entrenamiento se ignoran en Git para no publicar registros
de uso por accidente. Las fichas CSV de la corrida sí pueden versionarse.
