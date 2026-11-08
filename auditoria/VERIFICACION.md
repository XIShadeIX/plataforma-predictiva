# Verificación de fichas logísticas y tiempos

Ejecución: 07/11/2026, Linux / Python 3.12.14. Código orientado a Windows y VS Code;
no se probó Windows en esta sesión. TensorFlow CPU 2.21.0 y statsmodels 0.15.0.

Se ejecutó `python entrenar_y_exportar.py` de principio a fin después de incorporar
el cruce de Sales Order y Delivery To distributor. Terminó correctamente. Los
hashes de código incluidos en el JSON coincidieron con los archivos ejecutados.

El cruce produjo 248 registros (31 semanas completas × 8 SKUs). Se verificaron
fechas, SKUs, ausencia de duplicados y coherencia entre las cantidades semanales
y las sumas diarias originales. Se rechazan fechas distintas, días ausentes,
SKUs faltantes, duplicados, valores negativos y no finitos.

Pasaron 11 pruebas automatizadas (`python -m unittest discover -s tests -v`).
Cubren selección sin información futura, semanas completas, MAPE con ceros,
formulación logística, excedentes, denominadores cero, agregación por semana,
recalcular los registros exportados, nueve mediciones de entrenamiento,
rutas Flask y exportaciones CSV. Una prueba con reloj controlado confirmó que
el tiempo HTTP incluye tanto lectura de datos como serialización JSON.

Se inició `python app.py` y se verificó por HTTP real en 127.0.0.1:5000:
`/`, `/api/data` y las tres descargas CSV. Todas respondieron 200, con cabecera de
duración e indicador de registro persistido. El registro de `/api/data` midió
aproximadamente 0,00359 s en esa petición; no es un tiempo constante del sistema.
El entrenamiento de los nueve modelos sumó 17,88796 s (0,29813 min).
Las mediciones detalladas y sus timestamps están en los CSV/JSONL adjuntos.

La sintaxis del JavaScript pasó `node --check`. No se completó inspección visual
en navegador: el navegador local no estaba instalado y falló su descarga.
No se afirma una validación visual del layout.

Resumen logístico del portafolio para las ocho semanas del test:
OTD y Fill Rate 99,88965%; quiebre y ventas perdidas 4,32502%.
Los porcentajes de quiebre se obtienen sumando déficits calculados por semana,
sin compensarlos con excedentes de otras semanas. Las tarjetas no representan
un promedio simple ni una comparación O1/O2 de entregas generadas por el modelo.

MAE, MAPE y WAPE predictivos permanecen iguales a la ejecución anterior:
MAPE O1=49,77%, O2=61,47%. Se conserva el resultado desfavorable del híbrido.

No se publicó en GitHub ni se modificó ningún despliegue remoto.
