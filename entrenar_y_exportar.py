"""Ejecutar: python entrenar_y_exportar.py. Exporta solo después del éxito total."""
import os
# Configurar antes de importar TensorFlow/NumPy. CPU y un hilo para repetibilidad.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import hashlib
import importlib.metadata
import json
import platform
import uuid
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from data_pipeline import cargar_y_preparar_datos, cargar_logistica_semanal
from modelo_hibrido import entrenar_modelo_hibrido, pronosticar_hibrido
from modelo_lstm import VENTANA
from indicadores import calcular_metricas_sku, calcular_metricas_portafolio, construir_logistica

BASE_DIR = Path(__file__).resolve().parent


def baseline_walkforward(serie, n_test):
    inicio = len(serie)-n_test
    return np.array([serie.iloc[i-4:i].mean() for i in range(inicio,len(serie))])


def registro(serie, modelo, o1, o2):
    n = modelo['n_test']
    return {'fechas': serie.index[-n:].strftime('%Y-%m-%d').tolist(),
            'demanda_real': serie.iloc[-n:].tolist(), 'pronostico_o1': o1.tolist(),
            'pronostico_o2': o2.tolist(), 'pronostico_arima': modelo['pred_arima_test'].tolist(),
            'orden_arima': list(modelo['orden_arima']),
            'epocas_lstm': modelo['modelo_lstm'].epocas_ejecutadas,
            'secuencias_lstm': len(modelo['residuos_train'])-VENTANA,
            'pronostico_proximo': modelo['pronostico_proximo']}


def ejecutar_pipeline_multi_sku():
    n_test, seed = 8, 42
    series, portafolio, skus, fuente = cargar_y_preparar_datos(n_test, incluir_metadata=True)
    cruce, fuentes_logisticas = cargar_logistica_semanal(skus)
    logistica = construir_logistica(cruce, portafolio.index[-n_test:], fuentes_logisticas)
    corrida_id = str(uuid.uuid4())
    mediciones = []
    def registrar_entrenamiento(sku, modelo):
        mediciones.append({'corrida_id': corrida_id, 'sku': sku, 'modulo': 'ARIMA-LSTM',
                           'inicio_utc': modelo['inicio_entrenamiento_utc'],
                           'fin_utc': modelo['fin_entrenamiento_utc'],
                           'segundos': modelo['tiempo_entrenamiento_seg'],
                           'minutos': modelo['tiempo_entrenamiento_seg']/60})
    recomendaciones, detalle, proximos = [], {}, {}
    total_entrenamiento = 0.0
    for sku in skus:
        print(f'Entrenando ARIMA + LSTM real: {sku}...', flush=True)
        serie = series[sku]
        modelo = entrenar_modelo_hibrido(serie, n_test, seed)
        registrar_entrenamiento(sku, modelo)
        o2, _ = pronosticar_hibrido(modelo, n_test)
        o1 = baseline_walkforward(serie, n_test)
        total_entrenamiento += modelo['tiempo_entrenamiento_seg']
        proximos[sku] = modelo['pronostico_proximo']
        item = calcular_metricas_sku(serie.iloc[-n_test:], o1, o2, proximos[sku])
        log_sku = logistica['resumen_por_sku_prueba'][sku]
        tasa = log_sku['tasa_quiebre']
        riesgo = 'N/D' if tasa is None else 'Alto' if tasa > 35 else 'Medio' if tasa > 20 else 'Bajo'
        recomendaciones.append({'sku': sku, **item, 'logistica_observada': log_sku, 'nivel_riesgo': riesgo})
        detalle[sku] = registro(serie, modelo, o1, o2)
        print(f"  MAPE={item['mape_o2']}% | orden={modelo['orden_arima']}", flush=True)
    print('Entrenando modelo consolidado de portafolio...', flush=True)
    modelo = entrenar_modelo_hibrido(portafolio, n_test, seed)
    registrar_entrenamiento('PORTAFOLIO', modelo)
    o2, respuesta = pronosticar_hibrido(modelo, n_test)
    o1 = baseline_walkforward(portafolio, n_test)
    globales = calcular_metricas_portafolio(portafolio.iloc[-n_test:], o1, o2,
                    total_entrenamiento+modelo['tiempo_entrenamiento_seg'], respuesta)
    recomendaciones.sort(key=lambda x: {'Alto':0,'Medio':1,'Bajo':2,'N/D':3}[x['nivel_riesgo']])
    riesgo = sum(r['nivel_riesgo']=='Alto' for r in recomendaciones)
    versiones = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','scikit-learn','keras','flask']}
    import tensorflow as tf
    versiones['tensorflow'] = tf.__version__
    datos = {
        'metadata': {'generado_utc': datetime.now(timezone.utc).isoformat(), 'modelo':'ARIMA + LSTM (statsmodels + TensorFlow)',
            'semilla': seed, 'python': platform.python_version(), 'plataforma': platform.platform(),
            'versiones': versiones, 'datos': fuente,
            'hashes_codigo': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE_DIR.glob('*.py'))},
            'protocolo':'Backtest semanal a un paso; parámetros fijos; actualización con observaciones pasadas.',
            'baseline':'O1: media móvil de las 4 semanas anteriores, actualizada en cada paso.',
            'portafolio':'Modelo independiente del agregado; no suma de predicciones por SKU.',
            'proxima_semana_fin': (portafolio.index[-1] + __import__('pandas').Timedelta(days=7)).strftime('%Y-%m-%d'),
            'advertencias': [
                'Benchmark público: las entregas observadas no demuestran un efecto causal del modelo.',
                'O1 es una línea base matemática, no un pretest observado de la empresa.',
                'Logística calculada exclusivamente de Sales Order y Delivery To distributor; no del pronóstico.',
                'OTD según fórmula acordada: cumplimiento por volumen; no verifica puntualidad contractual.',
                'Solo 23 semanas de entrenamiento (9 secuencias LSTM) y 8 de prueba: evidencia exploratoria.',
                'Pronóstico siguiente relativo al final del histórico, no a la fecha actual.',
                'MAPE excluye demanda cero y no se recorta; WAPE es una métrica complementaria.',
            ]},
        'banner': {'skus_en_riesgo': riesgo, 'total_skus':len(skus),
                   'mensaje':f'{riesgo} de {len(skus)} productos con déficit histórico alto de entregas'},
        'recomendaciones_skus': recomendaciones, 'pronostico_proximo':proximos,
        'metricas_globales':globales, 'grafico':registro(portafolio, modelo, o1, o2),
        'evaluacion_skus': detalle, 'logistica_observada': logistica,
        'mediciones_entrenamiento': mediciones, 'corrida_id': corrida_id,
    }
    destino = BASE_DIR/'resultados.json'
    temporal = destino.with_suffix('.json.tmp')
    temporal.write_text(json.dumps(datos, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporal.replace(destino)
    auditoria = BASE_DIR/'auditoria'
    auditoria.mkdir(exist_ok=True)
    for nombre, filas in [('ficha3_semanal_sku.csv', logistica['semanas_sku']),
                          ('ficha3_semanal_portafolio.csv', logistica['semanas_portafolio']),
                          ('ficha2_entrenamiento.csv', mediciones)]:
        pd.DataFrame(filas).to_csv(auditoria/nombre, index=False, encoding='utf-8-sig')
    with (auditoria/'entrenamientos.jsonl').open('a', encoding='utf-8') as archivo:
        for medicion in mediciones:
            archivo.write(json.dumps(medicion, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'Completado: {destino}\nMAPE: {globales["mape"]}\nWAPE: {globales["wape"]}', flush=True)
    return datos


if __name__ == '__main__':
    ejecutar_pipeline_multi_sku()
