"""Entrenamiento y evaluación separados; sin importaciones del pipeline de datos."""
import time
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from modelo_arima import entrenar_arima, pronosticar_arima_walkforward
from modelo_lstm import entrenar_lstm_sobre_residuos, predecir_residuos_walkforward, predecir_un_residuo, VENTANA


def entrenar_modelo_hibrido(serie_completa, n_test, seed=42):
    if n_test < 1 or len(serie_completa) - n_test <= VENTANA:
        raise ValueError('Historia insuficiente para entrenar ARIMA-LSTM.')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    train = serie_completa.iloc[:-n_test]
    inicio_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    arima, orden = entrenar_arima(train)
    residuos = train.to_numpy() - arima.fittedvalues.to_numpy()
    lstm, escalador = entrenar_lstm_sobre_residuos(residuos)
    duracion = time.perf_counter() - t0
    fin_utc = datetime.now(timezone.utc).isoformat()
    return {'inicio_entrenamiento_utc': inicio_utc, 'fin_entrenamiento_utc': fin_utc, 'resultado_arima': arima, 'orden_arima': orden, 'modelo_lstm': lstm,
            'escalador': escalador, 'residuos_train': residuos, 'serie_completa': serie_completa,
            'n_test': n_test, 'tiempo_entrenamiento_seg': duracion}


def pronosticar_hibrido(modelo_entrenado, n_test):
    m = modelo_entrenado
    if n_test != m['n_test']:
        raise ValueError('n_test difiere del usado en entrenamiento.')
    t0 = time.perf_counter()
    arima, estado = pronosticar_arima_walkforward(m['resultado_arima'], m['serie_completa'], n_test, True)
    residuos_test = m['serie_completa'].iloc[-n_test:].to_numpy() - arima
    correccion = predecir_residuos_walkforward(m['modelo_lstm'], m['escalador'], m['residuos_train'], residuos_test)
    pred = np.maximum(arima + correccion, 0)
    t_respuesta = (time.perf_counter() - t0) / n_test
    m['pred_arima_test'] = arima
    historial = np.concatenate([m['residuos_train'], residuos_test])
    siguiente_arima = max(float(estado.forecast(1).iloc[0]), 0)
    m['pronostico_proximo'] = max(0.0, siguiente_arima + predecir_un_residuo(m['modelo_lstm'], m['escalador'], historial))
    return pred, t_respuesta
