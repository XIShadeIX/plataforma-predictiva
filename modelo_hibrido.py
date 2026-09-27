"""
modelo_hibrido.py
Orquesta el modelo hibrido ARIMA + LSTM con pronostico WALK-FORWARD:
  1. ARIMA se entrena una vez y luego pronostica un paso adelante,
     actualizandose con cada dato real que se va revelando (sin re-fit).
  2. LSTM se entrena una vez sobre los residuos de entrenamiento y luego
     corrige cada paso de prueba usando SOLO residuos reales ya observados.
  3. Pronostico final de cada dia = pronostico_ARIMA(ese dia) + correccion_LSTM(ese dia)

Por que walk-forward y no un pronostico "ciego" a 56 dias:
Un pronostico ciego de 8 semanas hace que el LSTM recursivo (y el propio
ARIMA a largo plazo) acumulen error sobre sus propias predicciones pasadas
en vez de sobre la realidad. Ademas, no es representativo de como operaria
la plataforma en produccion: un sistema real se re-alimenta cada dia con
el dato que ya se observo. Ver diagnostico y comparacion de resultados en
el README.

Este es el codigo de PRODUCCION (usa statsmodels + TensorFlow reales).
Ejecutar en un entorno con internet (Google Colab recomendado).
"""
import time
import numpy as np
import pandas as pd

from data_pipeline import cargar_serie_categoria_a
from modelo_arima import entrenar_arima, pronosticar_arima_walkforward
from modelo_lstm import entrenar_lstm_sobre_residuos, predecir_residuos_walkforward, VENTANA


def entrenar_modelo_hibrido(serie_completa: pd.Series, n_test: int):
    serie_train = serie_completa.iloc[:-n_test]
    serie_test = serie_completa.iloc[-n_test:]

    t0 = time.time()

    # 1) ARIMA: se entrena una vez sobre el train
    resultado_arima, orden = entrenar_arima(serie_train)

    # 2) Pronostico ARIMA walk-forward sobre todo el periodo de prueba
    #    (esto ya incluye la actualizacion dia a dia con el dato real)
    pred_arima_test = pronosticar_arima_walkforward(resultado_arima, serie_completa, n_test)

    # 3) Residuos de ENTRENAMIENTO (para entrenar la red LSTM)
    pred_train_arima = resultado_arima.fittedvalues.values
    residuos_train = serie_train.values[-len(pred_train_arima):] - pred_train_arima

    # 4) Residuos REALES del periodo de prueba (conocidos en el backtest,
    #    se le presentan al LSTM walk-forward paso a paso, nunca de una vez)
    residuos_test_reales = serie_test.values - pred_arima_test

    # 5) LSTM: se entrena una vez sobre los residuos de entrenamiento
    modelo_lstm, escalador = entrenar_lstm_sobre_residuos(residuos_train)

    tiempo_entrenamiento = time.time() - t0

    return {
        "resultado_arima": resultado_arima,
        "orden_arima": orden,
        "modelo_lstm": modelo_lstm,
        "escalador": escalador,
        "residuos_train": residuos_train,
        "residuos_test_reales": residuos_test_reales,
        "pred_arima_test": pred_arima_test,
        "tiempo_entrenamiento_seg": tiempo_entrenamiento,
    }


def pronosticar_hibrido(modelo_entrenado: dict, n_test: int):
    t0 = time.time()

    correccion_lstm = predecir_residuos_walkforward(
        modelo_entrenado["modelo_lstm"],
        modelo_entrenado["escalador"],
        modelo_entrenado["residuos_train"],
        modelo_entrenado["residuos_test_reales"],
    )
    pred_hibrido = np.clip(modelo_entrenado["pred_arima_test"] + correccion_lstm, 0, None)

    tiempo_respuesta = (time.time() - t0) / n_test
    return pred_hibrido, tiempo_respuesta


if __name__ == "__main__":
    pedidos, entregas = cargar_serie_categoria_a()
    N_TEST = 56

    print("Entrenando modelo hibrido ARIMA + LSTM (walk-forward)...")
    modelo = entrenar_modelo_hibrido(pedidos, N_TEST)
    print("Orden ARIMA seleccionado:", modelo["orden_arima"])
    print("Tiempo de entrenamiento (s):", round(modelo["tiempo_entrenamiento_seg"], 3))

    pred, tiempo_respuesta = pronosticar_hibrido(modelo, N_TEST)
    test = pedidos.iloc[-N_TEST:]
    mae = np.mean(np.abs(test.values - pred))
    mape = np.mean(np.abs((test.values - pred) / np.where(test.values == 0, np.nan, test.values))) * 100
    print("MAE hibrido walk-forward en test:", round(mae, 2))
    print("MAPE hibrido walk-forward en test (%):", round(mape, 2))
    print("Tiempo de respuesta promedio por prediccion (s):", round(tiempo_respuesta, 6))
