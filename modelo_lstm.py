"""LSTM real sobre residuos ARIMA. Una observación corresponde a una semana."""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

VENTANA = 14  # semanas; se conserva el valor original, sin ajuste sobre el test


def construir_secuencias(serie_1d, ventana=VENTANA):
    X = [serie_1d[i-ventana:i] for i in range(ventana, len(serie_1d))]
    y = serie_1d[ventana:]
    if not X:
        raise ValueError(f'Se requieren más de {ventana} residuos para entrenar LSTM.')
    return np.asarray(X, dtype=np.float32).reshape(-1, ventana, 1), np.asarray(y, dtype=np.float32)


def construir_modelo_lstm(ventana=VENTANA):
    modelo = Sequential([Input(shape=(ventana, 1)), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(1)])
    modelo.compile(optimizer='adam', loss='mae')
    return modelo


def entrenar_lstm_sobre_residuos(residuos_train, epochs=150, ventana=VENTANA):
    escalador = MinMaxScaler(feature_range=(-1, 1))
    valores = escalador.fit_transform(np.asarray(residuos_train).reshape(-1, 1)).flatten()
    X, y = construir_secuencias(valores, ventana)
    modelo = construir_modelo_lstm(ventana)
    # Dataset explícito limita hilos y evita agotar recursos en equipos pequeños.
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(8)
    opciones = tf.data.Options()
    opciones.threading.private_threadpool_size = 1
    opciones.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(opciones)
    historia = modelo.fit(dataset, epochs=epochs, verbose=0, shuffle=False,
                          callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)])
    modelo.epocas_ejecutadas = len(historia.history['loss'])
    return modelo, escalador


def predecir_un_residuo(modelo, escalador, historial, ventana=VENTANA):
    entrada = escalador.transform(np.asarray(historial[-ventana:]).reshape(-1, 1)).reshape(1, ventana, 1)
    pred = float(modelo(entrada, training=False).numpy()[0, 0])
    return float(escalador.inverse_transform([[pred]])[0, 0])


def predecir_residuos_walkforward(modelo, escalador, residuos_train, residuos_reales_test, ventana=VENTANA):
    historial = list(residuos_train)
    predicciones = []
    for real in residuos_reales_test:
        predicciones.append(predecir_un_residuo(modelo, escalador, historial, ventana))
        historial.append(real)  # se revela DESPUÉS de predecir
    return np.asarray(predicciones)
