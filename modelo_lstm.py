"""
modelo_lstm.py
Componente LSTM (no lineal) del motor de pronostico hibrido: se entrena
sobre los RESIDUOS del componente ARIMA, siguiendo el enfoque de
correccion de errores de Wang et al. (2022), ya citado en el Cap. II.

Requiere: pip install tensorflow scikit-learn

IMPORTANTE (v2): la prediccion en produccion es WALK-FORWARD. En cada paso
de prueba, la ventana de entrada usa los RESIDUOS REALES ya observados
(actual - prediccion_ARIMA de dias anteriores), nunca residuos generados
por el propio modelo. Esto evita el error compuesto (compounding error)
que se detecto al pronosticar 56 dias "a ciegas" de una sola vez.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

VENTANA = 14  # dias de historial que ve la red para predecir el siguiente


def construir_secuencias(serie_1d: np.ndarray, ventana: int = VENTANA):
    X, y = [], []
    for i in range(ventana, len(serie_1d)):
        X.append(serie_1d[i - ventana:i])
        y.append(serie_1d[i])
    X = np.array(X).reshape(-1, ventana, 1)
    y = np.array(y)
    return X, y


def construir_modelo_lstm(ventana: int = VENTANA):
    modelo = Sequential([
        LSTM(32, activation="tanh", input_shape=(ventana, 1), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mae")
    return modelo


def entrenar_lstm_sobre_residuos(residuos_train: np.ndarray, epochs: int = 150, ventana: int = VENTANA):
    """Entrena la red LSTM para predecir el residuo (error) que deja ARIMA."""
    escalador = MinMaxScaler(feature_range=(-1, 1))
    residuos_esc = escalador.fit_transform(residuos_train.reshape(-1, 1)).flatten()

    X, y = construir_secuencias(residuos_esc, ventana)

    modelo = construir_modelo_lstm(ventana)
    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    modelo.fit(X, y, epochs=epochs, batch_size=8, verbose=0, callbacks=[early_stop])

    return modelo, escalador


def predecir_residuos_walkforward(modelo, escalador, residuos_train: np.ndarray,
                                   residuos_reales_test: np.ndarray, ventana: int = VENTANA):
    """
    Predice la correccion de residuo para cada paso de prueba usando
    UNICAMENTE residuos reales ya observados (los de entrenamiento, y a
    medida que avanza, los reales del propio periodo de prueba que ya
    se revelaron). Nunca reutiliza una prediccion propia como si fuera dato
    real -- asi se elimina el error compuesto.
    """
    historial = list(residuos_train[-ventana:])  # arranca con los ultimos reales de train
    predicciones = []

    for i in range(len(residuos_reales_test)):
        ventana_actual = np.array(historial[-ventana:]).reshape(-1, 1)
        ventana_esc = escalador.transform(ventana_actual).flatten()
        entrada = ventana_esc.reshape(1, ventana, 1)
        pred_esc = modelo.predict(entrada, verbose=0)[0, 0]
        pred = escalador.inverse_transform([[pred_esc]])[0, 0]
        predicciones.append(pred)

        # se revela el residuo REAL de este dia (ya conocido en el backtest)
        # y se usa para construir la ventana del dia siguiente, no la prediccion.
        historial.append(residuos_reales_test[i])

    return np.array(predicciones)
