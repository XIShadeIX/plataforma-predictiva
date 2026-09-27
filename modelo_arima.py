"""
modelo_arima.py
Componente ARIMA (lineal) del motor de pronostico hibrido.
Requiere: pip install statsmodels

IMPORTANTE (v2): se usa pronostico WALK-FORWARD (un paso adelante), no un
pronostico "ciego" a N dias. Esto es tanto mas preciso como mas realista:
una plataforma en produccion se re-alimenta cada dia con el dato real que
ya se observo, no adivina 8 semanas de una sola vez sin retroalimentacion.
"""
import numpy as np
import pandas as pd
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def seleccionar_mejor_orden(serie: pd.Series, p_range=range(0, 4), d_range=range(0, 2), q_range=range(0, 3)):
    """Busqueda del mejor (p, d, q) por AIC."""
    mejor_aic = np.inf
    mejor_orden = (1, 1, 1)
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            resultado = ARIMA(serie, order=(p, d, q)).fit()
            if resultado.aic < mejor_aic:
                mejor_aic = resultado.aic
                mejor_orden = (p, d, q)
        except Exception:
            continue
    return mejor_orden, mejor_aic


def entrenar_arima(serie_train: pd.Series, orden=None):
    """Ajusta el modelo ARIMA una sola vez sobre el set de entrenamiento."""
    if orden is None:
        orden, _ = seleccionar_mejor_orden(serie_train)
    resultado = ARIMA(serie_train, order=orden).fit()
    return resultado, orden


def pronosticar_arima_walkforward(resultado_train, serie_completa: pd.Series, n_test: int):
    """
    Pronostico un paso adelante: para cada dia de prueba, predice el
    siguiente valor y LUEGO incorpora el dato real observado ese dia
    (mediante `.append`, que actualiza el filtro de Kalman sin re-estimar
    los parametros, por lo que es rapido) antes de predecir el dia
    siguiente. Devuelve el arreglo de predicciones (misma longitud que
    n_test).
    """
    resultado_actual = resultado_train
    predicciones = []
    n_train = len(serie_completa) - n_test

    for i in range(n_test):
        pred_siguiente = resultado_actual.forecast(steps=1).iloc[0]
        predicciones.append(max(pred_siguiente, 0))

        # revelar el dato real de ese dia y actualizar el modelo (sin refit)
        idx_real = n_train + i
        valor_real = serie_completa.iloc[[idx_real]]
        resultado_actual = resultado_actual.append(valor_real, refit=False)

    return np.array(predicciones)


if __name__ == "__main__":
    from data_pipeline import cargar_serie_categoria_a

    pedidos, _ = cargar_serie_categoria_a()
    n_test = 56
    train = pedidos.iloc[:-n_test]
    test = pedidos.iloc[-n_test:]

    resultado, orden = entrenar_arima(train)
    print("Mejor orden ARIMA (p,d,q):", orden)
    pred = pronosticar_arima_walkforward(resultado, pedidos, n_test)
    mae = np.mean(np.abs(test.values - pred))
    print("MAE ARIMA walk-forward en test:", round(mae, 2))
