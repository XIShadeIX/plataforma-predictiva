"""
indicadores.py
Calcula los 8 indicadores definidos en la Matriz de Operacionalizacion
(Tabla 2) a partir de las predicciones del modelo y la demanda real.
"""
import numpy as np
import pandas as pd

FACTOR_SEGURIDAD = 1.15  # colchon de stock aplicado sobre el pronostico


def mae(actual: np.ndarray, pronostico: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pronostico)))


def mape(actual: np.ndarray, pronostico: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - pronostico[mask]) / actual[mask])) * 100)


def simular_politica_inventario(actual: np.ndarray, pronostico: np.ndarray,
                                 fechas: pd.DatetimeIndex, factor_seguridad: float = FACTOR_SEGURIDAD) -> dict:
    """
    Simula la politica de abastecimiento: stock_disponible = pronostico * factor_seguridad.
    A partir de ahi deriva Tasa de Quiebre, Ventas Perdidas, Fill Rate y OTD.
    """
    stock_disponible = np.clip(pronostico, 0, None) * factor_seguridad
    entregado = np.minimum(actual, stock_disponible)
    quiebre = actual > stock_disponible
    deficit = np.clip(actual - stock_disponible, 0, None)

    tasa_quiebre = float(quiebre.mean() * 100)
    ventas_perdidas = float(deficit.sum() / actual.sum() * 100)

    df = pd.DataFrame({"actual": actual, "entregado": entregado}, index=fechas)
    semanal = df.resample("W").sum()
    ratio_semanal = (semanal["entregado"] / semanal["actual"]).clip(upper=1.0)
    fill_rate = float(ratio_semanal.mean() * 100)
    otd = float((ratio_semanal >= 0.95).mean() * 100)

    return {
        "tasa_quiebre_pct": round(tasa_quiebre, 1),
        "ventas_perdidas_pct": round(ventas_perdidas, 1),
        "fill_rate_pct": round(fill_rate, 1),
        "otd_pct": round(otd, 1),
    }


def calcular_todos_los_indicadores(actual: np.ndarray, pronostico: np.ndarray,
                                    fechas: pd.DatetimeIndex, tiempo_entrenamiento_seg: float,
                                    tiempo_respuesta_seg: float) -> dict:
    resultado = {
        "MAE": round(mae(actual, pronostico), 2),
        "MAPE_pct": round(mape(actual, pronostico), 2),
        "tiempo_entrenamiento_seg": round(tiempo_entrenamiento_seg, 4),
        "tiempo_respuesta_seg": round(tiempo_respuesta_seg, 6),
    }
    resultado.update(simular_politica_inventario(actual, pronostico, fechas))
    return resultado
