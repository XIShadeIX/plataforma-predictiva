import numpy as np
import pandas as pd

def calcular_metricas_sku(demanda_real: pd.Series, pred_o1: np.ndarray, pred_o2: np.ndarray, factor_seguridad: float = 1.15):
    n = len(demanda_real)
    demanda = demanda_real.values
    
    pronostico_proxima_semana = float(np.mean(pred_o2[-2:]))
    cobertura_recomendada = float(pronostico_proxima_semana * factor_seguridad)
    
    quiebres_o1 = np.sum(pred_o1 < demanda)
    quiebres_o2 = np.sum(pred_o2 < demanda)
    
    tasa_quiebre_o1 = float((quiebres_o1 / n) * 100)
    tasa_quiebre_o2 = float((quiebres_o2 / n) * 100)
    
    if tasa_quiebre_o2 > 35:
        nivel_riesgo = "Alto"
        badge_color = "danger"
    elif tasa_quiebre_o2 > 20:
        nivel_riesgo = "Medio"
        badge_color = "warning"
    else:
        nivel_riesgo = "Bajo"
        badge_color = "success"
        
    mae_o2 = float(np.mean(np.abs(demanda - pred_o2)))
    mape_o2 = float(np.mean(np.abs((demanda - pred_o2) / np.maximum(demanda, 1))) * 100)

    return {
        "pronostico_semanal": round(pronostico_proxima_semana, 2),
        "cobertura_recomendada": round(cobertura_recomendada, 2),
        "tasa_quiebre_o1": round(tasa_quiebre_o1, 1),
        "tasa_quiebre_o2": round(tasa_quiebre_o2, 1),
        "nivel_riesgo": nivel_riesgo,
        "badge_color": badge_color,
        "mae_o2": round(mae_o2, 2),
        "mape_o2": round(mape_o2, 2)
    }

def calcular_metricas_portafolio(df_real, pred_o1, pred_o2, t_entren, t_resp):
    real = df_real.values
    mae_o1 = float(np.mean(np.abs(real - pred_o1)))
    mae_o2 = float(np.mean(np.abs(real - pred_o2)))
    
    mape_o1 = float(np.mean(np.abs((real - pred_o1) / np.maximum(real, 1))) * 100)
    mape_o2 = float(np.mean(np.abs((real - pred_o2) / np.maximum(real, 1))) * 100)
    
    quiebre_o1 = float((np.sum(pred_o1 < real) / len(real)) * 100)
    quiebre_o2 = float((np.sum(pred_o2 < real) / len(real)) * 100)
    
    ventas_perdidas_o1 = float((np.sum(np.maximum(0, real - pred_o1)) / np.sum(real)) * 100)
    ventas_perdidas_o2 = float((np.sum(np.maximum(0, real - pred_o2)) / np.sum(real)) * 100)
    
    fill_rate_o1 = float((np.sum(np.minimum(real, pred_o1)) / np.sum(real)) * 100)
    fill_rate_o2 = float((np.sum(np.minimum(real, pred_o2)) / np.sum(real)) * 100)
    
    otd_o1 = float((np.sum(pred_o1 >= real * 0.95) / len(real)) * 100)
    otd_o2 = float((np.sum(pred_o2 >= real * 0.95) / len(real)) * 100)
    
    return {
        "mae": {"o1": round(mae_o1, 2), "o2": round(mae_o2, 2)},
        "mape": {"o1": round(mape_o1, 2), "o2": round(mape_o2, 2)},
        "quiebre": {"o1": round(quiebre_o1, 1), "o2": round(quiebre_o2, 1)},
        "ventas_perdidas": {"o1": round(ventas_perdidas_o1, 1), "o2": round(ventas_perdidas_o2, 1)},
        "fill_rate": {"o1": round(fill_rate_o1, 1), "o2": round(fill_rate_o2, 1)},
        "otd": {"o1": round(otd_o1, 1), "o2": round(otd_o2, 1)},
        "tiempo_entrenamiento": round(t_entren, 2),
        "tiempo_respuesta": round(t_resp, 4)
    }