import json
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Pipeline de Datos ---
def cargar_y_preparar_datos():
    print("Descargando dataset público desde GitHub (SupplyGraph)...")
    url = "https://raw.githubusercontent.com/ciol-researchlab/SupplyGraph/main/Raw%20Dataset/Homogenoeus/Temporal%20Data/Unit/Sales%20Order.csv"
    pedidos = pd.read_csv(url, parse_dates=["Date"])
    pedidos.set_index("Date", inplace=True)
    
    volumen = pedidos.sum().sort_values(ascending=False)
    skus_cat_a = volumen.head(8).index.tolist()
    
    series_skus = {}
    for sku in skus_cat_a:
        series_skus[sku] = pedidos[sku].resample('W').sum()
        
    serie_portafolio = pedidos[skus_cat_a].sum(axis=1).resample('W').sum()
    return series_skus, serie_portafolio, skus_cat_a

# --- 2. Modelo (Simplificado para evitar TensorFlow/LSTM local y ahorrar memoria) ---
# Usamos un ARIMA robusto con ajuste de suavizado como proxy del modelo híbrido para generar las métricas de demostración.
def entrenar_y_pronosticar_rapido(serie, n_test):
    train = serie.iloc[:-n_test]
    real = serie.iloc[-n_test:]
    
    # Pronóstico O1 (Método actual - Promedio móvil simple 4 periodos)
    pred_o1 = np.full(n_test, train.iloc[-4:].mean())
    
    # Pronóstico O2 (Modelo propuesto)
    pred_o2 = []
    historia = list(train)
    t0 = time.time()
    
    for t in range(n_test):
        try:
            modelo = ARIMA(historia, order=(1,1,1))
            ajuste = modelo.fit()
            yhat = ajuste.forecast()[0]
        except:
            yhat = np.mean(historia[-3:]) # Respaldo si no converge
            
        pred_o2.append(max(0, yhat)) # No permitir pronósticos negativos
        historia.append(real.iloc[t])
        
    t_resp = (time.time() - t0) / n_test
    return pred_o1, np.array(pred_o2), t_resp

# --- 3. Indicadores ---
def calcular_metricas_sku(demanda_real, pred_o1, pred_o2):
    demanda = demanda_real.values
    pronostico_semana = float(np.mean(pred_o2[-2:]))
    cobertura = float(pronostico_semana * 1.15)
    
    quiebres_o2 = np.sum(pred_o2 < demanda)
    tasa_quiebre_o2 = float((quiebres_o2 / len(demanda)) * 100)
    
    riesgo = "Alto" if tasa_quiebre_o2 > 35 else "Medio" if tasa_quiebre_o2 > 20 else "Bajo"
    badge = "danger" if riesgo == "Alto" else "warning" if riesgo == "Medio" else "success"
    
    return {
        "pronostico_semanal": round(pronostico_semana, 2),
        "cobertura_recomendada": round(cobertura, 2),
        "tasa_quiebre_o2": round(tasa_quiebre_o2, 1),
        "nivel_riesgo": riesgo,
        "badge_color": badge
    }

def calcular_metricas_portafolio(real, pred_o1, pred_o2, t_resp):
    real_vals = real.values
    return {
        "mape": {
            "o1": round(float(np.mean(np.abs((real_vals - pred_o1) / np.maximum(real_vals, 1))) * 100), 2),
            "o2": round(float(np.mean(np.abs((real_vals - pred_o2) / np.maximum(real_vals, 1))) * 100), 2)
        },
        "quiebre": {
            "o1": round(float((np.sum(pred_o1 < real_vals) / len(real_vals)) * 100), 1),
            "o2": round(float((np.sum(pred_o2 < real_vals) / len(real_vals)) * 100), 1)
        },
        "ventas_perdidas": {
            "o1": round(float((np.sum(np.maximum(0, real_vals - pred_o1)) / np.sum(real_vals)) * 100), 1),
            "o2": round(float((np.sum(np.maximum(0, real_vals - pred_o2)) / np.sum(real_vals)) * 100), 1)
        },
        "fill_rate": {
            "o1": round(float((np.sum(np.minimum(real_vals, pred_o1)) / np.sum(real_vals)) * 100), 1),
            "o2": round(float((np.sum(np.minimum(real_vals, pred_o2)) / np.sum(real_vals)) * 100), 1)
        },
        "otd": {
            "o1": round(float((np.sum(pred_o1 >= real_vals * 0.95) / len(real_vals)) * 100), 1),
            "o2": round(float((np.sum(pred_o2 >= real_vals * 0.95) / len(real_vals)) * 100), 1)
        },
        "tiempo_respuesta": round(t_resp, 4)
    }

# --- 4. Orquestación y Exportación ---
def ejecutar():
    print("Iniciando procesamiento multi-SKU...")
    series_skus, serie_portafolio, skus_cat_a = cargar_y_preparar_datos()
    n_test = 8
    
    recomendaciones = []
    
    for sku_id in skus_cat_a:
        serie = series_skus[sku_id]
        pred_o1, pred_o2, _ = entrenar_y_pronosticar_rapido(serie, n_test)
        
        mets = calcular_metricas_sku(serie.iloc[-n_test:], pred_o1, pred_o2)
        mets["sku"] = sku_id
        recomendaciones.append(mets)
        print(f"[{sku_id}] procesado.")
        
    orden = {"Alto": 0, "Medio": 1, "Bajo": 2}
    recomendaciones.sort(key=lambda x: orden[x["nivel_riesgo"]])
    skus_riesgo = sum(1 for r in recomendaciones if r["nivel_riesgo"] == "Alto")
    
    print("Procesando portafolio consolidado...")
    pred_o1_port, pred_o2_port, t_resp_port = entrenar_y_pronosticar_rapido(serie_portafolio, n_test)
    mets_globales = calcular_metricas_portafolio(serie_portafolio.iloc[-n_test:], pred_o1_port, pred_o2_port, t_resp_port)
    
    fechas = [d.strftime('%Y-%m-%d') for d in serie_portafolio.index[-n_test:]]
    datos = {
        "banner": {
            "skus_en_riesgo": skus_riesgo,
            "total_skus": len(skus_cat_a),
            "mensaje": f"{skus_riesgo} de {len(skus_cat_a)} productos en riesgo ALTO" if skus_riesgo > 0 else "Cobertura adecuada"
        },
        "recomendaciones_skus": recomendaciones,
        "metricas_globales": mets_globales,
        "grafico": {
            "fechas": fechas,
            "demanda_real": serie_portafolio.iloc[-n_test:].tolist(),
            "pronostico_o1": pred_o1_port.tolist(),
            "pronostico_o2": pred_o2_port.tolist()
        }
    }
    
    with open("resultados.json", "w") as f:
        json.dump(datos, f, indent=4)
        
    print("¡Éxito! resultados.json actualizado.")

if __name__ == "__main__":
    ejecutar()