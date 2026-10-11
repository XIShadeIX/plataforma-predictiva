import json
import time
import numpy as np
from data_pipeline import cargar_y_preparar_datos
from modelo_hibrido import entrenar_modelo_hibrido, pronosticar_hibrido
from indicadores import calcular_metricas_sku, calcular_metricas_portafolio

def ejecutar_pipeline_multi_sku():
    print("Iniciando entrenamiento individual de modelos ARIMA-LSTM por SKU...")
    series_skus, serie_portafolio, skus_cat_a = cargar_y_preparar_datos()
    n_test = 8 # Semanas de evaluacion
    
    recomendaciones_skus = []
    t_entrenamiento_total = 0
    
    for sku_id in skus_cat_a:
        t0 = time.time()
        serie_sku = series_skus[sku_id]
        
        modelo_sku = entrenar_modelo_hibrido(serie_sku, n_test)
        pred_o2_sku, t_resp_sku = pronosticar_hibrido(modelo_sku, n_test)
        
        serie_train = serie_sku.iloc[:-n_test]
        pred_o1_sku = np.full(n_test, serie_train.iloc[-4:].mean())
        
        t_entren_sku = time.time() - t0
        t_entrenamiento_total += t_entren_sku
        real_sku = serie_sku.iloc[-n_test:]
        
        metricas = calcular_metricas_sku(real_sku, pred_o1_sku, pred_o2_sku)
        metricas["sku"] = sku_id
        recomendaciones_skus.append(metricas)
        print(f"[{sku_id}] entrenado - Riesgo: {metricas['nivel_riesgo']}")
        
    orden_riesgo = {"Alto": 0, "Medio": 1, "Bajo": 2}
    recomendaciones_skus.sort(key=lambda x: orden_riesgo[x["nivel_riesgo"]])
    skus_en_riesgo = sum(1 for r in recomendaciones_skus if r["nivel_riesgo"] == "Alto")
    
    print("Entrenando modelo consolidado de portafolio...")
    t0_port = time.time()
    modelo_port = entrenar_modelo_hibrido(serie_portafolio, n_test)
    pred_o2_port, t_resp_port = pronosticar_hibrido(modelo_port, n_test)
    pred_o1_port = np.full(n_test, serie_portafolio.iloc[:-n_test].iloc[-4:].mean())
    t_entren_port = time.time() - t0_port
    
    metricas_globales = calcular_metricas_portafolio(
        serie_portafolio.iloc[-n_test:], pred_o1_port, pred_o2_port, 
        t_entrenamiento_total + t_entren_port, t_resp_port
    )
    
    fechas_test = [d.strftime('%Y-%m-%d') for d in serie_portafolio.index[-n_test:]]
    datos_exportar = {
        "banner": {
            "skus_en_riesgo": skus_en_riesgo,
            "total_skus": len(skus_cat_a),
            "mensaje": f"{skus_en_riesgo} de {len(skus_cat_a)} productos en riesgo ALTO de quiebre" if skus_en_riesgo > 0 else "Todos los productos presentan cobertura adecuada"
        },
        "recomendaciones_skus": recomendaciones_skus,
        "metricas_globales": metricas_globales,
        "grafico": {
            "fechas": fechas_test,
            "demanda_real": serie_portafolio.iloc[-n_test:].tolist(),
            "pronostico_o1": pred_o1_port.tolist(),
            "pronostico_o2": pred_o2_port.tolist()
        }
    }
    
    with open("resultados.json", "w") as f:
        json.dump(datos_exportar, f, indent=4)
        
    print("¡Completado! resultados.json generado correctamente.")

if __name__ == "__main__":
    ejecutar_pipeline_multi_sku()