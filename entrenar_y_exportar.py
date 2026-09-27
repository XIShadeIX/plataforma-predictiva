"""
entrenar_y_exportar.py
Ejecuta el pipeline completo (datos -> ARIMA+LSTM -> indicadores) UNA VEZ
y guarda los resultados en resultados.json, que luego sirve la app Flask.

Ejecutar:
    python entrenar_y_exportar.py
"""
import json
import numpy as np
import pandas as pd

from data_pipeline import cargar_serie_categoria_a
from modelo_hibrido import entrenar_modelo_hibrido, pronosticar_hibrido
from indicadores import calcular_todos_los_indicadores, mae, mape

DIAS_TEST = 56  # 8 semanas


def pronostico_ingenuo(serie: pd.Series, dias_test: int) -> np.ndarray:
    """Baseline O1: media movil de 4 semanas (28 dias), igual que en el
    diagnostico inicial de la Tabla 1."""
    pred = serie.shift(1).rolling(28).mean()
    return pred.iloc[-dias_test:].values


def main():
    pedidos, entregas = cargar_serie_categoria_a()
    train, test = pedidos.iloc[:-DIAS_TEST], pedidos.iloc[-DIAS_TEST:]
    fechas_test = test.index

    print("== Entrenando modelo hibrido ARIMA + LSTM (walk-forward) ==")
    modelo = entrenar_modelo_hibrido(pedidos, DIAS_TEST)
    pred_hibrido, tiempo_respuesta = pronosticar_hibrido(modelo, DIAS_TEST)

    print("== Calculando baseline O1 (pronostico ingenuo) ==")
    pred_naive = pronostico_ingenuo(pedidos, DIAS_TEST)

    O1 = calcular_todos_los_indicadores(
        test.values, pred_naive, fechas_test,
        tiempo_entrenamiento_seg=0.0, tiempo_respuesta_seg=0.0,
    )
    O2 = calcular_todos_los_indicadores(
        test.values, pred_hibrido, fechas_test,
        tiempo_entrenamiento_seg=modelo["tiempo_entrenamiento_seg"],
        tiempo_respuesta_seg=tiempo_respuesta,
    )

    resultados = {
        "periodo_test": f"{fechas_test[0].date()} a {fechas_test[-1].date()}",
        "orden_arima": list(modelo["orden_arima"]),
        "O1_pretest": O1,
        "O2_postest": O2,
        "serie_actual": test.values.tolist(),
        "serie_pred_naive": pred_naive.tolist(),
        "serie_pred_hibrido": pred_hibrido.tolist(),
        "fechas": [str(d.date()) for d in fechas_test],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("Resultados guardados en resultados.json")
    print(json.dumps({"O1": O1, "O2": O2}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
