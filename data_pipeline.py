"""
data_pipeline.py
Carga y preparacion de los datos SupplyGraph para el modelo ARIMA-LSTM.

Uso (en Google Colab o entorno local con internet):
    !git clone https://github.com/ciol-researchlab/SupplyGraph.git
    from data_pipeline import cargar_serie_categoria_a
    serie = cargar_serie_categoria_a()
"""
import pandas as pd
import os

# SKUs de Categoria A, obtenidos mediante analisis de Pareto (ABC) sobre
# la demanda total del periodo 2023-01-01 a 2023-08-09. Ver Anexo 1
# (Matriz de consistencia) y Anexo 2 (Evidencias de obtencion de datos).
CATEGORIA_A_SKUS = [
    'SOS001L12P', 'SOS005L04P', 'SOS002L09P', 'ATN01K24P',
    'SOS500M24P', 'POV001L24P', 'SOS003L04P', 'POV500M24P'
]

RUTA_BASE = "SupplyGraph/Raw Dataset/Homogenoeus/Temporal Data/Unit"


def cargar_crudos(ruta_base: str = RUTA_BASE):
    """Carga los archivos Sales Order y Delivery To distributor originales."""
    pedidos = pd.read_csv(f"{ruta_base}/Sales Order.csv", parse_dates=["Date"])
    entregas = pd.read_csv(f"{ruta_base}/Delivery To distributor.csv", parse_dates=["Date"])
    return pedidos.set_index("Date"), entregas.set_index("Date")


def cargar_serie_categoria_a(ruta_base: str = RUTA_BASE):
    """
    Devuelve dos series diarias (pandas.Series) agregadas para los 8 SKUs
    de Categoria A: pedidos (demanda solicitada) y entregas (despachadas).
    """
    pedidos, entregas = cargar_crudos(ruta_base)
    serie_pedidos = pedidos[CATEGORIA_A_SKUS].sum(axis=1)
    serie_entregas = entregas[CATEGORIA_A_SKUS].sum(axis=1)
    return serie_pedidos, serie_entregas


def hacer_features_de_rezago(serie: pd.Series, lags=(1, 2, 3, 7, 14)):
    """Construye una matriz de variables de rezago (lag features) para el
    componente autorregresivo / de entrenamiento supervisado."""
    df = pd.DataFrame({"y": serie})
    for lag in lags:
        df[f"lag_{lag}"] = serie.shift(lag)
    return df.dropna()


if __name__ == "__main__":
    so, dd = cargar_serie_categoria_a()
    print("Dias cargados:", len(so))
    print(so.describe())
