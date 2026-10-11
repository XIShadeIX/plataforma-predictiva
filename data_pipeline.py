import pandas as pd

def cargar_y_preparar_datos():
    print("Descargando dataset público desde GitHub (SupplyGraph)...")
    # Leemos directamente desde la URL cruda para no depender de archivos locales
    url = "https://raw.githubusercontent.com/ciol-researchlab/SupplyGraph/main/Raw%20Dataset/Homogenoeus/Temporal%20Data/Unit/Sales%20Order.csv"
    
    pedidos = pd.read_csv(url, parse_dates=["Date"])
    pedidos.set_index("Date", inplace=True)
    
    # Identificar los 8 SKUs Categoría A (por volumen total histórico)
    volumen = pedidos.sum().sort_values(ascending=False)
    skus_cat_a = volumen.head(8).index.tolist()
    
    series_skus = {}
    for sku in skus_cat_a:
        # Agrupar por semana para estabilizar el pronóstico
        series_skus[sku] = pedidos[sku].resample('W').sum()
        
    # Serie consolidada del portafolio Categoría A
    serie_portafolio = pedidos[skus_cat_a].sum(axis=1).resample('W').sum()
    
    return series_skus, serie_portafolio, skus_cat_a