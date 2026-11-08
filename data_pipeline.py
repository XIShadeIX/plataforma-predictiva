"""Carga reproducible de SupplyGraph; selección de SKUs solo con entrenamiento."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / 'data/source/sales_order.csv'
SOURCE_URL = ('https://raw.githubusercontent.com/ciol-researchlab/SupplyGraph/main/'
              'Raw%20Dataset/Homogenoeus/Temporal%20Data/Unit/Sales%20Order.csv')


def cargar_y_preparar_datos(n_test=8, ruta=None, incluir_metadata=False):
    path = Path(ruta).resolve() if ruta else SOURCE_PATH
    if not path.exists():
        if ruta:
            raise FileNotFoundError(path)
        print('Descargando SupplyGraph (se conservará una copia local)...', flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            contenido = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contenido)
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
    if df.empty or df.index.hasnans or df.index.has_duplicates:
        raise ValueError('Fechas vacías, inválidas o duplicadas en el CSV.')
    if not df.index.equals(pd.date_range(df.index.min(), df.index.max(), freq='D', name='Date')):
        raise ValueError('Faltan días en el CSV: no se imputan como demanda cero.')
    df = df.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all() or (df < 0).any().any():
        raise ValueError('La demanda debe ser numérica, finita y no negativa.')
    conteos = df.resample('W-SUN').size()
    semanal = df.resample('W-SUN').sum().loc[conteos.eq(7)]
    if not isinstance(n_test, int) or n_test < 1 or len(semanal) - n_test <= 14:
        raise ValueError('Se requieren más de 14 semanas completas de entrenamiento y n_test >= 1.')
    train = semanal.iloc[:-n_test]
    skus = train.sum().sort_values(ascending=False, kind='stable').head(8).index.tolist()
    series = {sku: semanal[sku].asfreq('W-SUN') for sku in skus}
    portafolio = semanal[skus].sum(axis=1).asfreq('W-SUN')
    metadata = {
        'fuente': 'SupplyGraph: Sales Order y Delivery To distributor (dataset público)',
        'url': SOURCE_URL if ruta is None else None,
        'archivo': str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'filas_diarias': len(df), 'frecuencia': 'W-SUN (lunes a domingo)',
        'semanas_excluidas': [{'fin': d.strftime('%Y-%m-%d'), 'dias': int(n)}
                              for d, n in conteos.items() if n != 7],
        'semanas_entrenamiento': len(train), 'semanas_prueba': n_test,
        'inicio_train': train.index[0].strftime('%Y-%m-%d'),
        'fin_train': train.index[-1].strftime('%Y-%m-%d'),
        'inicio_test': semanal.index[-n_test].strftime('%Y-%m-%d'),
        'fin_test': semanal.index[-1].strftime('%Y-%m-%d'),
        'seleccion_skus': '8 mayores volúmenes del entrenamiento; no clasificación ABC formal',
        'skus': skus,
    }
    result = (series, portafolio, skus)
    return (*result, metadata) if incluir_metadata else result


DELIVERY_PATH = BASE_DIR / 'data/source/delivery_to_distributor.csv'
DELIVERY_URL = SOURCE_URL.rsplit('/', 1)[0] + '/Delivery%20To%20distributor.csv'


def leer_fuente_diaria(path, url=None):
    path = Path(path)
    if not path.exists():
        if url is None:
            raise FileNotFoundError(path)
        with urllib.request.urlopen(url, timeout=60) as response:
            contenido = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporal = path.with_suffix('.tmp')
        temporal.write_bytes(contenido)
        temporal.replace(path)
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
    if df.empty or df.index.hasnans or df.index.has_duplicates:
        raise ValueError('Fechas vacías, inválidas o duplicadas en ' + path.name)
    if not df.index.equals(pd.date_range(df.index.min(), df.index.max(), freq='D', name='Date')):
        raise ValueError('Faltan días en ' + path.name + '; no se imputan como cero.')
    df = df.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all() or (df < 0).any().any():
        raise ValueError('Pedidos/entregas deben ser finitos y no negativos.')
    return df


def cargar_logistica_semanal(skus, ruta_pedidos=None, ruta_entregas=None):
    """Join explícito por semana y SKU; no acepta pérdidas silenciosas de registros."""
    path_p = Path(ruta_pedidos) if ruta_pedidos else SOURCE_PATH
    path_e = Path(ruta_entregas) if ruta_entregas else DELIVERY_PATH
    pedidos = leer_fuente_diaria(path_p, None if ruta_pedidos else SOURCE_URL)
    entregas = leer_fuente_diaria(path_e, None if ruta_entregas else DELIVERY_URL)
    if not pedidos.index.equals(entregas.index):
        raise ValueError('Pedidos y entregas tienen fechas distintas; alinear fuentes antes de calcular.')
    if len(skus) != 8 or len(set(skus)) != 8:
        raise ValueError('Se requieren exactamente ocho SKUs distintos.')
    for nombre, df in [('pedidos', pedidos), ('entregas', entregas)]:
        if set(skus) - set(df.columns):
            raise ValueError('Faltan SKUs seleccionados en ' + nombre)
    completos = pedidos.resample('W-SUN').size().eq(7)
    def semanal_largo(df, valor):
        w = df[skus].resample('W-SUN').sum().loc[completos]
        return w.rename_axis('semana').reset_index().melt(id_vars='semana', var_name='sku', value_name=valor)
    cruce = semanal_largo(pedidos, 'pedidos_totales').merge(
        semanal_largo(entregas, 'entregas_a_tiempo'),
        on=['semana','sku'], how='outer', validate='one_to_one', indicator=True)
    if not cruce['_merge'].eq('both').all():
        raise ValueError('No hay correspondencia completa por semana/SKU.')
    cruce = cruce.drop(columns='_merge').sort_values(['semana','sku']).reset_index(drop=True)
    fuentes = {nombre: {'archivo': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'url': url}
               for nombre, path, url in [('pedidos', path_p, SOURCE_URL if not ruta_pedidos else None),
                                        ('entregas', path_e, DELIVERY_URL if not ruta_entregas else None)]}
    return cruce, fuentes
