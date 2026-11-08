"""Errores predictivos e indicadores logísticos de pedidos/entregas, separados."""
import numpy as np


def validar(demanda, pred):
    y, p = np.asarray(demanda, dtype=float), np.asarray(pred, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('Demanda y predicción deben ser vectores no vacíos de igual tamaño.')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError('Demanda y predicciones deben ser finitas y no negativas.')
    return y, p


def metricas(demanda, pred):
    y, p = validar(demanda, pred)
    positivos = y > 0
    total = y.sum()
    # MAPE indefinido con y=0: excluye ceros y registra su cantidad.
    return {
        'mae': round(float(np.mean(abs(y-p))), 2),
        'mape': round(float(np.mean(abs(y[positivos]-p[positivos])/y[positivos])*100), 2) if positivos.any() else None,
        'wape': round(float(abs(y-p).sum()/total*100), 2) if total else None,
        'ceros_excluidos_mape': int((~positivos).sum()),
    }


def calcular_metricas_sku(demanda_real, pred_o1, pred_o2, pronostico_proximo, factor_seguridad=1.15):
    a, b = metricas(demanda_real, pred_o1), metricas(demanda_real, pred_o2)
    return {'pronostico_semanal': round(float(pronostico_proximo), 2),
            'cobertura_recomendada': round(float(pronostico_proximo)*factor_seguridad, 2),
            'mae_o2': b['mae'], 'mape_o2': b['mape'], 'wape_o2': b['wape'],
            'ceros_excluidos_mape': b['ceros_excluidos_mape']}


def calcular_metricas_portafolio(df_real, pred_o1, pred_o2, t_entren, t_resp):
    a, b = metricas(df_real, pred_o1), metricas(df_real, pred_o2)
    resultado = {k: {'o1': a[k], 'o2': b[k]} for k in a if k != 'ceros_excluidos_mape'}
    return {**resultado, 'ceros_excluidos_mape': a['ceros_excluidos_mape'],
            'tiempo_entrenamiento_segundos': t_entren, 'tiempo_entrenamiento_minutos': t_entren/60,
            'tiempo_backtest_por_paso_segundos': t_resp}


def indicadores_logisticos(pedidos, entregas):
    """Ficha 3: magnitudes en unidades. No limita OTD/Fill Rate a 100%."""
    p, e = float(pedidos), float(entregas)
    if not np.isfinite([p, e]).all() or min(p, e) < 0:
        raise ValueError('Pedidos y entregas deben ser finitos y no negativos.')
    deficit = max(p-e, 0.0)
    return {'pedidos_totales': p, 'entregas_a_tiempo': e,
            'unidades_solicitadas': p, 'unidades_despachadas': e,
            'pedidos_con_quiebre': deficit,
            'tasa_quiebre': deficit/p*100 if p else None,
            'otd': e/p*100 if p else None,
            'ventas_perdidas': deficit/p*100 if p else None,
            'fill_rate': e/p*100 if p else None}


def resumen_logistico(filas):
    p = sum(r['pedidos_totales'] for r in filas)
    e = sum(r['entregas_a_tiempo'] for r in filas)
    q = sum(r['pedidos_con_quiebre'] for r in filas)
    resultado = indicadores_logisticos(p, e)
    # Suma déficits calculados por semana; un excedente posterior no los borra.
    resultado.update(pedidos_con_quiebre=q, tasa_quiebre=q/p*100 if p else None,
                     ventas_perdidas=q/p*100 if p else None,
                     registros=len(filas), periodos_sin_pedidos=sum(r['pedidos_totales']==0 for r in filas))
    return resultado


def construir_logistica(cruce, fechas_test, fuentes):
    semanas = []
    skus = []
    for row in cruce.itertuples(index=False):
        skus.append({'semana': row.semana.strftime('%Y-%m-%d'), 'sku': row.sku,
                     **indicadores_logisticos(row.pedidos_totales, row.entregas_a_tiempo)})
    for semana, df in cruce.groupby('semana', sort=True):
        # La ficha del portafolio resta totales semanales; no suma déficits de SKUs.
        semanas.append({'semana': semana.strftime('%Y-%m-%d'),
                        **indicadores_logisticos(df.pedidos_totales.sum(), df.entregas_a_tiempo.sum())})
    fechas = {d.strftime('%Y-%m-%d') for d in fechas_test}
    return {'fuentes': fuentes, 'unidad': 'unidades físicas, no cantidad de órdenes',
            'otd_interpretacion': 'Cumplimiento por volumen; entregas del período usadas como entregas a tiempo. No verifica fechas prometidas.',
            'politica_ceros': 'Sin pedidos: porcentajes null (N/D); cantidades conservadas.',
            'agregacion': 'Portafolio: sumar 8 SKUs por semana y luego calcular déficit. Resumen: razón de sumas; sumar déficits semanales sin compensarlos entre semanas.',
            'periodo_resumen': {'inicio': min(fechas), 'fin': max(fechas)},
            'semanas_portafolio': semanas, 'semanas_sku': skus,
            'resumen_periodo_prueba': resumen_logistico([r for r in semanas if r['semana'] in fechas]),
            'resumen_por_sku_prueba': {sku: resumen_logistico([r for r in skus if r['sku']==sku and r['semana'] in fechas])
                                      for sku in sorted(cruce.sku.unique())}}
