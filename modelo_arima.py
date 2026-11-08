"""ARIMA real con selección por AIC únicamente sobre entrenamiento."""
import itertools
import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def seleccionar_mejor_orden(serie, p_range=range(4), d_range=range(2), q_range=range(3)):
    mejor_aic, mejor_orden = np.inf, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)
        for orden in itertools.product(p_range, d_range, q_range):
            try:
                fit = ARIMA(serie, order=orden).fit()
                if fit.mle_retvals.get('converged', True) and np.isfinite(fit.aic) and fit.aic < mejor_aic:
                    mejor_aic, mejor_orden = fit.aic, orden
            except (ValueError, np.linalg.LinAlgError):
                continue
    if mejor_orden is None:
        raise RuntimeError('Ningún candidato ARIMA convergió; no se sustituye por un modelo ficticio.')
    return mejor_orden, mejor_aic


def entrenar_arima(serie_train, orden=None):
    if orden is None:
        orden, _ = seleccionar_mejor_orden(serie_train)
    fit = ARIMA(serie_train, order=orden).fit()
    if not fit.mle_retvals.get('converged', True):
        raise RuntimeError('El ajuste ARIMA final no convergió.')
    return fit, orden


def pronosticar_arima_walkforward(resultado_train, serie_completa, n_test, devolver_estado=False):
    estado, predicciones = resultado_train, []
    for i in range(len(serie_completa) - n_test, len(serie_completa)):
        predicciones.append(max(float(estado.forecast(1).iloc[0]), 0.0))
        estado = estado.append(serie_completa.iloc[[i]], refit=False)
    predicciones = np.asarray(predicciones)
    return (predicciones, estado) if devolver_estado else predicciones
