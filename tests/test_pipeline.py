import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from data_pipeline import cargar_y_preparar_datos
from indicadores import metricas


class DatosTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch_http = patch('app.HTTP_LOG', str(Path(self.temp.name)/'http.jsonl'))
        self.patch_http.start()
        self.addCleanup(self.patch_http.stop)

    def test_semanas_completas_y_seleccion_sin_futuro(self):
        fechas = pd.date_range('2023-01-01', periods=221)
        df = pd.DataFrame({f'S{i}':np.full(221, float(i+1)) for i in range(9)})
        df['Date'] = fechas
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'datos.csv'; df.to_csv(p,index=False)
            a, _, skus, md = cargar_y_preparar_datos(ruta=p, incluir_metadata=True)
            self.assertEqual(md['semanas_entrenamiento'],23)
            self.assertEqual(md['semanas_excluidas'],[{'fin':'2023-01-01','dias':1},{'fin':'2023-08-13','dias':3}])
            self.assertNotIn('S0',skus)
            df.loc[df.Date >= '2023-06-12','S0']=1e9
            df.to_csv(p,index=False)
            b, _, skus2 = cargar_y_preparar_datos(ruta=p)
            self.assertEqual(skus,skus2)
            for sku in skus: np.testing.assert_array_equal(a[sku].iloc[:-8],b[sku].iloc[:-8])

    def test_no_imputa_dias_ausentes(self):
        df=pd.DataFrame({'Date':pd.date_range('2023-01-01',periods=221),'S':1}).drop(10)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'datos.csv';df.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Faltan días'):cargar_y_preparar_datos(ruta=p)

    def test_metricas_ceros_y_conservacion(self):
        m=metricas([0,10,20],[5,5,25])
        self.assertEqual(m['mape'],37.5)
        self.assertEqual(m['ceros_excluidos_mape'],1)
        self.assertIsNone(metricas([0],[5])['mape'])
        self.assertGreater(metricas([1],[10])['mape'],100)

    def test_json_y_flask(self):
        from app import app, RESULTADOS
        data=json.loads(Path(RESULTADOS).read_text())
        g=data['grafico'];m=data['metricas_globales']
        for k in ('o1','o2'):
            recalculado=metricas(g['demanda_real'],g[f'pronostico_{k}'])
            for nombre in ('mae','mape','wape'):
                self.assertEqual(m[nombre][k],recalculado[nombre])
        for sku,item in data['evaluacion_skus'].items():
            self.assertEqual(len(item['fechas']),8)
            self.assertEqual(item['secuencias_lstm'],9)
            self.assertTrue(np.isfinite(item['pronostico_o2']).all())
        cliente=app.test_client()
        self.assertEqual(cliente.get('/').status_code,200)
        self.assertEqual(cliente.get('/api/data').get_json(),data)
        sku=next(iter(data['pronostico_proximo']))
        r=cliente.post('/api/simular',json={'sku':sku,'lead_time':7,'safety_stock':0.15,'stock_actual':0})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['nuevo_pedido'],int(np.ceil(data['pronostico_proximo'][sku]*1.15)))
        self.assertEqual(cliente.post('/api/simular',json={'sku':'inexistente'}).status_code,400)
        self.assertEqual(cliente.post('/api/cargar-escenario/campana').status_code,200)
        with tempfile.TemporaryDirectory() as d, patch('app.ORDENES_LOG',str(Path(d)/'orders.json')):
            r=cliente.post('/api/ordenes/crear',json={'sku':sku,'qty':10})
            self.assertEqual(r.status_code,200)
            self.assertTrue(r.json['orden']['simulada'])


if __name__=='__main__':unittest.main()
