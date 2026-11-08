import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from data_pipeline import cargar_logistica_semanal, cargar_y_preparar_datos, SOURCE_PATH, DELIVERY_PATH
from indicadores import indicadores_logisticos, construir_logistica


class LogisticaTest(unittest.TestCase):
    def test_formulas_y_excedentes(self):
        m=indicadores_logisticos(100,80)
        self.assertEqual(m['pedidos_con_quiebre'],20)
        self.assertEqual([m[k] for k in ['tasa_quiebre','otd','ventas_perdidas','fill_rate']], [20,80,20,80])
        m=indicadores_logisticos(100,120)
        self.assertEqual(m['pedidos_con_quiebre'],0)
        self.assertEqual(m['ventas_perdidas'],0)
        self.assertEqual(m['otd'],120)
        self.assertEqual(m['fill_rate'],120)
        self.assertEqual(indicadores_logisticos(100,100)['tasa_quiebre'],0)

    def test_ceros_invalidos(self):
        for e in [0,10]:
            m=indicadores_logisticos(0,e)
            self.assertEqual(m['pedidos_con_quiebre'],0)
            for k in ['tasa_quiebre','otd','ventas_perdidas','fill_rate']:self.assertIsNone(m[k])
        for p,e in [(-1,2),(1,-2),(float('nan'),1),(1,float('inf'))]:
            with self.assertRaises(ValueError):indicadores_logisticos(p,e)

    def test_agregar_antes_de_restar_y_no_compensar_semanas(self):
        # Excedente de B compensa A dentro de una semana; no entre semanas.
        c=pd.DataFrame({'semana':pd.to_datetime(['2023-01-08']*2+['2023-01-15']*2),
                        'sku':['A','B','A','B'], 'pedidos_totales':[100,100,100,100],
                        'entregas_a_tiempo':[60,160,70,70]})
        l=construir_logistica(c,pd.to_datetime(['2023-01-08','2023-01-15']),{})
        self.assertEqual([r['pedidos_con_quiebre'] for r in l['semanas_portafolio']],[0,60])
        self.assertEqual(l['resumen_periodo_prueba']['ventas_perdidas'],15)
        self.assertEqual(l['resumen_periodo_prueba']['otd'],90)
        self.assertEqual(l['semanas_sku'][0]['pedidos_con_quiebre'],40)

    def test_cruce_fuentes_reales_y_rechazo_desalineacion(self):
        _,_,skus=cargar_y_preparar_datos()
        c,_=cargar_logistica_semanal(skus)
        self.assertEqual(len(c),31*8)
        self.assertFalse(c.duplicated(['semana','sku']).any())
        p=pd.read_csv(SOURCE_PATH,parse_dates=['Date']); e=pd.read_csv(DELIVERY_PATH,parse_dates=['Date'])
        r=c.iloc[0];ini=r.semana-pd.Timedelta(days=6)
        self.assertAlmostEqual(r.pedidos_totales,p.loc[p.Date.between(ini,r.semana),r.sku].sum())
        self.assertAlmostEqual(r.entregas_a_tiempo,e.loc[e.Date.between(ini,r.semana),r.sku].sum())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'delivery.csv'
            e.iloc[1:].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'fechas distintas'):cargar_logistica_semanal(skus,ruta_entregas=path)
            e.drop(columns=skus[0]).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Faltan SKUs'):cargar_logistica_semanal(skus,ruta_entregas=path)
            pd.concat([e,e.iloc[[0]]]).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'duplicadas'):cargar_logistica_semanal(skus,ruta_entregas=path)

    def test_exportacion_coherente(self):
        from app import RESULTADOS
        d=json.loads(Path(RESULTADOS).read_text())
        for key in ['semanas_sku','semanas_portafolio']:
            for r in d['logistica_observada'][key]:
                esperado=indicadores_logisticos(r['pedidos_totales'],r['entregas_a_tiempo'])
                for k,v in esperado.items():self.assertEqual(r[k],v)
        self.assertEqual(len(d['mediciones_entrenamiento']),9)
        self.assertAlmostEqual(sum(m['segundos'] for m in d['mediciones_entrenamiento']),d['metricas_globales']['tiempo_entrenamiento_segundos'])
        self.assertNotIn('tiempo_respuesta',d['metricas_globales'])
        self.assertNotIn('quiebre',d['metricas_globales'])


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.log=Path(self.temp.name)/'http.jsonl'
        self.p=patch('app.HTTP_LOG',str(self.log));self.p.start();self.addCleanup(self.p.stop)

    def test_incluye_lectura_y_serializacion_json(self):
        import app as modulo
        clock=[100.0];real_jsonify=modulo.jsonify
        def cargar():clock[0]+=2;return {'prueba':'ok'},None
        def serializar(*args,**kwargs):
            result=real_jsonify(*args,**kwargs);clock[0]+=.25;return result
        with patch('app.time.perf_counter',side_effect=lambda:clock[0]), patch('app.cargar_resultados',side_effect=cargar), patch('app.jsonify',side_effect=serializar):
            r=modulo.app.test_client().get('/api/data')
        self.assertEqual(r.json,{'prueba':'ok'})
        self.assertEqual(float(r.headers['X-Response-Time-Seconds']),2.25)
        registro=json.loads(self.log.read_text())
        self.assertEqual(registro['tiempo_respuesta_segundos'],2.25)
        self.assertEqual(registro['peticion_id'],r.headers['X-Request-ID'])
        self.assertEqual(registro['ruta'],'/api/data')

    def test_render_html_errores_y_csv(self):
        import app as modulo
        client=modulo.app.test_client()
        for url in ['/', '/no-existe', '/api/exportar/ficha3_sku','/api/exportar/ficha3_portafolio','/api/exportar/ficha2_entrenamiento']:
            r=client.get(url)
            self.assertEqual(r.status_code,404 if url=='/no-existe' else 200)
            self.assertGreaterEqual(float(r.headers['X-Response-Time-Seconds']),0)
            self.assertEqual(r.headers['X-Audit-Recorded'],'true')
            if 'exportar' in url:self.assertIn('attachment',r.headers['Content-Disposition'])
        rows=[json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(rows),5)
        self.assertEqual(len(set(r['peticion_id'] for r in rows)),5)
        self.assertEqual(rows[1]['estado_http'],404)


if __name__=='__main__':unittest.main()
