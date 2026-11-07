
import json
import logging
import math
import os
import threading
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(BASE_DIR, "resultados.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
ORDENES_LOG = os.path.join(DATA_DIR, "ordenes_simuladas.json")

ESCENARIOS = {
    "estable": "escenario_estable.csv",
    "campana": "escenario_campana.csv",
    "quiebre": "escenario_quiebre.csv",
}

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("plataforma")
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def cargar_resultados():
    """Lee resultados.json; devuelve (dict, None) o (None, mensaje_error)."""
    if not os.path.exists(RESULTADOS):
        return None, ("No existe resultados.json. Ejecuta entrenar_y_exportar.py "
                      "para generarlo antes de usar la plataforma.")
    try:
        with open(RESULTADOS, "r", encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, json.JSONDecodeError) as exc:
        log.exception("Error leyendo resultados.json")
        return None, f"resultados.json no se pudo leer ({type(exc).__name__})."


def validar_csv(ruta):
    """Lee un CSV y valida que sea tabular y tenga datos numéricos."""
    df = pd.read_csv(ruta)
    if df.empty:
        raise ValueError("El archivo CSV no contiene registros.")
    if df.select_dtypes("number").shape[1] == 0:
        raise ValueError("El CSV no contiene columnas numéricas.")
    return df


def error(msg, codigo=400):
    return jsonify({"error": msg}), codigo


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/data")
def get_data():
    data, err = cargar_resultados()
    if err:
        return error(err, 404)
    return jsonify(data)


@app.route("/api/simular", methods=["POST"])
def simular():
    """
    Pedido sugerido para un SKU con parámetros del usuario.

    pedido = pronóstico_diario * lead_time * (1 + safety_stock) - stock_actual

    El pronóstico semanal por SKU se toma de resultados.json
    ("pronostico_proximo": {sku: unidades/semana}). Es una ayuda de decisión
    ilustrativa: la validación de la tesis usa lead time = 0 y la política
    de reposición semanal de indicadores.py.
    """
    try:
        req = request.get_json(silent=True) or {}
        sku = req.get("sku")
        lead_time = float(req.get("lead_time", 7))
        safety_stock = float(req.get("safety_stock", 0.15))
        stock_actual = float(req.get("stock_actual", 0))

        if not (0 < lead_time <= 90):
            return error("lead_time debe estar entre 1 y 90 días.")
        if not (0 <= safety_stock <= 1):
            return error("safety_stock debe estar entre 0 y 1 (por ejemplo, 0.15).")
        if stock_actual < 0:
            return error("stock_actual no puede ser negativo.")

        data, err = cargar_resultados()
        if err:
            return error(err, 404)
        pronosticos = data.get("pronostico_proximo")
        if not isinstance(pronosticos, dict) or not pronosticos:
            return error("resultados.json no incluye 'pronostico_proximo' por SKU. "
                         "Agrégalo en entrenar_y_exportar.py.", 422)
        if sku not in pronosticos:
            return error(f"SKU no válido. Disponibles: {', '.join(pronosticos)}.")

        semanal = float(pronosticos[sku])
        demanda_cobertura = semanal / 7.0 * lead_time
        objetivo = demanda_cobertura * (1 + safety_stock)
        nuevo_pedido = max(0, math.ceil(objetivo - stock_actual))

        return jsonify({
            "sku": sku,
            "nuevo_pedido": nuevo_pedido,
            "detalle": {
                "pronostico_semanal": round(semanal, 2),
                "demanda_durante_lead_time": round(demanda_cobertura, 2),
                "nivel_objetivo_con_seguridad": round(objetivo, 2),
                "stock_actual": stock_actual,
            },
            "nota": "Cálculo ilustrativo; no forma parte de la validación retrospectiva.",
        })
    except (TypeError, ValueError):
        return error("Parámetros numéricos inválidos.")
    except Exception:
        log.exception("Error en /api/simular")
        return error("Error interno al simular.", 500)


@app.route("/api/ordenes/crear", methods=["POST"])
def crear_orden():
    """Registra una orden SIMULADA (no hay integración con ERP ni proveedor)."""
    try:
        req = request.get_json(silent=True) or {}
        sku = req.get("sku")
        obs = str(req.get("obs", "")).strip()[:300] or "Sin observaciones"
        try:
            qty = int(float(req.get("qty")))
        except (TypeError, ValueError):
            return error("La cantidad debe ser un número.")
        if qty <= 0:
            return error("La cantidad debe ser mayor que cero.")

        data, _ = cargar_resultados()
        validos = list((data or {}).get("pronostico_proximo", {}).keys())
        if validos and sku not in validos:
            return error(f"SKU no válido. Disponibles: {', '.join(validos)}.")

        orden = {"fecha": datetime.now().isoformat(timespec="seconds"),
                 "sku": sku, "cantidad": qty, "observacion": obs, "simulada": True}
        os.makedirs(DATA_DIR, exist_ok=True)
        with _lock:
            historial = []
            if os.path.exists(ORDENES_LOG):
                with open(ORDENES_LOG, "r", encoding="utf-8") as f:
                    historial = json.load(f)
            historial.append(orden)
            with open(ORDENES_LOG, "w", encoding="utf-8") as f:
                json.dump(historial, f, ensure_ascii=False, indent=2)

        return jsonify({
            "mensaje": f"Orden SIMULADA registrada: {qty} unidades"
                       f"{' de ' + sku if sku else ''}. Observación: '{obs}'. "
                       "No se envió a ningún proveedor.",
            "orden": orden,
        })
    except Exception:
        log.exception("Error en /api/ordenes/crear")
        return error("Error interno al registrar la orden.", 500)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Valida y guarda un CSV. NO reentrena el modelo."""
    try:
        if "file" not in request.files:
            return error("No se encontró el archivo.")
        archivo = request.files["file"]
        if archivo.filename == "":
            return error("Nombre de archivo vacío.")
        nombre = secure_filename(archivo.filename)
        if not nombre.lower().endswith(".csv"):
            return error("Solo se admiten archivos .csv.")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        destino = os.path.join(
            UPLOAD_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{nombre}")
        archivo.save(destino)
        try:
            df = validar_csv(destino)
        except Exception as exc:
            os.remove(destino)
            return error(f"CSV inválido: {exc}")

        return jsonify({
            "mensaje": f"Dataset '{nombre}' validado y guardado "
                       f"({len(df)} registros, {df.shape[1]} columnas). "
                       "El modelo NO se reentrena desde la web: ejecuta "
                       "entrenar_y_exportar.py para actualizar resultados.json.",
            "registros": int(len(df)),
            "columnas": list(map(str, df.columns)),
            "reentrenado": False,
        })
    except Exception:
        log.exception("Error en /api/upload")
        return error("Error interno al procesar el archivo.", 500)


@app.route("/api/cargar-escenario/<nombre>", methods=["POST"])
def cargar_escenario(nombre):
    """Valida un escenario de prueba local. NO reentrena el modelo."""
    try:
        if nombre not in ESCENARIOS:
            return error("Escenario no válido.")
        ruta = os.path.join(DATA_DIR, ESCENARIOS[nombre])
        if not os.path.exists(ruta):
            return error(f"No se encontró el archivo del escenario '{nombre}'.", 404)
        try:
            df = validar_csv(ruta)
        except Exception as exc:
            return error(f"CSV inválido: {exc}")

        return jsonify({
            "mensaje": f"Escenario '{nombre}' leído correctamente "
                       f"({len(df)} registros). El modelo no se reentrena "
                       "desde la web.",
            "registros": int(len(df)),
            "reentrenado": False,
        })
    except Exception:
        log.exception("Error en /api/cargar-escenario")
        return error("Error interno al cargar el escenario.", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000,
            debug=os.environ.get("FLASK_DEBUG") == "1")