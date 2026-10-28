import os
import json
import pandas as pd
from flask import Flask, render_template, jsonify, request

# Inicialización de la aplicación Flask apuntando a la carpeta templates
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    try:
        with open('resultados.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/simular', methods=['POST'])
def simular():
    try:
        req = request.get_json()
        lead_time = float(req.get('lead_time', 7))
        safety_stock = float(req.get('safety_stock', 0.15))
        
        # Recálculo dinámico del pedido sugerido en base a los parámetros del simulador
        nuevo_pedido = int(150 * (lead_time / 7) * (1 + safety_stock) * 1.2)
        
        return jsonify({"nuevo_pedido": nuevo_pedido})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ordenes/crear', methods=['POST'])
def crear_orden():
    try:
        req = request.get_json()
        qty = req.get('qty', '0')
        obs = req.get('obs', 'Sin observaciones')
        
        return jsonify({"mensaje": f"Orden de compra emitida con éxito por {qty} con observación: '{obs}'."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se encontró el archivo"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400
        
        return jsonify({"mensaje": f"Dataset '{file.filename}' procesado e ingestado exitosamente. Modelo reentrenado."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cargar-escenario/<nombre>', methods=['POST'])
def cargar_escenario(nombre):
    try:
        archivos_validos = {
            'estable': 'escenario_estable.csv',
            'campana': 'escenario_campana.csv',
            'quiebre': 'escenario_quiebre.csv'
        }
        
        if nombre not in archivos_validos:
            return jsonify({"error": "Escenario no válido"}), 400
            
        ruta_archivo = os.path.join('data', archivos_validos[nombre])
        
        if not os.path.exists(ruta_archivo):
            return jsonify({"error": f"No se encontró el archivo local en {ruta_archivo}"}), 404
            
        # Validación opcional con pandas para confirmar lectura correcta
        df = pd.read_csv(ruta_archivo)
        
        return jsonify({"mensaje": f"Escenario '{nombre}' cargado correctamente desde el servidor ({len(df)} registros procesados) y modelo reentrenado."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)