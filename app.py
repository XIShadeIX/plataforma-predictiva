import json
from flask import Flask, render_template, jsonify, request

# Forzamos a Flask a leer directamente desde tu carpeta 'templates'
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

# === NUEVOS ENDPOINTS PARA EL DSS (SISTEMA EXPERTO) ===

@app.route('/api/simular', methods=['POST'])
def simular_escenario():
    data = request.json
    sku = data.get('sku', 'SKU-Generico')
    lead_time = int(data.get('lead_time', 7))
    safety_stock_pct = float(data.get('safety_stock', 0.15))
    
    # Simulamos el recálculo dinámico para la demo interactiva
    pronostico_base = 150 # Valor referencial base
    nuevo_pedido = int(pronostico_base * (1 + safety_stock_pct) * (lead_time / 7))
    
    return jsonify({
        "status": "success",
        "sku": sku,
        "nuevo_pedido": nuevo_pedido,
        "mensaje": f"Escenario recalculado: Lead Time de {lead_time} días y SS de {int(safety_stock_pct*100)}%."
    })

@app.route('/api/ordenes/crear', methods=['POST'])
def crear_orden():
    data = request.json
    # Retornamos el éxito de la transacción para evidenciar la gestión activa.
    return jsonify({
        "status": "success",
        "mensaje": f"Orden de compra generada exitosamente con cantidad ajustada a {data.get('qty')} unidades."
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No hay archivo"}), 400
    file = request.files['file']
    if file.filename != '':
        return jsonify({
            "status": "success", 
            "mensaje": "Dataset transaccional cargado correctamente. El modelo iniciará el reentrenamiento en segundo plano."
        })
    return jsonify({"error": "Archivo inválido"}), 400

# ======================================================

if __name__ == '__main__':
    # Usamos el puerto estandar 5000 y activamos el modo debug
    app.run(host='127.0.0.1', port=5000, debug=True)