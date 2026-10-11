import os
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    # Aseguramos que apunte a dashboard.html que es el archivo que tienes en la carpeta templates
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    # Lee los datos que ya calculaste localmente con tu script de 8 SKUs
    try:
        with open('resultados.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)