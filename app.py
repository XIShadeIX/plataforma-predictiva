"""
app.py
Plataforma web predictiva - Variable Independiente del proyecto de tesis.
Sirve un dashboard con los indicadores O1 (pretest) vs O2 (postest) y la
serie de pronostico del modelo hibrido ARIMA-LSTM.

Ejecutar:
    python entrenar_y_exportar.py   # una sola vez, genera resultados.json
    python app.py                   # levanta el servidor en localhost:5000
"""
import json
from flask import Flask, render_template

app = Flask(__name__)


def cargar_resultados():
    with open("resultados.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def dashboard():
    datos = cargar_resultados()
    return render_template("dashboard.html", datos=datos)


@app.route("/api/resultados")
def api_resultados():
    """Endpoint JSON, por si se quiere consumir desde otro frontend."""
    return cargar_resultados()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
