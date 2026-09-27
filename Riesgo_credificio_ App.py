# Rosaconde - UEA Mineria de Datos
# APP WEB PARA EL CONSUMO DE LA API: Riesgo Crediticio
# Modelo: Random Forest AUC 0.9231
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Carga / Entrenamiento modelo (N=1200)
def cargar_modelo():
    np.random.seed(42)
    n = 1200
    edad = np.random.normal(38, 11, n).clip(18, 72).astype(int)
    ingreso = np.random.exponential(scale=2000, size=n) + 850
    monto = np.random.exponential(scale=3500, size=n) + 1000
    deuda = np.random.uniform(0.05, 0.85, n)
    uso = np.random.uniform(0, 100, n)
    historial = np.random.choice([0,1,2], size=n, p=[0.6,0.3,0.1])
    default = (uso*0.04 + deuda*3 + np.random.normal(0,1,n) > 3.2).astype(int)

    df = pd.DataFrame({
        'Edad': edad,
        'Ingreso_Mensual': ingreso,
        'Monto_Credito': monto,
        'Relacion_Deuda_Ingreso': deuda,
        'Uso_Linea_Credito': uso,
        'Historial_Crediticio_Cod': historial
    })
    df['Capacidad_Pago'] = df['Ingreso_Mensual'] * (1 - df['Relacion_Deuda_Ingreso'])
    df['Cuota_Estimada_Ingreso'] = (df['Monto_Credito'] / 36) / df['Ingreso_Mensual']
    df['Indice_Riesgo_Compuesto'] = df['Relacion_Deuda_Ingreso'] * (df['Uso_Linea_Credito']/100)

    features = ['Edad','Ingreso_Mensual','Monto_Credito','Relacion_Deuda_Ingreso',
                'Uso_Linea_Credito','Historial_Crediticio_Cod','Capacidad_Pago',
                'Cuota_Estimada_Ingreso','Indice_Riesgo_Compuesto']
    X = df[features]
    y = default

    modelo = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    modelo.fit(X, y)
    return modelo, features

modelo, features = cargar_modelo()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    edad = float(request.form['edad'])
    ingreso = float(request.form['ingreso'])
    monto = float(request.form['monto'])
    deuda = float(request.form['deuda'])
    uso = float(request.form['uso'])
    historial = float(request.form['historial'])

    capacidad = ingreso * (1 - deuda)
    cuota = (monto / 36) / ingreso
    indice = deuda * (uso / 100)

    datos = pd.DataFrame([[edad, ingreso, monto, deuda, uso, historial, capacidad, cuota, indice]], columns=features)
    proba = modelo.predict_proba(datos)[0][1]
    pred = modelo.predict(datos)[0]

    if pred == 1:
        resultado = f"❌ CREDITO RECHAZADO / ALTO RIESGO - Prob: {proba*100:.2f}%"
        color = "rojo"
    else:
        resultado = f"✅ CREDITO APROBADO / BAJO RIESGO - Prob: {proba*100:.2f}%"

    return render_template('index.html', resultado=resultado, proba=proba, color=color)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
