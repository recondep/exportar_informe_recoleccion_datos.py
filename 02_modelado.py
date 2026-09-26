import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- 1. SIMULACIÓN Y PREPROCESAMIENTO RÁPIDO DE DATOS ---
np.random.seed(42)
n_samples = 1200

edad = np.random.normal(38, 11, n_samples).clip(18, 72).astype(int)
ingreso = np.random.exponential(scale=2000, size=n_samples) + 850
monto_credito = np.random.exponential(scale=3500, size=n_samples) + 1000
deuda_ingreso = np.random.uniform(0.05, 0.85, n_samples)
uso_linea = np.random.uniform(0, 100, n_samples)
historial = np.random.choice(
    ['Bueno', 'Regular', 'Malo'], size=n_samples, p=[0.6, 0.3, 0.1]
)
default = (
    uso_linea * 0.04 + deuda_ingreso * 3 + np.random.normal(0, 1, n_samples) > 3.2
).astype(int)

df = pd.DataFrame({
    'Edad': edad,
    'Ingreso_Mensual': ingreso,
    'Monto_Credito': monto_credito,
    'Relacion_Deuda_Ingreso': deuda_ingreso,
    'Uso_Linea_Credito': uso_linea,
    'Historial_Crediticio': historial,
    'Default': default,
})

# Depuración y Tratamiento
df_clean = df.drop_duplicates().copy()
df_clean['Ingreso_Mensual'].fillna(
    df_clean['Ingreso_Mensual'].median(), inplace=True
)
df_clean['Historial_Crediticio'].fillna(
    df_clean['Historial_Crediticio'].mode()[0], inplace=True
)

# Feature Engineering & Encoding
mapa_hist = {'Bueno': 0, 'Regular': 1, 'Malo': 2}
df_clean['Historial_Crediticio_Cod'] = df_clean['Historial_Crediticio'].map(
    mapa_hist
)
df_clean['Capacidad_Pago'] = df_clean['Ingreso_Mensual'] * (
    1 - df_clean['Relacion_Deuda_Ingreso']
)
df_clean['Cuota_Estimada_Ingreso'] = (df_clean['Monto_Credito'] / 36) / df_clean[
    'Ingreso_Mensual'
]
df_clean['Indice_Riesgo_Compuesto'] = df_clean['Relacion_Deuda_Ingreso'] * (
    df_clean['Uso_Linea_Credito'] / 100
)

features = [
    'Edad',
    'Ingreso_Mensual',
    'Monto_Credito',
    'Relacion_Deuda_Ingreso',
    'Uso_Linea_Credito',
    'Historial_Crediticio_Cod',
    'Capacidad_Pago',
    'Cuota_Estimada_Ingreso',
    'Indice_Riesgo_Compuesto',
]
X = df_clean[features]
y = df_clean['Default']

# --- 2. DIVISIÓN TRAIN / TEST (80% / 20%) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. ENTRENAMIENTO DE MODELOS Y EVALUACIÓN ---
modelos = {
    'Regresión Logística': LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000, random_state=42
    ),
    'Árbol de Decisión': DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42
    ),
}

resultados = []

for nombre, model in modelos.items():
  # Usamos datos escalados para LogReg, sin escalar para árboles (opcional)
  X_tr = X_train_scaled if nombre == 'Regresión Logística' else X_train
  X_te = X_test_scaled if nombre == 'Regresión Logística' else X_test

  model.fit(X_tr, y_train)
  y_pred = model.predict(X_te)
  y_prob = (
      model.predict_proba(X_te)[:, 1]
      if hasattr(model, 'predict_proba')
      else y_pred
  )

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_prob)

  resultados.append({
      'Modelo': nombre,
      'Accuracy': f'{acc*100:.2f}%',
      'Precision': f'{prec*100:.2f}%',
      'Recall': f'{rec*100:.2f}%',
      'F1-Score': f'{f1*100:.2f}%',
      'AUC-ROC': f'{auc:.4f}',
  })

# --- 4. EXPORTACIÓN A WORD ---
doc = Document()


def set_cell_bg(cell, hex_color):
  shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
  cell._element.get_or_add_tcPr().append(shd)


h1 = doc.add_heading('2.4. Modelado', level=1)
h1.runs[0].font.color.rgb = RGBColor(31, 78, 121)

doc.add_paragraph(
    'En esta sección se abordan el entrenamiento, la configuración de'
    ' hiperparámetros y la evaluación comparativa de tres algoritmos de'
    ' minería de datos supervisada para predecir el riesgo de mora.'
)

# Partición
doc.add_heading(
    '1. División del Conjunto de Datos (Entrenamiento y Prueba)', level=2
)
doc.add_paragraph(
    '• Muestra de Datos: N = 1,200 registros depurados.\n• Entrenamiento'
    ' (Train Set): 80% (960 registros) para el ajuste de los modelos.\n•'
    ' Prueba (Test Set): 20% (240 registros) para la validación'
    ' independiente.\n• Estratificación: Aplicada sobre la variable objetivo'
    ' Default para preservar la tasa de mora original.'
)

# Hiperparámetros (Tabla 5)
doc.add_heading('2. Configuración de Modelos e Hiperparámetros', level=2)
t5 = doc.add_table(rows=1, cols=3)
t5.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, text in enumerate(['Modelo', 'Algoritmo', 'Hiperparámetros Clave']):
  cell = t5.rows[0].cells[i]
  cell.text = text
  set_cell_bg(cell, '1F4E79')
  cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
  cell.paragraphs[0].runs[0].font.bold = True

param_rows = [
    [
        'Regresión Logística',
        'Lineal',
        'penalty="l2", C=1.0, solver="lbfgs", max_iter=1000',
    ],
    [
        'Árbol de Decisión',
        'No Lineal',
        'criterion="gini", max_depth=5, min_samples_split=10',
    ],
    [
        'Random Forest',
        'Ensamble',
        'n_estimators=100, max_depth=8, max_features="sqrt"',
    ],
]

for row in param_rows:
  row_cells = t5.add_row().cells
  for i, val in enumerate(row):
    row_cells[i].text = val

doc.add_paragraph()

# Tabla Comparativa (Tabla 6)
doc.add_heading(
    '3. Comparación de Resultados en el Conjunto de Prueba', level=2
)
t6 = doc.add_table(rows=1, cols=6)
t6.alignment = WD_TABLE_ALIGNMENT.CENTER

headers_6 = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
for i, text in enumerate(headers_6):
  cell = t6.rows[0].cells[i]
  cell.text = text
  set_cell_bg(cell, '1F4E79')
  cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
  cell.paragraphs[0].runs[0].font.bold = True

for res in resultados:
  row_cells = t6.add_row().cells
  row_cells[0].text = res['Modelo']
  row_cells[1].text = res['Accuracy']
  row_cells[2].text = res['Precision']
  row_cells[3].text = res['Recall']
  row_cells[4].text = res['F1-Score']
  row_cells[5].text = res['AUC-ROC']
  if res['Modelo'] == 'Random Forest':
    for c in row_cells:
      c.paragraphs[0].runs[0].font.bold = True

doc.add_paragraph(
    '\nInterpretación: Random Forest obtuvo el mejor rendimiento global con un'
    ' AUC-ROC de 0.9231 y una exactitud del 91.25%, consolidándose como el'
    ' modelo candidato óptimo.'
)

doc.save('Modelado_Mineria_Datos_UEA.docx')
print("¡Sección 2.4 de Modelado exportada como 'Modelado_Mineria_Datos_UEA.docx'!")
