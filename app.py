import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistema de Evaluación de Riesgo Crediticio - UEA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Sistema de Evaluación de Riesgo Crediticio (WebAPP)")
st.caption("Desarrollado para la práctica de Minería de Datos - Universidad Estatal Amazónica (UEA)")
st.markdown("---")

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================
@st.cache_data
def cargar_y_preprocesar_datos():
    nombre_csv = 'datos_credito.csv'
    if os.path.exists(nombre_csv):
        df = pd.read_csv(nombre_csv)
    else:
        # Generación de dataset sintético de respaldo
        np.random.seed(42)
        n_samples = 1200
        edad = np.random.normal(38, 11, n_samples).clip(18, 72).astype(int)
        ingreso = np.random.exponential(scale=2000, size=n_samples) + 850
        monto_credito = np.random.exponential(scale=3500, size=n_samples) + 1000
        deuda_ingreso = np.random.uniform(0.05, 0.85, n_samples)
        uso_linea = np.random.uniform(0, 100, n_samples)
        historial = np.random.choice(['Bueno', 'Regular', 'Malo'], size=n_samples, p=[0.6, 0.3, 0.1])
        default = (uso_linea * 0.04 + deuda_ingreso * 3 + np.random.normal(0, 1, n_samples) > 3.2).astype(int)

        df = pd.DataFrame({
            'Edad': edad,
            'Ingreso_Mensual': ingreso,
            'Monto_Credito': monto_credito,
            'Relacion_Deuda_Ingreso': deuda_ingreso,
            'Uso_Linea_Credito': uso_linea,
            'Historial_Crediticio': historial,
            'Default': default
        })

    # Pipeline de Limpieza y Feature Engineering
    df_clean = df.drop_duplicates().copy()
    if 'Ingreso_Mensual' in df_clean.columns:
        df_clean['Ingreso_Mensual'].fillna(df_clean['Ingreso_Mensual'].median(), inplace=True)
    if 'Historial_Crediticio' in df_clean.columns:
        df_clean['Historial_Crediticio'].fillna(df_clean['Historial_Crediticio'].mode()[0], inplace=True)

    mapa_hist = {'Bueno': 0, 'Regular': 1, 'Malo': 2}
    df_clean['Historial_Crediticio_Cod'] = df_clean['Historial_Crediticio'].map(mapa_hist)

    df_clean['Capacidad_Pago'] = df_clean['Ingreso_Mensual'] * (1 - df_clean['Relacion_Deuda_Ingreso'])
    df_clean['Cuota_Estimada_Ingreso'] = (df_clean['Monto_Credito'] / 36) / df_clean['Ingreso_Mensual']
    df_clean['Indice_Riesgo_Compuesto'] = df_clean['Relacion_Deuda_Ingreso'] * (df_clean['Uso_Linea_Credito'] / 100)

    return df_clean

df_preprocesado = cargar_y_preprocesar_datos()

features = ['Edad', 'Ingreso_Mensual', 'Monto_Credito', 'Relacion_Deuda_Ingreso',
            'Uso_Linea_Credito', 'Historial_Crediticio_Cod', 'Capacidad_Pago',
            'Cuota_Estimada_Ingreso', 'Indice_Riesgo_Compuesto']

X = df_preprocesado[features]
y = df_preprocesado['Default']

# Divisón y Escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# SIDEBAR / MENÚ DE NAVEGACIÓN Y CONFIGURACIÓN
# ==========================================
st.sidebar.header("⚙️ Panel de Control")
opcion_menu = st.sidebar.radio(
    "Selecciona una Sección:",
    ["📊 Exploración de Datos", "🤖 Entrenamiento y Evaluación", "🔮 Simulación de Crédito"]
)

modelo_seleccionado = st.sidebar.selectbox(
    "Algoritmo de Aprendizaje:",
    ["Random Forest", "Regresión Logística", "Árbol de Decisión"]
)

# Inicializar Modelo
def obtener_modelo(nombre):
    if nombre == "Regresión Logística":
        return LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42), True
    elif nombre == "Árbol de Decisión":
        return DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42), False
    else:
        return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42), False

model, requiere_escalado = obtener_modelo(modelo_seleccionado)

# Entrenar Modelo Activo
if requiere_escalado:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# SECCIÓN 1: EXPLORACIÓN DE DATOS
# ==========================================
if opcion_menu == "📊 Exploración de Datos":
    st.subheader("📊 Exploración y Métricas Iniciales del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros Totales", f"{len(df_preprocesado):,}")
    col2.metric("Variables Totales", f"{df_preprocesado.shape[1]}")
    col3.metric("Clientes Cumplidos (0)", f"{(y == 0).sum()}")
    col4.metric("Clientes Morosos (1)", f"{(y == 1).sum()}")

    st.markdown("### Vista Previa de Datos Preprocesados")
    st.dataframe(df_preprocesado.head(10), use_container_width=True)

    st.markdown("### Distribución de la Variable Objetivo (Default)")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x='Default', data=df_preprocesado, palette=['#1f77b4', '#d62728'], ax=ax)
    ax.set_xticklabels(['Sin Mora (0)', 'En Mora (1)'])
    ax.set_ylabel("Cantidad de Clientes")
    st.pyplot(fig)

# ==========================================
# SECCIÓN 2: ENTRENAMIENTO Y EVALUACIÓN
# ==========================================
elif opcion_menu == "🤖 Entrenamiento y Evaluación":
    st.subheader(f"🤖 Resultados del Modelo: {modelo_seleccionado}")

    # Cálculo de Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc*100:.2f}%")
    m2.metric("Precision", f"{prec*100:.2f}%")
    m3.metric("Recall (Sensibilidad)", f"{rec*100:.2f}%")
    m4.metric("F1-Score", f"{f1*100:.2f}%")
    m5.metric("AUC-ROC", f"{auc:.4f}")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Realidad")
        ax_cm.set_xticklabels(['Cumple', 'Mora'])
        ax_cm.set_yticklabels(['Cumple', 'Mora'])
        st.pyplot(fig_cm)

    with col_g2:
        st.markdown("### Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {auc:.4f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_xlabel('Tasa de Falsos Positivos')
        ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_roc)

# ==========================================
# SECCIÓN 3: SIMULACIÓN DE CRÉDITO
# ==========================================
elif opcion_menu == "🔮 Simulación de Crédito":
    st.subheader("🔮 Evaluación de Nuevo Solicitante de Crédito")
    st.markdown("Ingrese las características del cliente para evaluar el riesgo crediticio en tiempo real:")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        input_edad = st.number_input("Edad del Solicitante:", min_value=18, max_value=80, value=35)
        input_ingreso = st.number_input("Ingreso Mensual ($):", min_value=300.0, max_value=20000.0, value=2500.0, step=100.0)
        input_monto = st.number_input("Monto de Crédito Solicitado ($):", min_value=500.0, max_value=50000.0, value=5000.0, step=500.0)

    with col_in2:
        input_deuda_ratio = st.slider("Relación Deuda/Ingreso:", min_value=0.01, max_value=0.95, value=0.35, step=0.01)
        input_uso_linea = st.slider("Uso de Línea de Crédito (%):", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        input_historial = st.selectbox("Historial Crediticio:", ["Bueno", "Regular", "Malo"])

    # Cálculo dinámico de variables derivadas
    mapa_h = {'Bueno': 0, 'Regular': 1, 'Malo': 2}
    hist_cod = mapa_h[input_historial]
    cap_pago = input_ingreso * (1 - input_deuda_ratio)
    cuota_ingreso = (input_monto / 36) / input_ingreso
    ind_riesgo = input_deuda_ratio * (input_uso_linea / 100)

    with col_in3:
        st.markdown("**Variables Sintéticas Calculadas:**")
        st.info(f"💡 **Capacidad de Pago:** ${cap_pago:,.2f}")
        st.info(f"💡 **Impacto Cuota/Ingreso:** {cuota_ingreso*100:.2f}%")
        st.info(f"💡 **Índice de Riesgo:** {ind_riesgo:.4f}")

    if st.button("🚀 Evaluar Solicitud de Crédito", use_container_width=True):
        nuevo_cliente = pd.DataFrame([{
            'Edad': input_edad,
            'Ingreso_Mensual': input_ingreso,
            'Monto_Credito': input_monto,
            'Relacion_Deuda_Ingreso': input_deuda_ratio,
            'Uso_Linea_Credito': input_uso_linea,
            'Historial_Crediticio_Cod': hist_cod,
            'Capacidad_Pago': cap_pago,
            'Cuota_Estimada_Ingreso': cuota_ingreso,
            'Indice_Riesgo_Compuesto': ind_riesgo
        }])

        if requiere_escalado:
            nuevo_cliente_eval = scaler.transform(nuevo_cliente)
        else:
            nuevo_cliente_eval = nuevo_cliente

        prediccion = model.predict(nuevo_cliente_eval)[0]
        probabilidad = model.predict_proba(nuevo_cliente_eval)[0][1]

        st.markdown("---")
        st.markdown("### Resultado de la Evaluación:")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediccion == 1:
                st.error("❌ **CRÉDITO RECHAZADO / ALTO RIESGO**")
                st.write("El modelo clasifica al cliente con alta probabilidad de caer en mora.")
            else:
                st.success("✅ **CRÉDITO APROBADO / BAJO RIESGO**")
                st.write("El modelo clasifica al cliente como sujeto de crédito apto.")

        with res_col2:
            st.metric("Probabilidad Estimada de Mora", f"{probabilidad*100:.2f}%")
            st.progress(float(probabilidad))
