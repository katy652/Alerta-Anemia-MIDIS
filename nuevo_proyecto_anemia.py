import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os

# Si deseas conectar el Dashboard a Supabase (como se veía en las imágenes anteriores),
# necesitas descomentar las siguientes líneas.
# from supabase import create_client, Client 

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Anemia Infantil",
    page_icon="👶",
    layout="wide"
)

# Título principal
st.title("🩸 Sistema de Predicción Temprana de Anemia en Menores de 5 Años")
st.markdown("---")

# Función mejorada para cargar datos - CORREGIDA
@st.cache_data
def load_data():
    """Carga datos con manejo robusto de errores - VERSIÓN CORREGIDA"""
    
    # PRIMERO: Mostrar que la aplicación está funcionando
    st.sidebar.success("✅ Aplicación cargada correctamente")
    
    # Intentar cargar archivo real
    file_options = [
        "dataset_midis_anemia_20000_realistic.xlsx - Sheet1.csv",
        "datos_anemia.csv",
        "dataset_anemia.csv"
    ]
    
    for file_path in file_options:
        if os.path.exists(file_path):
            st.sidebar.info(f"📁 Archivo encontrado: {file_path}")
            
            # Probar diferentes codificaciones
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'windows-1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    st.sidebar.success(f"✅ Datos cargados con {encoding}")
                    st.sidebar.info(f"📊 {df.shape[0]} registros, {df.shape[1]} variables")
                    
                    # VERIFICAR SI HAY COLUMNA REGION Y NO ESTÁ VACÍA
                    if 'Region' in df.columns and not df['Region'].isna().all():
                        return df
                    else:
                        st.sidebar.warning("❌ Columna 'Region' no encontrada o vacía en el CSV")
                        break
                        
                except Exception as e:
                    continue
            
            st.sidebar.error("❌ No se pudo leer el archivo con ninguna codificación")
            break
    
    st.sidebar.warning("📁 Archivo CSV no encontrado o no se pudo leer")
    st.sidebar.info("💡 Usando datos de ejemplo para demostración")
    
    # CREAR DATOS DE EJEMPLO MEJORADOS - CORREGIDO
    np.random.seed(42)
    n_samples = 1500
    
    # Lista de regiones del Perú - CORREGIDO
    regiones_peru = [
        'Lima', 'Arequipa', 'Cusco', 'Piura', 'Loreto', 
        'La Libertad', 'Junín', 'Cajamarca', 'Puno', 'Áncash',
        'Lambayeque', 'Ica', 'Huánuco', 'San Martín', 'Tacna',
        'Amazonas', 'Moquegua', 'Pasco', 'Madre de Dios', 'Tumbes',
        'Huancavelica', 'Ucayali', 'Ayacucho', 'Callao'
    ]
    
    # Crear DataFrame con datos realistas
    df = pd.DataFrame({
        'Region': np.random.choice(regiones_peru, n_samples),
        'Edad_meses': np.random.randint(0, 60, n_samples),
        'Sexo': np.random.choice(['M', 'F'], n_samples),
        'Peso_kg': np.round(np.random.uniform(3.0, 15.0, n_samples), 1),
        'Talla_cm': np.round(np.random.uniform(50.0, 100.0, n_samples), 1),
        'Hemoglobina_g_dL': np.round(np.random.uniform(8.0, 14.0, n_samples), 1),
        'Altitud_m': np.random.randint(0, 4000, n_samples),
        'Ingreso_Familiar_Soles': np.random.randint(300, 2000, n_samples),
        'Suplemento_Hierro': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'Programa_QaliWarma': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Programa_Juntos': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Programa_VasoLeche': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'Acceso_Agua': np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    })
    
    # VERIFICAR QUE NO HAYA NaN EN REGION
    st.sidebar.info(f"🔍 Regiones únicas creadas: {df['Region'].nunique()}")
    
    # Crear variable de anemia basada en reglas realistas
    def calcular_anemia(row):
        # Criterios médicos reales para anemia
        if row['Hemoglobina_g_dL'] < 11.0:
            return 1
        elif row['Hemoglobina_g_dL'] < 11.5 and row['Altitud_m'] > 2500:
            return 1
        elif (row['Ingreso_Familiar_Soles'] < 500 and 
              row['Suplemento_Hierro'] == 0 and 
              row['Acceso_Agua'] == 0):
            return 1
        else:
            return 0
    
    df['Anemia'] = df.apply(calcular_anemia, axis=1)
    
    # Crear variable de riesgo
    def calcular_riesgo(row):
        if row['Hemoglobina_g_dL'] < 10.0:
            return 'Alto'
        elif row['Hemoglobina_g_dL'] < 11.0:
            return 'Medio'
        elif (row['Altitud_m'] > 2500 and 
              row['Suplemento_Hierro'] == 0 and 
              row['Ingreso_Familiar_Soles'] < 500):
            return 'Medio'
        else:
            return 'Bajo'
    
    df['Riesgo_Anemia'] = df.apply(calcular_riesgo, axis=1)
    
    st.sidebar.success(f"🎯 Datos de ejemplo creados: {n_samples} registros")
    st.sidebar.success(f"📍 Regiones: {df['Region'].nunique()} diferentes")
    
    return df

# Cargar datos
df = load_data()

# VERIFICACIÓN EXTRA - Mostrar información de las regiones
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Verificación de Datos")
st.sidebar.write(f"Regiones únicas: {df['Region'].nunique()}")
st.sidebar.write(f"Primeras regiones: {list(df['Region'].unique()[:5])}")

# Sidebar para navegación
st.sidebar.markdown("---")
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox(
    "Selecciona el módulo:",
    ["📊 Análisis Exploratorio", "🎯 Predictor Individual", "📈 Modelo ML", "ℹ️ Acerca del Proyecto"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Consejo**: 
- Comienza con el análisis exploratorio
- Usa el predictor individual para casos específicos
- Entrena modelos en la sección ML
""")

# Preprocesamiento de datos MEJORADO
def preprocess_data(df):
    """Limpieza básica de datos - MEJORADA"""
    df_clean = df.copy()
    
    # Asegurar que Region no tenga NaN
    if 'Region' in df_clean.columns:
        df_clean['Region'] = df_clean['Region'].fillna('Desconocido')
        # Eliminar filas donde Region sea string vacío
        df_clean = df_clean[df_clean['Region'].astype(str).str.strip() != '']
    
    # Asegurar tipos numéricos
    numeric_cols = ['Peso_kg', 'Talla_cm', 'Altitud_m', 'Ingreso_Familiar_Soles', 'Hemoglobina_g_dL']
    for col in numeric_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
    
    return df_clean

df_clean = preprocess_data(df)

# =============================================================================
# MÓDULO: ANÁLISIS EXPLORATORIO - CORREGIDO
# =============================================================================

if app_mode == "📊 Análisis Exploratorio":
    st.header("📊 Análisis Exploratorio de Datos")
    
    # VERIFICACIÓN DE REGIONES EN TIEMPO REAL
    with st.expander("🔍 Verificación de Calidad de Datos", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Registros", df_clean.shape[0])
        with col2:
            st.metric("Regiones Únicas", df_clean['Region'].nunique())
        with col3:
            st.metric("Registros sin Region", df_clean['Region'].isna().sum())
        
        # Mostrar algunas regiones
        st.write("**Muestra de regiones:**", list(df_clean['Region'].unique()[:10]))
    
    # Mostrar datos
    with st.expander("🔍 Ver datos de muestra", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df_clean.head(10), use_container_width=True)
        with col2:
            st.metric("Total Variables", df_clean.shape[1])
    
    # Estadísticas rápidas
    st.subheader("📈 Métricas Clave")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        anemia_rate = df_clean['Anemia'].mean() * 100
        st.metric("Prevalencia de Anemia", f"{anemia_rate:.1f}%")
    
    with col2:
        avg_age = df_clean['Edad_meses'].mean()
        st.metric("Edad Promedio", f"{avg_age:.1f} meses")
    
    with col3:
        avg_hemo = df_clean['Hemoglobina_g_dL'].mean()
        st.metric("Hemoglobina Promedio", f"{avg_hemo:.1f} g/dL")
    
    with col4:
        avg_alt = df_clean['Altitud_m'].mean()
        st.metric("Altitud Promedio", f"{avg_alt:.0f} msnm")
    
    st.markdown("---")
    
    # Visualizaciones - CORREGIDAS
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de Anemia por Región - CORREGIDO
        st.subheader("📍 Anemia por Región")
        
        # Verificar que hay datos de región
        if 'Region' in df_clean.columns and not df_clean['Region'].isna().all():
            anemia_region = df_clean.groupby('Region')['Anemia'].mean().sort_values(ascending=False)
            
            # Tomar solo las top 10 regiones para mejor visualización
            if len(anemia_region) > 10:
                anemia_region = anemia_region.head(10)
            
            fig = px.bar(
                anemia_region, 
                title="Prevalencia de Anemia por Región (Top 10)",
                labels={'value': 'Prevalencia (%)', 'Region': 'Región'},
                color=anemia_region.values,
                color_continuous_scale='reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No hay datos de región disponibles para mostrar")
        
        # Hemoglobina vs Edad
        st.subheader("📈 Hemoglobina vs Edad")
        fig = px.scatter(
            df_clean, 
            x='Edad_meses', 
            y='Hemoglobina_g_dL',
            color='Anemia',
            title="Relación entre Edad y Nivel de Hemoglobina",
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución de Riesgo
        st.subheader("🎯 Distribución de Riesgo")
        if 'Riesgo_Anemia' in df_clean.columns:
            riesgo_counts = df_clean['Riesgo_Anemia'].value_counts()
            fig = px.pie(
                riesgo_counts, 
                values=riesgo_counts.values, 
                names=riesgo_counts.index,
                title="Distribución del Riesgo de Anemia"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No hay datos de riesgo disponibles")
        
        # Impacto de programas sociales
        st.subheader("🏛️ Impacto de Programas Sociales")
        if 'Programa_QaliWarma' in df_clean.columns and 'Anemia' in df_clean.columns:
            programas_data = []
            
            qali_effect = df_clean.groupby('Programa_QaliWarma')['Anemia'].mean()
            programas_data.append(('Qali Warma', qali_effect))
            
            if 'Programa_Juntos' in df_clean.columns:
                juntos_effect = df_clean.groupby('Programa_Juntos')['Anemia'].mean()
                programas_data.append(('Juntos', juntos_effect))
            
            programas_df = pd.DataFrame({
                name: effect for name, effect in programas_data
            })
            programas_df.index = ['No Participa', 'Participa']
            
            fig = px.bar(
                programas_df, 
                barmode='group',
                title="Prevalencia de Anemia por Participación en Programas",
                labels={'value': 'Prevalencia de Anemia', 'variable': 'Programa'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Datos de programas sociales no disponibles")
    
    # Mapa de correlaciones
    st.subheader("🔗 Mapa de Correlaciones")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        corr_matrix = df_clean[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        st.pyplot(fig)
    else:
        st.info("ℹ️ No hay suficientes columnas numéricas para calcular correlaciones")

# =============================================================================
# MÓDULO: PREDICTOR INDIVIDUAL - CORREGIDO
# =============================================================================

elif app_mode == "🎯 Predictor Individual":
    st.header("🎯 Predictor de Riesgo Individual")
    st.info("Ingresa los datos del menor para predecir el riesgo de anemia")
    
    # VERIFICAR QUE HAY REGIONES DISPONIBLES
    if 'Region' not in df_clean.columns or df_clean['Region'].isna().all():
        st.error("❌ No hay datos de región disponibles. Por favor, usa el módulo de Análisis Exploratorio primero.")
    else:
        with st.form("prediccion_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("👶 Datos Personales")
                edad = st.slider("Edad (meses)", 0, 60, 24)
                sexo = st.selectbox("Sexo", ["Masculino", "Femenino"])
                region_options = sorted(df_clean['Region'].unique())
                region = st.selectbox("Región", region_options)
            
            with col2:
                st.subheader("⚖️ Datos Antropométricos")
                peso = st.number_input("Peso (kg)", min_value=2.0, max_value=25.0, value=12.0, step=0.1)
                talla = st.number_input("Talla (cm)", min_value=50.0, max_value=120.0, value=85.0, step=0.1)
                hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
            
            with col3:
                st.subheader("🏠 Contexto Socioeconómico")
                altitud = st.number_input("Altitud (msnm)", min_value=0, max_value=5000, value=1500)
                ingreso = st.number_input("Ingreso Familiar (S/.)", min_value=0, value=1000, step=50)
                acceso_agua = st.checkbox("Acceso a agua potable", value=True)
                suplemento_hierro = st.checkbox("Recibe suplemento de hierro", value=True)
            
            st.subheader("📋 Programas Sociales")
            col_prog1, col_prog2, col_prog3 = st.columns(3)
            with col_prog1:
                programa_qw = st.checkbox("Programa Qali Warma", value=True)
            with col_prog2:
                programa_juntos = st.checkbox("Programa Juntos")
            with col_prog3:
                programa_vaso = st.checkbox("Programa Vaso Leche")
            
            submitted = st.form_submit_button("🔍 Predecir Riesgo")
            
            if submitted:
                # Lógica de predicción basada en criterios médicos
                st.success("✅ Predicción completada!")
                
                # Criterios médicos para determinar riesgo
                if hemoglobina < 10.0:
                    riesgo = "ALTO"
                    probabilidad = 0.85
                    recomendacion = "🚨 Suplementación urgente y evaluación médica inmediata"
                    color_riesgo = "red"
                elif hemoglobina < 11.0:
                    riesgo = "MEDIO"
                    probabilidad = 0.65
                    recomendacion = "⚠️ Seguimiento cercano y suplementación preventiva"
                    color_riesgo = "orange"
                else:
                    if altitud > 2500 and ingreso < 500 and not suplemento_hierro:
                        riesgo = "MEDIO"
                        probabilidad = 0.45
                        recomendacion = "⚠️ Factores de riesgo presentes, monitoreo recomendado"
                        color_riesgo = "orange"
                    else:
                        riesgo = "BAJO"
                        probabilidad = 0.15
                        recomendacion = "✅ Continuar con prevención y controles regulares"
                        color_riesgo = "green"
                
                # Mostrar resultados
                st.markdown("---")
                st.subheader("📊 Resultados de la Predicción")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric(
                        "Riesgo de Anemia", 
                        riesgo, 
                        delta=f"{probabilidad*100:.0f}% probabilidad", 
                        delta_color="inverse"
                    )
                with col_res2:
                    st.metric("Nivel de Hemoglobina", f"{hemoglobina} g/dL")
                with col_res3:
                    st.metric("Recomendación", recomendacion.split(' ')[0])
                
                # Barra de probabilidad visual
                st.subheader("📈 Probabilidad de Riesgo")
                prob_percent = probabilidad * 100
                
                if riesgo == "ALTO":
                    color = "red"
                    emoji = "🚨"
                elif riesgo == "MEDIO":
                    color = "orange" 
                    emoji = "⚠️"
                else:
                    color = "green"
                    emoji = "✅"
                
                st.markdown(f"""
                <div style="background-color: #f0f0f0; border-radius: 10px; padding: 5px; margin: 10px 0;">
                    <div style="background-color: {color}; width: {prob_percent}%; border-radius: 8px; padding: 10px; color: white; text-align: center; font-weight: bold;">
                        {emoji} {prob_percent:.0f}% Probabilidad de Anemia
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Factores de riesgo identificados
                st.subheader("🔍 Factores de Riesgo Identificados")
                factores = []
                
                if hemoglobina < 11.0:
                    factores.append(f"Bajo nivel de hemoglobina ({hemoglobina} g/dL)")
                if altitud > 2500:
                    factores.append(f"Vive en altitud elevada ({altitud} msnm)")
                if ingreso < 500:
                    factores.append(f"Bajo ingreso familiar (S/. {ingreso})")
                if not suplemento_hierro:
                    factores.append("No recibe suplemento de hierro")
                if not acceso_agua:
                    factores.append("Sin acceso a agua potable")
                
                if factores:
                    st.warning("**Se identificaron los siguientes factores de riesgo:**")
                    for factor in factores:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ No se identificaron factores de riesgo significativos")
                
                # Recomendaciones específicas
                st.subheader("💡 Recomendaciones Específicas")
                st.info(f"**{recomendacion}**")
                
                if riesgo == "ALTO":
                    st.error("""
                    **Acciones inmediatas recomendadas:**
                    - Consulta médica urgente
                    - Suplementación con hierro bajo supervisión
                    - Control nutricional intensivo
                    - Seguimiento semanal
                    """)
                elif riesgo == "MEDIO":
                    st.warning("""
                    **Acciones preventivas recomendadas:**
                    - Control médico en 15 días
                    - Suplementación preventiva con hierro
                    - Educación nutricional familiar
                    - Seguimiento mensual
                    """)
                else:
                    st.success("""
                    **Acciones de mantenimiento:**
                    - Continuar controles regulares
                    - Mantener alimentación balanceada
                    - Seguir con suplementación si está indicada
                    - Próximo control en 3 meses
                    """)

# Los otros módulos (Modelo ML y Acerca del Proyecto) se mantienen igual...

elif app_mode == "📈 Modelo ML":
    st.header("📈 Modelo de Machine Learning")
    st.info("Entrena y evalúa modelos predictivos para riesgo de anemia")
    
    # Preparar datos para el modelo
    st.subheader("1. Preparación de Datos")
    
    # Seleccionar características disponibles
    available_features = []
    feature_mapping = {
        'Edad_meses': 'Edad_meses',
        'Peso_kg': 'Peso_kg', 
        'Talla_cm': 'Talla_cm',
        'Altitud_m': 'Altitud_m',
        'Ingreso_Familiar_Soles': 'Ingreso_Familiar_Soles',
        'Suplemento_Hierro': 'Suplemento_Hierro',
        'Acceso_Agua': 'Acceso_Agua',
        'Programa_QaliWarma': 'Programa_QaliWarma',
        'Programa_Juntos': 'Programa_Juntos'
    }
    
    for feature, col_name in feature_mapping.items():
        if col_name in df_clean.columns:
            available_features.append(feature)
    
    selected_features = st.multiselect(
        "Selecciona las características para el modelo:",
        available_features,
        default=available_features[:4] if len(available_features) >= 4 else available_features
    )
    
    # Seleccionar variable objetivo
    target_variable = st.selectbox("Variable objetivo:", ['Anemia', 'Riesgo_Anemia'])
    
    if st.button("🎯 Entrenar Modelo Random Forest"):
        if len(selected_features) < 2:
            st.error("Selecciona al menos 2 características")
        else:
            with st.spinner("Entrenando modelo Random Forest..."):
                try:
                    # Preparar datos
                    feature_columns = [feature_mapping[feat] for feat in selected_features]
                    X = df_clean[feature_columns].copy()
                    y = df_clean[target_variable].copy()
                    
                    # Limpiar datos nulos
                    valid_indices = X.notna().all(axis=1) & y.notna()
                    X = X[valid_indices]
                    y = y[valid_indices]
                    
                    # Codificar variables categóricas si es necesario
                    if target_variable == 'Riesgo_Anemia' and y.dtype == 'object':
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        class_names = le.classes_
                    else:
                        y_encoded = y
                        class_names = ['No Anemia', 'Anemia']
                    
                    # Dividir datos
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )
                    
                    # Entrenar modelo
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Predecir
                    y_pred = model.predict(X_test)
                    
                    # Calcular métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.success("✅ Modelo entrenado exitosamente!")
                    
                    # Mostrar resultados
                    col_met1, col_met2, col_met3 = st.columns(3)
                    with col_met1:
                        st.metric("Precisión", f"{accuracy*100:.1f}%")
                    with col_met2:
                        st.metric("Características", len(selected_features))
                    with col_met3:
                        st.metric("Registros de entrenamiento", len(X_train))
                    
                    # Importancia de características
                    st.subheader("📊 Importancia de Características")
                    feature_importance = pd.DataFrame({
                        'caracteristica': selected_features,
                        'importancia': model.feature_importances_
                    }).sort_values('importancia', ascending=True)
                    
                    fig = px.bar(
                        feature_importance, 
                        x='importancia', 
                        y='caracteristica',
                        title='Importancia de Características en el Modelo',
                        orientation='h'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Matriz de confusión
                    st.subheader("🎯 Matriz de Confusión")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues', 
                        ax=ax,
                        xticklabels=class_names,
                        yticklabels=class_names
                    )
                    ax.set_xlabel('Predicción')
                    ax.set_ylabel('Real')
                    ax.set_title('Matriz de Confusión')
                    st.pyplot(fig)
                    
                    # Reporte de clasificación
                    st.subheader("📋 Reporte de Clasificación")
                    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.2f}"))
                    
                except Exception as e:
                    st.error(f"Error entrenando el modelo: {e}")

else:
    st.header("ℹ️ Acerca del Proyecto")
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown("""
        ### 🎯 Objetivo del Proyecto
        
        Desarrollar un sistema integrado de **visualización de datos** e **inteligencia artificial** para la **predicción temprana de anemia** en menores de 5 años en el Perú.
        
        ### 📊 Características Principales
        
        - **🔍 Análisis Exploratorio**: Visualización interactiva de datos sociodemográficos y clínicos
        - **🎯 Predictor Individual**: Evaluación de riesgo personalizado por niño
        - **🤖 Modelo de ML**: Algoritmos predictivos entrenados con datos realistas
        - **📈 Dashboard Interactivo**: Interfaz amigable para gestores públicos
        
        ### 🏥 Variables Clave Analizadas
        
        - **Sociodemográficas**: Edad, sexo, región, altitud
        - **Antropométricas**: Peso, talla, hemoglobina
        - **Socioeconómicas**: Ingreso familiar, acceso a servicios
        - **Programas Sociales**: Qali Warma, Juntos, suplemento de hierro
        """)
    
    with col_info2:
        st.markdown("""
        ### 🛠 Tecnologías
        
        - **Python** + Streamlit
        - **Scikit-learn** (ML)
        - **Pandas** + NumPy
        - **Plotly** + Matplotlib
        
        ### 👥 Beneficiarios
        
        - Ministerio de Desarrollo e Inclusión Social
        - Programas Qali Warma y Juntos
        - Gestores de salud pública
        - Organizaciones sociales
        """)
    
    st.markdown("---")
    
    st.success("""
    **💡 Nota**: Esta aplicación utiliza datos del programa MIDIS para el análisis y predicción 
    de anemia infantil. Los modelos están entrenados con información sociodemográfica, clínica y de 
    programas sociales.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sistema desarrollado para la predicción temprana de anemia infantil - Perú 2024 | "
    "👶 Protegiendo el futuro de nuestros niños"
    "</div>", 
    unsafe_allow_html=True
)
