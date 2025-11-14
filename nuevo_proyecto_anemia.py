import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from supabase import create_client, Client
from datetime import datetime

# --- CONFIGURACIÓN DE SUPABASE (Necesitas tus propias credenciales) ---
# Se recomienda usar los secretos de Streamlit para producción.
# Para este ejemplo, se asumen variables de entorno o valores de prueba.
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
DATABASE_TABLE = "AlertasAnemia" # Asegúrate de que esta sea tu tabla en Supabase

# Inicializar cliente Supabase
@st.cache_resource
def init_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.error(f"Error al inicializar Supabase: {e}")
            return None
    else:
        st.warning("Credenciales de Supabase no configuradas. El registro de alertas estará desactivado.")
        return None

supabase: Client = init_supabase_client()

# --- SIMULACIÓN DE CARGA DE MODELOS (Reemplaza con tus archivos reales) ---
# Asumimos que los archivos 'modelo_anemia.joblib' y 'modelo_columns.joblib' existen.
@st.cache_resource
def load_assets():
    try:
        # Cargar el modelo ML (simulación)
        # Reemplaza 'modelo_anemia.joblib' con la ruta a tu modelo entrenado
        # model = joblib.load('modelo_anemia.joblib')
        
        # Simulación de un clasificador binario (sustituye esto)
        class MockModel:
            def predict(self, X):
                # Predice 1 (Alto Riesgo) si Hemoglobina < 11.0 o Edad < 12 meses, sino 0
                return np.where((X[:, 0] < 11.0) | (X[:, 1] < 12), 1, 0)

        model = MockModel()

        # Cargar las columnas usadas en el entrenamiento (simulación)
        # Reemplaza 'modelo_columns.joblib' con la ruta a tu lista de columnas
        # feature_columns = joblib.load('modelo_columns.joblib')
        feature_columns = ['Hemoglobina', 'Edad_Meses', 'Peso_Hogar', 'Ingreso_Familiar_Soles']
        
        return model, feature_columns
    except FileNotFoundError:
        st.error("Error: Archivos del modelo (joblib) no encontrados. Asegúrate de que estén en la ruta correcta.")
        return None, None

model, feature_columns = load_assets()

# --- LÓGICA DE NEGOCIO Y CLÍNICA ---

def clasificar_anemia(hemoglobina, edad_meses, altitud=1500):
    """Clasifica el riesgo de anemia basado en la hemoglobina corregida por altitud."""
    
    # Factor de Corrección por Altitud (Ejemplo simplificado)
    # A > 1000 msnm se aplica corrección. (Ej: 0.8 a 1.5 g/dL)
    if altitud > 1000:
        correccion = 0.8 # Valor de ejemplo
        hemoglobina_corregida = hemoglobina + correccion
    else:
        hemoglobina_corregida = hemoglobina

    # Puntos de corte de diagnóstico (simplificado)
    if edad_meses >= 6 and edad_meses <= 59:
        if hemoglobina_corregida < 11.0:
            return "ANEMIA", hemoglobina_corregida
        elif hemoglobina_corregida >= 11.0 and hemoglobina_corregida < 13.0:
            return "RIESGO LEVE", hemoglobina_corregida
        else:
            return "NORMAL", hemoglobina_corregida
    elif edad_meses < 6:
        # Para menores de 6 meses, los puntos de corte son diferentes (simplificación)
        if hemoglobina_corregida < 10.0:
            return "ANEMIA SEVERA", hemoglobina_corregida
        else:
            return "NORMAL", hemoglobina_corregida
    
    return "NO APLICABLE", hemoglobina_corregida

def clasificar_clima(altitud):
    """Simula la clasificación climática basada en la altitud."""
    if altitud >= 2500:
        return "FRÍO/ALTOANDINO"
    elif altitud >= 1000:
        return "TEMPLADO"
    else:
        return "CÁLIDO/SECO"

# --- REGISTRO DE ALERTA EN SUPABASE ---

def registrar_alerta(data):
    """Guarda los datos del informe y la alerta en Supabase."""
    if supabase is None:
        st.error("No se pudo registrar la alerta: Supabase no está inicializado.")
        return False, "Supabase no inicializado."
    
    try:
        # Los datos son un diccionario. Supabase acepta diccionarios directamente.
        data['created_at'] = datetime.now().isoformat()
        
        response = supabase.table(DATABASE_TABLE).insert(data).execute()
        
        if response.data:
            return True, f"Registro exitoso con ID: {response.data[0]['id']}"
        else:
            # Manejar posibles errores devueltos por la API de Supabase
            st.error(f"Respuesta de Supabase sin datos: {response}")
            return False, "Error desconocido en Supabase."

    except Exception as e:
        st.error(f"Error al registrar la alerta en Supabase: {e}")
        return False, str(e)


# --- INTERFAZ DE USUARIO CON STREAMLIT ---

def app():
    st.set_page_config(
        page_title="Alerta Anemia IA",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título de la Aplicación y Sidebar
    st.sidebar.title("Sistema de Alerta IA")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Estado del Sistema**")
    st.sidebar.success("✅ Modelo ML Cargado")
    
    if supabase:
        st.sidebar.success("✅ DB Contacto (Supabase Activa)")
    else:
        st.sidebar.error("❌ DB Contacto (Supabase Inactiva)")

    st.sidebar.markdown("---")
    st.sidebar.info("⚠️ Los datos NO PERSISTEN al recargar. Se usa una simulación.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Módulos de Control")
    st.sidebar.radio("Navegación", ["Predicción y Reporte", "Monitoreo de Alertas"], index=0)

    # Título Principal del Informe
    st.title("Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.5 Altitud y Clima Automatizados)")

    # Formulario Principal
    with st.form(key='anemia_form'):
        
        # --- 0. Datos de Identificación y Contacto ---
        st.header("0. Datos de Identificación y Contacto")
        col_id, col_nombre, col_contacto = st.columns(3)
        with col_id:
            dni = st.text_input("DNI del Paciente", value="01234567", max_chars=8, help="Solo 8 dígitos")
        with col_nombre:
            nombre = st.text_input("Nombre y Apellido", value="Eva Torres")
        with col_contacto:
            contacto = st.text_input("Celular de Contacto", value="+51 900000000", max_chars=12)

        st.markdown("---")

        # --- 1. Factores Clínicos y Demográficos Clave ---
        st.header("1. Factores Clínicos y Demográficos Clave")
        col_hb, col_edad, col_region = st.columns(3)
        
        with col_hb:
            hemoglobina = st.number_input(
                "Hemoglobina (g/dL) - Criterio Clínico", 
                min_value=0.0, 
                max_value=20.0, 
                value=10.50, 
                step=0.1, 
                format="%.2f"
            )
        
        with col_edad:
            edad_meses = st.slider(
                "Edad (meses)", 
                min_value=0, 
                max_value=60, 
                value=36, 
                step=1
            )
        
        with col_region:
            region = st.selectbox(
                "Región (Define Altitud y Clima)", 
                ["LIMA (Metropolitana y Provincia)", "JUNIN (Sierra Central)"],
                index=0
            )
            # Simulación de Altitud y Clima
            altitud = 150 # Altitud de Lima (ejemplo)
            if "JUNIN" in region:
                altitud = 3200 # Altitud de Junín (ejemplo)

            st.info(f"Altitud asignada automáticamente: {altitud} msnm (Usada para corrección de Hemoglobina).")

        st.markdown("---")

        # --- 2. Factores Socioeconómicos y Contextuales ---
        st.header("2. Factores Socioeconómicos y Contextuales")
        col_peso, col_ingreso, col_residencia, col_sexo = st.columns(4)

        with col_peso:
            peso_hogar = st.selectbox("Nro. de hijos en el Hogar", [1, 2, 3, 4, 5], index=1)
        
        with col_ingreso:
            ingreso_familiar_soles = st.number_input("Ingreso Familiar (Soles/mes)", min_value=0, value=1500, step=100)
            
        with col_residencia:
            area_residencia = st.selectbox("Área de Residencia", ["Urbana", "Rural"])
            
        with col_sexo:
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
            
        clima_auto = clasificar_clima(altitud)
        st.info(f"Clima asignado automáticamente para {region}: **{clima_auto}**.")
        
        st.markdown("---")

        # --- 3. Acceso a Programas y Servicios (Simulación de Binarias) ---
        st.header("3. Acceso a Programas y Servicios")
        col_c_1, col_c_2, col_c_3, col_c_4 = st.columns(4)
        
        programa_wawa = col_c_1.radio("Programa Cuna Más", ["Sí", "No"], index=1)
        programa_juntos = col_c_2.radio("Programa Juntos", ["Sí", "No"], index=1)
        programa_leche = col_c_3.radio("Programa Vaso de Leche", ["Sí", "No"], index=1)
        recibe_suplemento = col_c_4.radio("Recibe Suplemento de Hierro", ["Sí", "No"], index=1)


        # Botón de Envío
        st.markdown("---")
        submit_button = st.form_submit_button(
            label='GENERAR INFORME PERSONALIZADO, REGISTRAR CASO Y ENVIAR ALERTA',
            help="Alerta y registra los datos para seguimiento.",
            type="primary"
        )


    if submit_button:
        if model is None:
            st.error("No se puede generar el informe. El modelo ML no pudo ser cargado.")
            return

        # --- 1. Lógica Clínica / Diagnóstico ---
        diagnostico_clinico, hb_corregida = clasificar_anemia(hemoglobina, edad_meses, altitud)
        
        # --- 2. Preparación de datos para ML ---
        # Asegúrate de que el orden de las columnas coincida con el modelo (feature_columns)
        input_data = {
            'Hemoglobina': hb_corregida,
            'Edad_Meses': edad_meses,
            'Peso_Hogar': peso_hogar,
            'Ingreso_Familiar_Soles': ingreso_familiar_soles
            # Añade otras variables del formulario aquí si tu modelo las usa
        }
        
        # Crear DataFrame y alinear columnas
        df_input = pd.DataFrame([input_data])
        # Solo usar las columnas que el modelo espera
        X_pred = df_input[feature_columns].values
        
        # --- 3. Predicción de Riesgo ML ---
        try:
            prediccion_ml = model.predict(X_pred)[0]
            
            # --- 4. Presentar Resultados ---
            st.markdown("## ✅ Resultados del Diagnóstico y Predicción de Riesgo")
            
            # Resultado Clínico
            if diagnostico_clinico == "ANEMIA" or diagnostico_clinico == "ANEMIA SEVERA":
                st.error(f"**DIAGNÓSTICO CLÍNICO:** {diagnostico_clinico} (Hemoglobina Corregida: {hb_corregida:.2f} g/dL)")
                alerta_status = "ALTO"
                alerta_color = "red"
            elif diagnostico_clinico == "RIESGO LEVE":
                st.warning(f"**DIAGNÓSTICO CLÍNICO:** {diagnostico_clinico} (Hemoglobina Corregida: {hb_corregida:.2f} g/dL)")
                alerta_status = "MEDIO"
                alerta_color = "orange"
            else:
                st.success(f"**DIAGNÓSTICO CLÍNICO:** {diagnostico_clinico} (Hemoglobina Corregida: {hb_corregida:.2f} g/dL)")
                alerta_status = "BAJO"
                alerta_color = "green"
            
            # Resultado ML
            if prediccion_ml == 1:
                st.error(f"**PREDICCIÓN ML:** ALTO RIESGO de desarrollar/mantener ANEMIA en los próximos meses.")
            else:
                st.success(f"**PREDICCIÓN ML:** BAJO RIESGO de desarrollar/mantener ANEMIA en los próximos meses.")

            # --- 5. Preparar y Registrar Alerta ---
            alerta_data = {
                "dni": dni,
                "nombre": nombre,
                "contacto": contacto,
                "hemoglobina_original": hemoglobina,
                "hemoglobina_corregida": float(f"{hb_corregida:.2f}"),
                "edad_meses": edad_meses,
                "diagnostico_clinico": diagnostico_clinico,
                "prediccion_ml_riesgo": "ALTO" if prediccion_ml == 1 else "BAJO",
                "altitud_msnm": altitud,
                "clima_auto": clima_auto,
                "area_residencia": area_residencia,
                "ingreso_familiar_soles": ingreso_familiar_soles,
                "peso_hogar": peso_hogar,
                "alerta_status": alerta_status,
                "datos_completos": json.dumps(input_data) # Guardar input como JSON string
            }

            st.markdown("---")
            with st.spinner("Registrando alerta en la base de datos Supabase..."):
                registro_exitoso, mensaje = registrar_alerta(alerta_data)

            if registro_exitoso:
                st.balloons()
                st.success(f"¡Alerta Registrada! {mensaje}. El caso de **{nombre}** ha sido notificado para seguimiento.")
                st.json(alerta_data)
            else:
                st.error(f"Fallo en el registro de la alerta. Mensaje: {mensaje}")
                
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
            st.code(f"Asegúrate de que tus datos de entrada coincidan con las columnas del modelo: {feature_columns}")


if __name__ == "__main__":
    app()
