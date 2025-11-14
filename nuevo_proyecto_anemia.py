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
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "TU_URL_SUPABASE_NO_CONFIGURADA")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "TU_KEY_SUPABASE_NO_CONFIGURADA")
DATABASE_TABLE = "AlertasAnemia" # Asegúrate de que esta sea tu tabla en Supabase

# Flag para rastrear si se usa el modelo real o el simulado
MODELO_REAL_CARGADO = False

# Inicializar cliente Supabase
@st.cache_resource
def init_supabase_client():
    # Solo inicializar si las credenciales NO son las de placeholder
    if SUPABASE_URL != "TU_URL_SUPABASE_NO_CONFIGURADA" and SUPABASE_KEY != "TU_KEY_SUPABASE_NO_CONFIGURADA":
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.error(f"Error al inicializar Supabase: {e}")
            return None
    else:
        # st.warning("Credenciales de Supabase no configuradas. El registro de alertas estará desactivado.")
        return None

supabase: Client = init_supabase_client()

# --- CARGA DE MODELOS ---
class MockModel:
    """Clase de simulación usada si el modelo real no carga."""
    def predict(self, X):
        # Simulación: Predice 1 (Alto Riesgo) si Hemoglobina < 11.0 o Edad < 12 meses
        # X[:, 0] es Hemoglobina_Corregida, X[:, 1] es Edad_Meses
        return np.where((X[:, 0] < 11.0) | (X[:, 1] < 12), 1, 0)

@st.cache_resource
def load_assets():
    global MODELO_REAL_CARGADO
    
    # 1. Intentar cargar el modelo REAL
    try:
        # Reemplaza con la ruta correcta a tu modelo y a tus columnas
        model = joblib.load('modelo_anemia.joblib')
        feature_columns = joblib.load('modelo_columns.joblib')
        
        MODELO_REAL_CARGADO = True
        return model, feature_columns
        
    except Exception as e:
        # 3. Si falla, usar la simulación y las columnas por defecto
        # st.error(f"Fallo al cargar modelo real (.joblib): {e}") # Se comenta para no mostrar el error interno
        st.warning("⚠️ **USANDO PREDICCIÓN SIMULADA:** El archivo real del modelo (modelo_anemia.joblib) no se encontró o no se pudo cargar. La lógica de predicción de riesgo es CLÍNICA/MOCK.")
        
        # Columnas de la simulación. AJUSTA ESTO a las 4 primeras variables usadas en tu modelo
        feature_columns = ['Hemoglobina', 'Edad_Meses', 'Peso_Hogar', 'Ingreso_Familiar_Soles']
        
        MODELO_REAL_CARGADO = False
        return MockModel(), feature_columns

model, feature_columns = load_assets()

# --- LÓGICA DE NEGOCIO Y CLÍNICA ---

def clasificar_anemia(hemoglobina, edad_meses, altitud=1500):
    """Clasifica el riesgo de anemia basado en la hemoglobina corregida por altitud."""
    
    # Factor de Corrección por Altitud (Según MINSA simplificado)
    correccion = 0.0
    if altitud > 1000 and altitud <= 2000:
        correccion = 0.5 
    elif altitud > 2000 and altitud <= 3000:
        correccion = 0.8 
    elif altitud > 3000:
        correccion = 1.0 # Simplificación
        
    hemoglobina_corregida = hemoglobina + correccion

    # Puntos de corte de diagnóstico (según OMS/MINSA simplificado)
    # Niños de 6 a 59 meses
    if edad_meses >= 6 and edad_meses <= 59:
        if hemoglobina_corregida < 7.0:
            return "ANEMIA SEVERA", hemoglobina_corregida
        elif hemoglobina_corregida >= 7.0 and hemoglobina_corregida < 11.0:
            return "ANEMIA LEVE/MODERADA", hemoglobina_corregida
        else:
            return "NORMAL", hemoglobina_corregida
    # Otros grupos (simplificación)
    elif edad_meses < 6:
        if hemoglobina_corregida < 10.0:
            return "ANEMIA (MENOR 6 MESES)", hemoglobina_corregida
        else:
            return "NORMAL (MENOR 6 MESES)", hemoglobina_corregida
    
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
        return False, "Supabase no inicializado. Revise las credenciales."
    
    try:
        data['created_at'] = datetime.now().isoformat()
        
        response = supabase.table(DATABASE_TABLE).insert(data).execute()
        
        if response.data:
            return True, f"Registro exitoso con ID: {response.data[0]['id']}"
        else:
            # st.error(f"Respuesta de Supabase sin datos: {response}")
            return False, "Error desconocido en Supabase."

    except Exception as e:
        # st.error(f"Error al registrar la alerta en Supabase: {e}")
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
    
    # Mostrar el estado real del modelo
    if MODELO_REAL_CARGADO:
        st.sidebar.success("✅ Modelo ML Cargado (Real)")
    else:
        st.sidebar.error("❌ Modelo ML NO Cargado (Usando Simulación)")
    
    if supabase:
        st.sidebar.success("✅ DB Contacto (Supabase Activa)")
    else:
        st.sidebar.error("❌ DB Contacto (Supabase Inactiva)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Módulos de Control")
    st.sidebar.radio("Navegación", ["Predicción y Reporte", "Monitoreo de Alertas"], index=0)

    # Título Principal del Informe
    st.title("Informe Personalizado y Diagnóstico de Riesgo de Anemia (MIDIS v2.5)")
    st.markdown("---")

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
                format="%.2f",
                help="Valor sin corrección de altitud."
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
                ["LIMA (Metropolitana y Provincia)", "JUNIN (Sierra Central)", "CUSCO (Sierra Sur)"],
                index=0,
                help="Selecciona la región de residencia habitual."
            )
            # Asignación de Altitud
            altitud = 150 # Altitud de Lima (ejemplo)
            if "JUNIN" in region:
                altitud = 3200 
            elif "CUSCO" in region:
                altitud = 3400

            st.info(f"Altitud: {altitud} msnm. Usada para corrección de Hemoglobina.")

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
        st.info(f"Clima automático: **{clima_auto}**.")
        
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
            label='GENERAR INFORME PERSONALIZADO Y REGISTRAR ALERTA',
            help="Alerta y registra los datos para seguimiento.",
            type="primary"
        )


    if submit_button:
        # --- 1. Lógica Clínica / Diagnóstico ---
        diagnostico_clinico, hb_corregida = clasificar_anemia(hemoglobina, edad_meses, altitud)
        
        # --- 2. Preparación de datos para ML ---
        
        # Crear un diccionario de input, incluyendo todas las variables necesarias
        input_data = {
            'Hemoglobina': hb_corregida, # Corregida se usa para predicción
            'Edad_Meses': edad_meses,
            'Peso_Hogar': peso_hogar,
            'Ingreso_Familiar_Soles': ingreso_familiar_soles,
            # NOTA: Si tu modelo real usa más variables (ej: one-hot de 'area_residencia'),
            # DEBES incluirlas aquí antes de crear el DataFrame.
        }
        
        # Solo usar las columnas que el modelo espera y convertirlas a un array numpy
        try:
            df_input = pd.DataFrame([input_data])
            X_pred = df_input[feature_columns].values
        except KeyError as e:
            st.error(f"Error en la preparación de datos: La columna {e} no está disponible en la entrada. Revisa tu lista 'feature_columns' vs. 'input_data'.")
            return

        # --- 3. Predicción de Riesgo ML ---
        try:
            prediccion_ml = model.predict(X_pred)[0]
            
            # --- 4. Presentar Resultados ---
            st.markdown("## ✅ Resultados del Diagnóstico y Predicción de Riesgo")
            
            col_diag, col_hb_corr = st.columns(2)
            with col_hb_corr:
                st.info(f"**Hemoglobina Corregida por Altitud ({altitud} msnm):** **{hb_corregida:.2f} g/dL**")

            # Resultado Clínico
            if "ANEMIA" in diagnostico_clinico:
                col_diag.error(f"**DIAGNÓSTICO CLÍNICO:** {diagnostico_clinico}")
                alerta_status = "ALTO"
                alerta_color = "red"
            else:
                col_diag.success(f"**DIAGNÓSTICO CLÍNICO:** {diagnostico_clinico}")
                alerta_status = "BAJO"
                alerta_color = "green"
            
            # Resultado ML
            st.markdown("---")
            st.subheader("Predicción de Riesgo ML/IA")
            if not MODELO_REAL_CARGADO:
                st.markdown(f"**ATENCIÓN:** Se está usando la lógica simulada para la predicción ML.")
                st.markdown("La simulación predice **ALTO RIESGO** si la Hemoglobina < 11.0 o Edad < 12 meses.")
            
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
            if supabase is None:
                 st.info(f"Registro en DB omitido: Credenciales de Supabase no configuradas o inicialización fallida.")
            else:
                with st.spinner("Registrando alerta en la base de datos Supabase..."):
                    registro_exitoso, mensaje = registrar_alerta(alerta_data)

                if registro_exitoso:
                    st.balloons()
                    st.success(f"¡Alerta Registrada! {mensaje}. El caso de **{nombre}** ha sido notificado para seguimiento.")
                    st.json(alerta_data)
                else:
                    st.error(f"Fallo en el registro de la alerta. Mensaje: {mensaje}")
                
        except Exception as e:
            st.error(f"Ocurrió un error grave durante la predicción o post-procesamiento: {e}")
            st.code(f"Asegúrate de que tus datos de entrada coincidan con las columnas del modelo: {feature_columns}")


if __name__ == "__main__":
    app()
