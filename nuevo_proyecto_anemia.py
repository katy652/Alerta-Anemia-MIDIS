import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# IMPORTACIONES ESPECÍFICAS DE SUPABASE
from supabase.client import create_client, Client


# ==============================================================================
# 1. INICIALIZACIÓN DE COMPONENTES EXTERNOS (MODELO Y SUPABASE)
# ==============================================================================

# ✅ Inicialización del Cliente de Supabase (Cacheado y con Manejo de Errores)
@st.cache_resource
def init_supabase_client() -> Client:
    """Inicializa y retorna el cliente de Supabase una sola vez, leyendo del entorno."""
    try:
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            # st.error("ERROR: Las credenciales SUPABASE_URL o SUPABASE_KEY NO están definidas en los Secrets.")
            return None
            
        supabase: Client = create_client(url, key)

        # 🟢 PRUEBA DE CONEXIÓN AÑADIDA: Verifica que la conexión REST funciona
        try:
            # Hacemos una consulta simple para verificar la conectividad REST API
            supabase.table("alertas").select("dni").limit(1).execute()
            # st.success("✅ Conexión a Supabase REST API establecida.")
            return supabase
        except Exception as api_e:
            # Muestra un error si la conexión REST falla (típicamente por red/firewall)
            st.error(f"❌ FALLO CRÍTICO DE RED/LIBRERÍA: La conexión a Supabase falló. Esto suele ser por bloqueo de red (IPv4/IPv6). Error: {api_e}")
            return None
            
    except Exception as e:
        # Captura cualquier error de inicialización
        # st.error(f"Error de inicialización de Supabase: {e}") 
        return None

# 🚀 Inicialización GLOBAL del cliente de Supabase (Se ejecuta una sola vez)
SUPABASE_CLIENT = init_supabase_client()


# ✅ Carga del Modelo Machine Learning (Cacheado)
@st.cache_resource
def load_model():
    """Carga el modelo de ML y la lista de columnas para garantizar el orden."""
    try:
        model = joblib.load('modelo_columns.joblib')
        return model, list(model.feature_names_in_)
    except Exception as e:
        st.error(f"Error al cargar el modelo o las columnas: {e}")
        return None, None

MODELO, COLUMNAS_MODELO = load_model()


# ==============================================================================
# 2. FUNCIONES DE LÓGICA DE NEGOCIO Y PREDICCIÓN
# ==============================================================================

def generar_prediccion(df_input: pd.DataFrame, modelo, columnas_modelo) -> tuple:
    """Asegura el orden de las columnas y genera la predicción y las probabilidades."""
    
    df_modelo = df_input.reindex(columns=columnas_modelo, fill_value=0)
    prediccion = modelo.predict(df_modelo)[0]
    probabilidades = modelo.predict_proba(df_modelo)[0]
    
    etiquetas = {0: 'RIESGO: BAJO RIESGO', 1: 'RIESGO: MEDIO RIESGO', 2: 'RIESGO: ALTO RIESGO'}
    resultado_riesgo = etiquetas.get(prediccion, 'RIESGO: DESCONOCIDO')
    
    return resultado_riesgo, probabilidades

def registrar_alerta_db(data_alerta: dict) -> bool:
    """Registra una alerta en la tabla 'alertas' de Supabase."""
    
    if not SUPABASE_CLIENT:
        st.error("❌ No se pudo registrar: La conexión a Supabase falló.")
        return False

    try:
        # 🟢 Asegúrate de que las claves coincidan con el esquema de la tabla 'alertas' (image_1df544.png)
        data_to_insert = {
            'dni': data_alerta['DNI'],
            'nombre_apellido': data_alerta['Nombre_Apellido'],
            'edad': data_alerta['Edad'],
            'hemoglobina_g_dL': data_alerta['Hemoglobina_g_dL'], 
            'riesgo': data_alerta['Riesgo'],
            'fecha_alerta': data_alerta['Fecha_Alerta'],
            'estado': data_alerta['Estado'],
            'sugerencias': data_alerta['Sugerencias'],
        }

        # Ejecutar la inserción
        response = SUPABASE_CLIENT.table('alertas').insert(data_to_insert).execute()
        
        # 🟢 LÓGICA DE VERIFICACIÓN DE INSERCIÓN (Captura errores de la base de datos)
        if response.error:
            st.error(f"❌ Error de Inserción Supabase. Código: {response.error.code}. Mensaje: {response.error.message}")
            return False
        else:
            st.success(f"✅ Caso registrado ({data_alerta['DNI']}) con éxito.")
            return True

    except Exception as e:
        # Captura errores de red/timeouts que ocurren durante la ejecución de la inserción
        st.error(f"❌ FALLO DE RED/TIMEOUT: El intento de inserción falló por un problema de conectividad. Error: {e}")
        return False

# ==============================================================================
# 3. CONFIGURACIÓN DE STREAMLIT (UI)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Diagnóstico de Riesgo de Anemia")
st.title("Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.1 Híbrida)")

# Definición de la estructura de las tabs
tab_informe, tab_monitoreo = st.tabs(["📝 Generar Informe (Predicción)", "📊 Monitoreo y Reportes"])

with tab_informe:
    with st.form(key='informe_form'):
        
        st.header("0. Datos de Identificación y Contacto")
        col1, col2 = st.columns(2)
        with col1:
            dni = st.text_input("DNI/Identificación", key="input_dni")
        with col2:
            nombre_apellido = st.text_input("Nombre y Apellido", key="input_nombre")

        # ... (Resto del formulario de entrada de datos) ...
        # Se omiten los inputs repetitivos para concisión, asumiendo que los valores
        # de 'edad', 'hemoglobina', etc. se obtienen de los inputs en el formulario.

        st.header("1. Factores Clínicos y Demográficos Clave")
        col3, col4, col5 = st.columns(3)
        with col3:
            edad = st.number_input("Edad (años)", min_value=0, max_value=120, key="input_edad")
        with col4:
            hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=0.0, max_value=20.0, step=0.1, key="input_hemoglobina")
        with col5:
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"], key="input_sexo")

        st.header("2. Factores Socioeconómicos y Contextuales")
        col6, col7, col8 = st.columns(3)
        with col6:
            agua_segura = st.selectbox("Acceso a agua segura", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key="input_agua")
        with col7:
            saneamiento = st.selectbox("Saneamiento básico", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key="input_saneamiento")
        with col8:
            zona_rural = st.selectbox("Vive en zona rural", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key="input_rural")

        st.header("3. Acceso a Programas y Servicios")
        col9, col10, col11 = st.columns(3)
        with col9:
            programa_nutricional = st.selectbox("Recibe programa nutricional", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key="input_programa")
        with col10:
            control_cred = st.selectbox("Control de CRED regular", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key="input_cred")
        with col11:
            suplementos_hierro = st.selectbox("Recibió suplementos de hierro", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No", key="input_hierro")


        submitted = st.form_submit_button("GENERAR INFORME PERSONALIZADO Y REGISTRAR CASO")
    
    
    # Lógica al enviar el formulario
    if submitted:
        if MODELO is None or COLUMNAS_MODELO is None:
            st.error("No se puede generar el informe. El modelo ML no se cargó correctamente.")
        elif not dni or not nombre_apellido:
            st.warning("Por favor, complete los campos DNI y Nombre/Apellido para el registro.")
        else:
            # 1. Preparar datos para la predicción
            data_dict = {
                'Edad': edad,
                'Hemoglobina_g_dL': hemoglobina,
                'Acceso a agua segura_Sí': agua_segura,
                'Saneamiento básico_Sí': saneamiento,
                'Vive en zona rural_Sí': zona_rural,
                'Recibe programa nutricional_Sí': programa_nutricional,
                'Control de CRED regular_Sí': control_cred,
                'Recibió suplementos de hierro_Sí': suplementos_hierro
            }
            df_prediccion = pd.DataFrame([data_dict])
            
            # 2. Generar la predicción
            riesgo, probabilidades = generar_prediccion(df_prediccion, MODELO, COLUMNAS_MODELO)
            prob_riesgo = probabilidades[MODELO.predict(df_prediccion)[0]] * 100 

            # 3. Mostrar el informe
            st.header("Análisis y Reporte de Control Oportuno")
            st.markdown(f"## {riesgo}", unsafe_allow_html=True)
            st.markdown(f"**Probabilidad de {riesgo.split(': ')[1]}:** {prob_riesgo:.2f}%")
            
            # 4. Generar sugerencias
            sugerencias = f"Recomendación para {riesgo.split(': ')[1]}. {sexo} de {edad} años."
            st.subheader("Sugerencias Personalizadas de Intervención Oportuna:")
            st.info(sugerencias)

            # 5. Preparar y Registrar en Supabase
            data_alerta_db = {
                'DNI': dni,
                'Nombre_Apellido': nombre_apellido,
                'Edad': edad,
                'Hemoglobina_g_dL': hemoglobina,
                'Riesgo': riesgo.split(': ')[1],
                'Fecha_Alerta': datetime.now().isoformat(),
                'Estado': 'PENDIENTE', 
                'Sugerencias': sugerencias
            }
            
            # Intentar el registro en la base de datos
            registro_exitoso = registrar_alerta_db(data_alerta_db)

            if not registro_exitoso:
                st.error("❌ El caso se procesó (riesgo: {riesgo}), pero el registro en la base de datos falló. Verifique el mensaje de error anterior para el diagnóstico (red/datos).")


with tab_monitoreo:
    st.header("Visualización de Casos Registrados")
    
    if SUPABASE_CLIENT:
        try:
            # Trae todos los datos de la tabla 'alertas'
            response = SUPABASE_CLIENT.table('alertas').select('*').order('fecha_alerta', desc=True).execute()
            
            if response.error:
                st.error(f"Error al cargar datos del monitoreo: {response.error.message}")
            elif len(response.data) > 0:
                df_monitoreo = pd.DataFrame(response.data)
                
                # Formato y limpieza para visualización
                df_monitoreo['fecha_alerta'] = pd.to_datetime(df_monitoreo['fecha_alerta']).dt.strftime('%Y-%m-%d %H:%M')
                
                # ... (Resto de la lógica de monitoreo) ...
                # Se omite el código de visualización de tablas/gráficos para concisión

                # Muestra el DataFrame, asumiendo que el código de visualización es funcional
                st.dataframe(df_monitoreo, use_container_width=True)
                
            else:
                st.info("Actualmente no hay casos registrados en la tabla de alertas. Registre un caso en la pestaña 'Generar Informe'.")

        except Exception as e:
            st.error(f"❌ Error al intentar conectar con la tabla 'alertas' para monitoreo: {e}")
    else:
        st.warning("⚠️ La conexión a Supabase no está disponible. No se pueden cargar los datos de monitoreo.")
