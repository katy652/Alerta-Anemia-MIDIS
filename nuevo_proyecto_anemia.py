import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from supabase.client import create_client, Client # Importación correcta

# ==============================================================================
# 1. INICIALIZACIÓN DE COMPONENTES EXTERNOS
# ==============================================================================

# ⚠️ La conexión fallará si la red de Streamlit Cloud bloquea IPv4/IPv6
@st.cache_resource
def init_supabase_client() -> Client:
    """Inicializa el cliente de Supabase usando la clave de service_role."""
    try:
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            # st.error("ERROR: Credenciales Supabase no definidas.")
            return None
            
        supabase: Client = create_client(url, key)

        # PRUEBA DE CONEXIÓN REST API (Para diagnosticar bloqueo de red)
        try:
            # Intenta una consulta simple: si falla, es un problema de red/firewall
            supabase.table("alertas").select("dni").limit(1).execute()
            # st.success("✅ Conexión a Supabase REST API establecida.") 
            return supabase
        except Exception as api_e:
            st.error(f"❌ FALLO CRÍTICO DE CONEXIÓN. Error: {api_e}. La aplicación NO PUEDE establecer comunicación de red con Supabase (posible bloqueo IPv4/IPv6).")
            return None
            
    except Exception as e:
        return None

SUPABASE_CLIENT = init_supabase_client()


# ✅ Carga del Modelo Machine Learning
@st.cache_resource
def load_model():
    """Carga el modelo y las columnas."""
    try:
        # Asegúrate de que 'modelo_columns.joblib' esté en la raíz del repositorio
        model = joblib.load('modelo_columns.joblib') 
        return model, list(model.feature_names_in_)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

MODELO, COLUMNAS_MODELO = load_model()

# ==============================================================================
# 2. FUNCIONES DE LÓGICA DE NEGOCIO Y REGISTRO
# ==============================================================================

def registrar_alerta_db(data_alerta: dict) -> bool:
    """Registra una alerta en la tabla 'alertas' de Supabase."""
    
    if not SUPABASE_CLIENT:
        # El mensaje de error ya apareció en init_supabase_client()
        st.error("❌ No se puede registrar. Conexión a Supabase no disponible.") 
        return False

    try:
        # Mapeo de datos (las claves deben coincidir con el esquema de la tabla 'alertas')
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

        response = SUPABASE_CLIENT.table('alertas').insert(data_to_insert).execute()
        
        # LÓGICA DE VERIFICACIÓN DE INSERCIÓN (Errores de base de datos)
        if response.error:
            st.error(f"❌ Error de Inserción Supabase. Código: {response.error.code}. Mensaje: {response.error.message}")
            return False
        else:
            st.success(f"✅ Caso registrado ({data_alerta['DNI']}) con éxito en la base de datos.")
            return True

    except Exception as e:
        # Captura errores de red/timeouts que ocurren durante la ejecución de la inserción
        st.error(f"❌ FALLO DE RED/TIMEOUT en el registro. La aplicación no pudo completar la petición. Error: {e}")
        return False

# La función generar_prediccion() (omite por espacio, asume que es funcional)
def generar_prediccion(df_input: pd.DataFrame, modelo, columnas_modelo) -> tuple:
    # ... (Cuerpo de la función) ...
    df_modelo = df_input.reindex(columns=columnas_modelo, fill_value=0)
    prediccion = modelo.predict(df_modelo)[0]
    probabilidades = modelo.predict_proba(df_modelo)[0]
    etiquetas = {0: 'RIESGO: BAJO RIESGO', 1: 'RIESGO: MEDIO RIESGO', 2: 'RIESGO: ALTO RIESGO'}
    resultado_riesgo = etiquetas.get(prediccion, 'RIESGO: DESCONOCIDO')
    return resultado_riesgo, probabilidades


# ==============================================================================
# 3. CONFIGURACIÓN DE STREAMLIT (UI)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Diagnóstico de Riesgo de Anemia")
st.title("Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.1 Híbrida)")

tab_informe, tab_monitoreo = st.tabs(["📝 Generar Informe (Predicción)", "📊 Monitoreo y Reportes"])

with tab_informe:
    with st.form(key='informe_form'):
        # ... (Campos del formulario, omitidos por espacio, deben ser funcionales) ...
        # Se asume que estos inputs se recogen: dni, nombre_apellido, edad, hemoglobina, etc.
        st.header("0. Datos de Identificación y Contacto")
        col1, col2 = st.columns(2)
        with col1:
            dni = st.text_input("DNI/Identificación", key="input_dni")
        with col2:
            nombre_apellido = st.text_input("Nombre y Apellido", key="input_nombre")
        
        # ... (Resto de los inputs) ...
        st.header("1. Factores Clínicos y Demográficos Clave")
        col3, col4, col5 = st.columns(3)
        with col3:
            edad = st.number_input("Edad (años)", min_value=0, max_value=120, key="input_edad", value=10)
        with col4:
            hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=0.0, max_value=20.0, step=0.1, key="input_hemoglobina", value=12.5)
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
    
    
    if submitted:
        # ... (Lógica de predicción) ...
        data_dict = {'Edad': edad, 'Hemoglobina_g_dL': hemoglobina, 'Acceso a agua segura_Sí': agua_segura, 'Saneamiento básico_Sí': saneamiento, 'Vive en zona rural_Sí': zona_rural, 'Recibe programa nutricional_Sí': programa_nutricional, 'Control de CRED regular_Sí': control_cred, 'Recibió suplementos de hierro_Sí': suplementos_hierro}
        df_prediccion = pd.DataFrame([data_dict])
        
        riesgo, probabilidades = generar_prediccion(df_prediccion, MODELO, COLUMNAS_MODELO)
        prob_riesgo = probabilidades[MODELO.predict(df_prediccion)[0]] * 100 
        st.header("Análisis y Reporte de Control Oportuno")
        st.markdown(f"## {riesgo}", unsafe_allow_html=True)
        st.markdown(f"**Probabilidad de {riesgo.split(': ')[1]}:** {prob_riesgo:.2f}%")
        sugerencias = f"Recomendación para {riesgo.split(': ')[1]}. {sexo} de {edad} años."
        st.subheader("Sugerencias Personalizadas de Intervención Oportuna:")
        st.info(sugerencias)

        # 5. Preparar y Registrar en Supabase
        data_alerta_db = {
            'DNI': dni, 'Nombre_Apellido': nombre_apellido, 'Edad': edad, 'Hemoglobina_g_dL': hemoglobina,
            'Riesgo': riesgo.split(': ')[1], 'Fecha_Alerta': datetime.now().isoformat(),
            'Estado': 'PENDIENTE', 'Sugerencias': sugerencias
        }
        
        registro_exitoso = registrar_alerta_db(data_alerta_db)

        if not registro_exitoso:
            st.error("❌ El caso se procesó, pero el registro en la base de datos falló. Verifique el mensaje de error de conexión en la parte superior.")


with tab_monitoreo:
    st.header("Visualización de Casos Registrados")
    
    if SUPABASE_CLIENT:
        try:
            response = SUPABASE_CLIENT.table('alertas').select('*').order('fecha_alerta', desc=True).execute()
            if response.error:
                st.error(f"Error al cargar datos del monitoreo: {response.error.message}")
            elif len(response.data) > 0:
                df_monitoreo = pd.DataFrame(response.data)
                # ... (Lógica de visualización omitida por espacio) ...
                st.dataframe(df_monitoreo, use_container_width=True)
            else:
                st.info("Actualmente no hay casos registrados.")

        except Exception as e:
            st.error(f"❌ Error al intentar cargar datos del monitoreo: {e}")
    else:
        st.warning("⚠️ La conexión a Supabase no está disponible. No se pueden cargar los datos de monitoreo.")
