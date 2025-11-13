import streamlit as st
import pandas as pd
import joblib
import unidecode
from supabase import create_client, Client
import datetime
from fpdf import FPDF
import base64
import json
import re
import os
import plotly.express as px
import requests

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELO
# ==============================================================================

# Configuración de página
st.set_page_config(
    page_title="Alerta de Riesgo de Anemia (IA)",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes de Umbral ---
UMBRAL_SEVERA = 7.0
UMBRAL_MODERADA = 9.0
UMBRAL_HEMOGLOBINA_ANEMIA = 11.0

# --- Nombres de Archivo ---
MODEL_FILENAME = "modelo_anemia.joblib"
# ID del archivo público de Google Drive
DRIVE_FILE_ID = "1vij71K2DtTHEc1seEOqeYk-fV2AQNfBK" 
COLUMNS_FILENAME = "modelo_columns.joblib"

# La función get_confirm_token se mantiene ya que es necesaria para la URL de confirmación
def get_confirm_token(response):
    """Extrae el token de confirmación necesario para descargar archivos grandes."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_file_from_google_drive(id, destination):
    """
    Descarga un archivo grande de Google Drive a una ubicación local.
    
    Se ha modificado para usar la URL de exportación más estable.
    """
    # Usaremos el endpoint estándar para manejar el token de archivos grandes.
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    # 1. Primer request para obtener el token de confirmación (si es necesario)
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        # 2. Segundo request con el token de confirmación
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    # 3. Manejo de errores
    if response.status_code != 200:
        st.error(f"❌ Error HTTP {response.status_code} al descargar el modelo. Razón: {response.reason}")
        st.error("Verifique que el archivo en Drive esté configurado como 'Cualquier persona con el enlace'.")
        return False
    
    # Manejo de redirección (a veces Drive redirige a otra URL de descarga)
    if 'Content-Disposition' not in response.headers:
        st.warning("⚠️ La respuesta de Drive no contiene encabezados de archivo. Esto puede indicar un problema, pero continuamos con la descarga.")
        
    # 4. Escribir el contenido del archivo descargado
    chunk_size = 32768
    try:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk: # filtrar los chunks vacíos
                    f.write(chunk)
            
        return os.path.exists(destination)
    except Exception as e:
        st.error(f"❌ Error al guardar el archivo descargado: {e}")
        return False

@st.cache_resource
def load_model_components():
    """Carga el modelo ML y los activos de columnas, con descarga automática si es necesario."""
    modelo = None
    model_columns = None
    
    # 1️⃣ Cargar Columnas
    try:
        model_columns = joblib.load(COLUMNS_FILENAME)
        st.success("✅ Activos de columna cargados exitosamente.")
    except FileNotFoundError:
        st.error(f"❌ CRÍTICO: No se encontró el archivo '{COLUMNS_FILENAME}'. La IA está deshabilitada.")
        return None, None
    except Exception as e:
        st.error(f"❌ ERROR al cargar las columnas: {e}")
        return None, None
        
    # 2️⃣ Lógica robusta de Carga/Descarga del Modelo
    try:
        model = joblib.load(MODEL_FILENAME)
        st.success("✅ Modelo de IA cargado correctamente desde almacenamiento local.")
        return model, model_columns
    except FileNotFoundError:
        st.warning(f"⚠️ Archivo '{MODEL_FILENAME}' no encontrado localmente. Intentando descargar desde Google Drive...")
        
        # *** LÓGICA DE DESCARGA ***
        if download_file_from_google_drive(DRIVE_FILE_ID, MODEL_FILENAME):
            try:
                model = joblib.load(MODEL_FILENAME)
                st.success("✅ Modelo de IA cargado correctamente.")
                return model, model_columns
            except Exception as e_retry:
                st.error(f"❌ ERROR CRÍTICO al cargar el modelo recién descargado. Detalle: {e_retry}")
                if os.path.exists(MODEL_FILENAME):
                    os.remove(MODEL_FILENAME)
                return None, model_columns
        else:
            st.error("❌ Falló la descarga desde Google Drive. La predicción de IA está deshabilitada.")
            return None, model_columns
            
    except Exception as e:
        st.error(f"❌ ERROR CRÍTICO al cargar el modelo local '{MODEL_FILENAME}'. Detalle: {e}")
        st.warning("⚠️ La predicción de IA está temporalmente deshabilitada.")
        return None, model_columns

# Inicialización de la Carga de Componentes
MODELO_ML, MODELO_COLUMNS = load_model_components()

# ==============================================================================
# 2. FUNCIONES AUXILIARES CLÍNICO-GEOGRÁFICAS
# ==============================================================================

def get_altitud_por_region(region):
    """Asigna una altitud representativa (en msnm) a cada región."""
    altitudes = {
        # Costa y Baja Altitud
        "LIMA (Metropolitana y Provincia)": 150,
        "CALLAO (Provincia Constitucional)": 20,
        "PIURA": 100, "LAMBAYEQUE": 50, "LA LIBERTAD": 100, "ICA": 150,
        "TUMBES": 20, "ÁNCASH (Costa)": 500,
        # Selva
        "LORETO": 150, "AMAZONAS": 500, "SAN MARTÍN": 300, "UCAYALI": 200, "MADRE DE DIOS": 250,
        # Sierra y Andes (Media Altitud)
        "HUÁNUCO": 1800, "JUNÍN (Andes)": 3300, "CUSCO (Andes)": 3400, "AYACUCHO": 2700,
        "APURÍMAC": 2900, "CAJAMARCA": 2700, "AREQUIPA": 2300, "MOQUEGUA": 1500,
        "TACNA": 1000, "PASCO": 4300,
        # Sierra Alta (Muy Alta Altitud)
        "PUNO (Sierra Alta)": 3800, 
        "HUANCAVELICA (Sierra Alta)": 3600,
        "OTRO / NO ESPECIFICADO": 500
    }
    return altitudes.get(region, 500)

def get_clima_por_region(region):
    """Asigna un clima predominante a cada región para el reporte."""
    climas = {
        "LIMA (Metropolitana y Provincia)": "Costero árido",
        "CALLAO (Provincia Constitucional)": "Costero árido",
        "PIURA": "Tropical seco", "LAMBAYEQUE": "Costero semiárido", 
        "LA LIBERTAD": "Costero semiárido", "ICA": "Desértico",
        "TUMBES": "Tropical húmedo", "ÁNCASH (Costa)": "Costero árido",
        "HUÁNUCO": "Templado de valle", "JUNÍN (Andes)": "Andino de altura", 
        "CUSCO (Andes)": "Andino de altura", "AYACUCHO": "Templado seco",
        "APURÍMAC": "Andino templado", "CAJAMARCA": "Templado serrano", 
        "AREQUIPA": "Templado seco", "MOQUEGUA": "Templado de valle",
        "TACNA": "Costero árido/Templado",
        "PUNO (Sierra Alta)": "Frío andino", "HUANCAVELICA (Sierra Alta)": "Frío andino", 
        "PASCO": "Frío de puna",
        "LORETO": "Selva tropical", "AMAZONAS": "Selva tropical de montaña", 
        "SAN MARTÍN": "Selva alta", "UCAYALI": "Selva tropical", 
        "MADRE DE DIOS": "Selva tropical",
        "OTRO / NO ESPECIFICADO": "No definido"
    }
    return climas.get(region, "No definido")

def calcular_correccion_altitud(altitud_m):
    """Calcula la corrección de hemoglobina según la altitud (ajustada para niños)."""
    if altitud_m < 1000:
        return 0.0
    elif altitud_m < 2000:
        return 0.2
    elif altitud_m < 3000:
        return 0.5
    elif altitud_m < 4000:
        return 0.8
    else: # > 4000 msnm
        return 1.2
    
def get_umbral_anemia_por_edad(edad_meses):
    """Define el umbral de hemoglobina (sin corrección) según la edad del paciente."""
    if 12 <= edad_meses <= 59:
        return 11.0
    else:
        return 11.0

def clasificar_anemia_clinica(hemoglobina_medida, edad_meses, altitud_m):
    """
    Realiza la clasificación clínica de la anemia (OMS) con ajuste por altitud.
    Retorna: (gravedad, umbral_clinico, hb_corregida, correccion_alt)
    """
    correccion = calcular_correccion_altitud(altitud_m)
    hb_corregida = hemoglobina_medida + correccion
    umbral_base = get_umbral_anemia_por_edad(edad_meses)
    
    umbral_ajustado = umbral_base
    
    if hb_corregida < UMBRAL_SEVERA:
        gravedad = "SEVERA"
    elif hb_corregida < UMBRAL_MODERADA:
        gravedad = "MODERADA"
    elif hb_corregida < umbral_ajustado:
        gravedad = "LEVE"
    else:
        gravedad = "NORMAL (Sin Anemia)"
        
    return gravedad, umbral_ajustado, hb_corregida, correccion

# ==============================================================================
# 3. PREDICCIÓN CON EL MODELO DE MACHINE LEARNING
# ==============================================================================

def preprocess_data(data):
    """
    Transforma el diccionario de entrada a un DataFrame con One-Hot Encoding
    y lo alinea con las columnas de entrenamiento del modelo.
    """
    if MODELO_COLUMNS is None:
        return None

    df = pd.DataFrame([data])

    # Convertir 'Sexo' a 0/1
    df['Sexo_Masc'] = df['Sexo'].apply(lambda x: 1 if x == 'Masculino' else 0)
    df = df.drop(columns=['Sexo'])
    
    # OHE para variables categóricas
    ohe_cols = ['Region', 'Area', 'Clima', 'Nivel_Educacion_Madre', 'Programa_QaliWarma', 'Programa_Juntos', 'Programa_VasoLeche', 'Suplemento_Hierro']
    
    # Limpieza de strings antes de OHE
    for col in ohe_cols:
        df[col] = df[col].apply(lambda x: unidecode.unidecode(str(x)).lower().replace(" ", "_").replace("(", "").replace(")", ""))

    df_processed = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols, dtype=int)
    
    final_features = list(MODELO_COLUMNS)
    missing_cols = set(final_features) - set(df_processed.columns)
    
    for c in missing_cols:
        df_processed[c] = 0
        
    df_final = df_processed[final_features]
    
    # Eliminar columnas de identificación
    if 'DNI' in df_final.columns: df_final = df_final.drop(columns=['DNI'])
    if 'Nombre_Apellido' in df_final.columns: df_final = df_final.drop(columns=['Nombre_Apellido'])

    # Aplicar la corrección de hemoglobina (la variable Hemoglobina_g_dL del modelo es la corregida)
    altitud_m_val = df['Altitud_m'].iloc[0]
    correccion = calcular_correccion_altitud(altitud_m_val)
    df_final['Hemoglobina_g_dL'] = df_final['Hemoglobina_g_dL'] + correccion
    
    return df_final

def predict_risk_ml(data):
    """
    Realiza la predicción de riesgo de alto impacto (Anemia Moderada/Severa)
    utilizando el modelo de Machine Learning.
    Retorna: (prob_alto_riesgo, resultado_ml)
    """
    if MODELO_ML is None:
        return 0.0, "NO APLICA (Modelo IA Deshabilitado)"

    df_final = preprocess_data(data)
    
    if df_final is None or df_final.empty:
        return 0.0, "ERROR (Fallo en preprocesamiento)"

    try:
        prob_alto_riesgo = MODELO_ML.predict_proba(df_final)[:, 1][0]
        
        if prob_alto_riesgo >= 0.75:
            resultado_ml = f"ALTO RIESGO (Vulnerabilidad IA > 75%)"
        elif prob_alto_riesgo >= 0.50:
            resultado_ml = f"ALTO RIESGO (Vulnerabilidad IA > 50%)"
        elif prob_alto_riesgo >= 0.35:
            resultado_ml = f"MEDIO RIESGO (Vulnerabilidad IA)"
        else:
            resultado_ml = "BAJO RIESGO (Vulnerabilidad IA)"
            
        return prob_alto_riesgo, resultado_ml
        
    except Exception as e:
        st.error(f"❌ ERROR durante la predicción del modelo: {e}")
        return 0.0, "ERROR (Fallo en predicción)"

# ==============================================================================
# 4. FUNCIONES DE BASE DE DATOS Y GENERACIÓN DE REPORTES (PDF)
# ==============================================================================

# --- Supabase Initialization ---
@st.cache_resource
def get_supabase_client():
    """Inicializa y retorna el cliente Supabase."""
    try:
        url = st.secrets["supabase_url"]
        key = st.secrets["supabase_key"]
    except:
        url = os.environ.get("SUPABASE_URL_FALLBACK", "https://FALLBACK_URL.supabase.co")
        key = os.environ.get("SUPABASE_KEY_FALLBACK", "FALLBACK_KEY")
        if "FALLBACK_URL" in url:
            st.warning("⚠️ Usando claves FALLBACK de Supabase. La conexión puede fallar en producción.")
            return None

    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"❌ Error de conexión a Supabase: {e}")
        return None

def registrar_alerta_db(alerta_data):
    """Registra la alerta de riesgo en la base de datos de Supabase."""
    supabase = get_supabase_client()
    if supabase is None:
        st.error("❌ Registro en DB fallido: Cliente Supabase no disponible.")
        return

    try:
        if alerta_data['gravedad_anemia'] in ['SEVERA', 'MODERADA']:
            estado_inicial = "PENDIENTE (CLÍNICO URGENTE)"
        elif alerta_data['riesgo'].startswith("ALTO RIESGO"):
            estado_inicial = "PENDIENTE (IA/VULNERABILIDAD)"
        else:
            estado_inicial = "REGISTRADO"

        data_to_insert = {
            "dni": alerta_data['DNI'],
            "nombre": alerta_data['Nombre_Apellido'],
            "hemoglobina_g_dl": alerta_data['Hemoglobina_g_dL'],
            "edad_meses": alerta_data['Edad_meses'],
            "riesgo_final": alerta_data['riesgo'],
            "gravedad_anemia": alerta_data['gravedad_anemia'],
            "sugerencias": " | ".join(alerta_data['sugerencias']),
            "fecha_alerta": datetime.datetime.now().isoformat(),
            "estado_gestion": estado_inicial,
            "region": alerta_data['Region']
        }

        response = supabase.table("alertas_anemia").insert(data_to_insert).execute()
        
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            st.toast(f"✅ Alerta para DNI {alerta_data['DNI']} registrada en Supabase.", icon='💾')
        else:
             st.error(f"❌ Error al insertar en Supabase: Respuesta inesperada o vacía. Detalle: {response}")

    except Exception as e:
        st.error(f"❌ Error al intentar registrar en Supabase (Excepción): {e}")

def actualizar_estado_alerta(dni, fecha_alerta_str, nuevo_estado):
    """Actualiza el estado de gestión de una alerta específica en Supabase."""
    supabase = get_supabase_client()
    if supabase is None: return False

    try:
        response = supabase.table("alertas_anemia").update({"estado_gestion": nuevo_estado}).match({"dni": dni, "fecha_alerta": fecha_alerta_str}).execute()

        if response.data and len(response.data) > 0:
            return True
        else:
            st.error(f"No se pudo actualizar la alerta DNI: {dni} / Fecha: {fecha_alerta_str}. (0 filas afectadas)")
            return False

    except Exception as e:
        st.error(f"Error al actualizar estado en Supabase: {e}")
        return False

def obtener_alertas_pendientes_o_seguimiento():
    """Obtiene registros con estado 'PENDIENTE' o 'EN SEGUIMIENTO'."""
    supabase = get_supabase_client()
    if supabase is None: 
        return pd.DataFrame()

    try:
        estados_activos = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO"]
        
        response = supabase.table("alertas_anemia").select('*').in_('estado_gestion', estados_activos).order('fecha_alerta', desc=True).limit(100).execute()
        
        data = response.data
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        df.rename(columns={
            'nombre': 'Nombre', 'dni': 'DNI', 'hemoglobina_g_dl': 'Hb Inicial',
            'riesgo_final': 'Riesgo', 'fecha_alerta': 'Fecha Alerta',
            'estado_gestion': 'Estado', 'sugerencias': 'Sugerencias',
            'id': 'ID_DB', 'region': 'Region'
        }, inplace=True)
        
        df['ID_GESTION'] = df['DNI'].astype(str) + "_" + df['Fecha Alerta'].astype(str)
        
        return df[['ID_DB', 'DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION']]
    
    except Exception as e:
        st.session_state['supabase_error_historial'] = str(e)
        st.error(f"❌ Error al obtener alertas: {e}")
        return pd.DataFrame()

def obtener_todos_los_registros():
    """Obtiene todo el historial de registros."""
    supabase = get_supabase_client()
    if supabase is None: 
        return pd.DataFrame()

    try:
        response = supabase.table("alertas_anemia").select('*').order('fecha_alerta', desc=True).limit(1000).execute()
        
        data = response.data
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        df.rename(columns={
            'nombre': 'Nombre', 'dni': 'DNI', 'hemoglobina_g_dl': 'Hb Inicial',
            'edad_meses': 'Edad (meses)', 'riesgo_final': 'Riesgo',
            'gravedad_anemia': 'Gravedad Clínica', 'fecha_alerta': 'Fecha Alerta',
            'estado_gestion': 'Estado', 'sugerencias': 'Sugerencias',
            'id': 'ID_DB', 'region': 'Region'
        }, inplace=True)
        
        cols_finales = ['ID_DB', 'DNI', 'Nombre', 'Region', 'Edad (meses)', 'Hb Inicial', 'Gravedad Clínica', 'Riesgo', 'Estado', 'Fecha Alerta', 'Sugerencias']
        return df[df.columns.intersection(cols_finales)]
    
    except Exception as e:
        st.session_state['supabase_error_historial'] = str(e)
        st.error(f"❌ Error al obtener el historial completo: {e}")
        return pd.DataFrame()

# --- Reporte PDF (FPDF) ---
class PDF(FPDF):
    """Clase personalizada para el informe PDF usando FPDF."""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Sistema de Alerta de Anemia - Informe Individual', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} | Fecha: {datetime.date.today().isoformat()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
        
def generar_informe_pdf_fpdf(data, resultado_final, prob_alto_riesgo, sugerencias, gravedad_anemia):
    """Genera el informe PDF usando FPDF y lo retorna como bytes."""
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- 1. Datos del Paciente y Diagnóstico ---
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_title('1. Información de Identificación y Contexto')
    pdf.set_font('Arial', '', 10)
    
    info_paciente = [
        f"DNI: {data['DNI']} | Nombre: {data['Nombre_Apellido']}",
        f"Edad (Meses): {data['Edad_meses']} | Sexo: {data['Sexo']}",
        f"Región: {data['Region']} ({data['Altitud_m']} msnm) | Área: {data['Area']}",
        f"Clima: {data['Clima']} | Ingreso Familiar (Soles/mes): {data['Ingreso_Familiar_Soles']}",
        f"Nivel Educ. Madre: {data['Nivel_Educacion_Madre']} | Nro. Hijos: {data['Nro_Hijos']}",
    ]
    
    for line in info_paciente:
        pdf.cell(0, 6, line, 0, 1)
    pdf.ln(5)

    # --- 2. Resultados Clave ---
    pdf.chapter_title('2. Diagnóstico de Riesgo y Hemoglobina')
    pdf.set_font('Arial', '', 10)
    
    pdf.cell(0, 6, f"Hemoglobina Medida: {data['Hemoglobina_g_dL']} g/dL", 0, 1)
    pdf.cell(0, 6, f"Corrección por Altitud (+{abs(data['Hemoglobina_g_dL'] - data['hb_corregida']):.1f} g/dL): {data['hb_corregida']:.1f} g/dL", 0, 1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, f"GRAVEDAD CLÍNICA (HB CORREGIDA): {gravedad_anemia}", 0, 1)
    pdf.set_font('Arial', 'B', 11)
    
    if resultado_final.startswith("ALTO"):
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 8, f"RESULTADO FINAL DE RIESGO: {resultado_final}", 1, 1, 'C')
    elif resultado_final.startswith("MEDIO"):
        pdf.set_text_color(255, 165, 0)
        pdf.cell(0, 8, f"RESULTADO FINAL DE RIESGO: {resultado_final}", 1, 1, 'C')
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, f"RESULTADO FINAL DE RIESGO: {resultado_final}", 1, 1, 'C')
        
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Probabilidad de Alto Riesgo (IA): {prob_alto_riesgo:.2%}", 0, 1)
    pdf.ln(5)

    # --- 3. Sugerencias de Intervención ---
    pdf.chapter_title('3. Sugerencias de Intervención Oportuna')
    pdf.set_font('Arial', '', 10)
    
    for i, sug in enumerate(sugerencias):
        sug_formato = sug.replace('|', ' - ')
        pdf.multi_cell(0, 5, f"{i+1}. {sug_formato}")
        
    pdf.ln(5)
    
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(0, 4, "Nota: Este informe es una herramienta de apoyo al diagnóstico. Las decisiones clínicas deben ser tomadas por personal de salud calificado.")

    return pdf.output(dest='S').encode('latin1')

def generar_sugerencias(data, resultado_final, gravedad_anemia):
    """Genera un listado de sugerencias basadas en el diagnóstico y factores de riesgo."""
    sugerencias = []
    
    # 1. Sugerencia Clínica Urgente (Máxima Prioridad)
    if gravedad_anemia == 'SEVERA':
        sugerencias.append("ALERTA CRÍTICA|Referir inmediatamente a EESS (Establecimiento de Salud) de mayor complejidad o a hospital para manejo especializado y posible transfusión.")
    elif gravedad_anemia == 'MODERADA':
        sugerencias.append("ALERTA|Iniciar o continuar tratamiento con suplemento de hierro terapéutico (ej. sulfato ferroso) y monitoreo estricto en 30 días.")
    elif gravedad_anemia == 'LEVE':
        sugerencias.append("TRATAMIENTO|Asegurar el inicio/continuación de suplementación preventiva o terapéutica y evaluación nutricional del niño.")
    else:
        sugerencias.append("PREVENCIÓN|Mantener controles de hemoglobina semestrales y asegurar la suplementación preventiva con Micronutrientes o Gotas de Hierro.")

    # 2. Sugerencias por Vulnerabilidad (ML/Socioeconómico)
    if resultado_final.startswith("ALTO RIESGO"):
        if data['Ingreso_Familiar_Soles'] < 1000:
            sugerencias.append("VULNERABILIDAD ECONÓMICA|Priorizar la atención del caso debido al bajo ingreso familiar. Evaluar acceso a programas sociales.")
        if data['Nivel_Educacion_Madre'] in ["Primaria", "Inicial", "Sin Nivel"]:
            sugerencias.append("VULNERABILIDAD EDUCATIVA|Reforzar consejería nutricional y adherencia al tratamiento, adaptando materiales a bajo nivel educativo.")
        if data['Area'] == 'Rural':
            sugerencias.append("ACCESIBILIDAD|Coordinar visita domiciliaria o seguimiento telefónico por posible barrera geográfica de acceso a EESS.")

    # 3. Sugerencias por Programas (Falta de acceso a beneficios)
    if gravedad_anemia != 'NORMAL (Sin Anemia)':
        if data['Programa_Juntos'] == 'No' and data['Area'] == 'Rural':
            sugerencias.append("GESTIÓN SOCIAL|Orientar a la familia sobre la postulación o registro al programa JUNTOS para incentivar controles de salud.")
        if data['Programa_QaliWarma'] == 'No' and data['Edad_meses'] > 24:
             sugerencias.append("GESTIÓN ESCOLAR|Verificar la inclusión en el programa Qali Warma al ingresar al nivel inicial (si aplica) para asegurar alimentación de calidad.")
        if data['Suplemento_Hierro'] == 'No':
            sugerencias.append("URGENCIA|Asegurar inmediatamente la entrega y correcto uso del suplemento de hierro (Gotas o Micronutrientes) según norma técnica.")

    # 4. Sugerencia por entorno geográfico
    if data['Altitud_m'] >= 3000 and gravedad_anemia != 'NORMAL (Sin Anemia)':
        sugerencias.append("ALTA ALTITUD|Considerar que la recuperación puede ser más lenta y garantizar la adherencia estricta al tratamiento por el estrés hipóxico.")

    return list(dict.fromkeys(sugerencias))

# ==============================================================================
# 5. VISTAS DE LA APLICACIÓN (STREAMLIT UI)
# ==============================================================================

def vista_prediccion():
    st.title("📝 Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.5 Altitud y Clima Automatizados)")
    st.markdown("---")

    if MODELO_COLUMNS is None:
        st.error(f"❌ El formulario está deshabilitado. No se pudo cargar los archivos necesarios. Revise los errores críticos de arriba.")
        return

    if MODELO_ML is None:
        st.warning("⚠️ El motor de Predicción de IA no está disponible. Solo se realizarán la **Clasificación Clínica** y la **Generación de PDF**.")

    REGIONES_PERU = [
        "LIMA (Metropolitana y Provincia)", "CALLAO (Provincia Constitucional)",
        "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ICA", "TUMBES", "ÁNCASH (Costa)",
        "HUÁNUCO", "JUNÍN (Andes)", "CUSCO (Andes)", "AYACUCHO", "APURÍMAC",
        "CAJAMARCA", "AREQUIPA", "MOQUEGUA", "TACNA",
        "PUNO (Sierra Alta)", "HUANCAVELICA (Sierra Alta)", "PASCO",
        "LORETO", "AMAZONAS", "SAN MARTÍN", "UCAYALI", "MADRE DE DIOS",
        "OTRO / NO ESPECIFICADO"
    ]

    if 'prediction_done' not in st.session_state: st.session_state.prediction_done = False
    
    with st.form("formulario_prediccion"):
        st.subheader("0. Datos de Identificación y Contacto")
        col_dni, col_nombre = st.columns(2)
        with col_dni: dni = st.text_input("DNI del Paciente", max_chars=8, placeholder="Solo 8 dígitos")
        with col_nombre: nombre = st.text_input("Nombre y Apellido", placeholder="Ej: Ana Torres")
        st.markdown("---")
        
        st.subheader("1. Factores Clínicos y Demográficos Clave")
        col_h, col_e, col_r = st.columns(3)
        with col_h: hemoglobina = st.number_input("Hemoglobina (g/dL) - CRÍTICO", min_value=5.0, max_value=18.0, value=10.5, step=0.1)
        with col_e: edad_meses = st.slider("Edad (meses)", min_value=12, max_value=60, value=36)
        with col_r: region = st.selectbox("Región (Define Altitud y Clima)", options=REGIONES_PERU)
        
        altitud_calculada = get_altitud_por_region(region)
        st.info(f"📍 Corrección por Altitud: Se usará **{altitud_calculada} msnm** (Altitud de {region}) para ajustar la HB. **Corrección: +{calcular_correccion_altitud(altitud_calculada):.1f} g/dL**.")
        st.markdown("---")
        
        st.subheader("2. Factores Socioeconómicos y Contextuales")
        
        clima_calculado = get_clima_por_region(region)
        clima = clima_calculado 
        
        col_c, col_ed = st.columns(2)
        with col_c:
            st.markdown(f"**Clima Predominante (Automático):**")
            st.markdown(f"*{clima}*")
            st.info(f"El clima asignado automáticamente para **{region}** es: **{clima}**.")
            
        with col_ed: educacion_madre = st.selectbox("Nivel Educ. Madre", options=["Secundaria", "Primaria", "Superior Técnica", "Universitaria", "Inicial", "Sin Nivel"])
        
        col_hijos, col_ing, col_area, col_s = st.columns(4)
        with col_hijos: nro_hijos = st.number_input("Nro. de Hijos en el Hogar", min_value=1, max_value=15, value=2)
        with col_ing: ingreso_familiar = st.number_input("Ingreso Familiar (Soles/mes)", min_value=0.0, max_value=5000.0, value=1800.0, step=10.0)
        with col_area: area = st.selectbox("Área de Residencia", options=['Urbana', 'Rural'])
        with col_s: sexo = st.selectbox("Sexo", options=["Femenino", "Masculino"])
        st.markdown("---")
        
        st.subheader("3. Acceso a Programas y Servicios")
        col_q, col_j, col_v, col_hierro = st.columns(4)
        with col_q: qali_warma = st.radio("Programa Qali Warma", options=["No", "Sí"], horizontal=True)
        with col_j: juntos = st.radio("Programa Juntos", options=["No", "Sí"], horizontal=True)
        with col_v: vaso_leche = st.radio("Programa Vaso de Leche", options=["No", "Sí"], horizontal=True)
        with col_hierro: suplemento_hierro = st.radio("Recibe Suplemento de Hierro", options=["No", "Sí"], horizontal=True)
        st.markdown("---")
        
        predict_button = st.form_submit_button("GENERAR INFORME PERSONALIZADO Y REGISTRAR CASO", type="primary", use_container_width=True)
        st.markdown("---")

        if predict_button:
            if not dni or len(dni) != 8: st.error("Por favor, ingrese un DNI válido de 8 dígitos."); return
            if not nombre: st.error("Por favor, ingrese un nombre."); return
            
            data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'Altitud_m': altitud_calculada, 'Sexo': sexo, 'Region': region, 'Area': area, 'Clima': clima, 'Ingreso_Familiar_Soles': ingreso_familiar, 'Nivel_Educacion_Madre': educacion_madre, 'Nro_Hijos': nro_hijos, 'Programa_QaliWarma': qali_warma, 'Programa_Juntos': juntos, 'Programa_VasoLeche': vaso_leche, 'Suplemento_Hierro': suplemento_hierro}

            gravedad_anemia, umbral_clinico, hb_corregida, correccion_alt = clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_calculada)
            prob_alto_riesgo, resultado_ml = predict_risk_ml(data)

            if gravedad_anemia in ['SEVERA', 'MODERADA']:
                resultado_final = f"ALTO RIESGO (Alerta Clínica - {gravedad_anemia})"
            elif resultado_ml.startswith("ALTO RIESGO"):
                resultado_final = f"ALTO RIESGO (Predicción ML - Anemia {gravedad_anemia})"
            else:
                resultado_final = resultado_ml
            
            data['hb_corregida'] = hb_corregida
            data['correccion_alt'] = correccion_alt

            sugerencias_finales = generar_sugerencias(data, resultado_final, gravedad_anemia)
            
            alerta_data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'riesgo': resultado_final, 'gravedad_anemia': gravedad_anemia, 'sugerencias': sugerencias_finales, 'Region': region}

            registrar_alerta_db(alerta_data)

            st.session_state.resultado = resultado_final
            st.session_state.prob_alto_riesgo = prob_alto_riesgo
            st.session_state.gravedad_anemia = gravedad_anemia
            st.session_state.sugerencias_finales = sugerencias_finales
            st.session_state.data_reporte = data
            st.session_state.hb_corregida = hb_corregida
            st.session_state.correccion_alt = correccion_alt
            st.session_state.prediction_done = True
            st.rerun()

    if st.session_state.prediction_done:
        resultado_final = st.session_state.resultado
        prob_alto_riesgo = st.session_state.prob_alto_riesgo
        gravedad_anemia = st.session_state.gravedad_anemia
        sugerencias_finales = st.session_state.sugerencias_finales
        data_reporte = st.session_state.data_reporte
        hb_corregida = st.session_state.hb_corregida
        correccion_alt = st.session_state.correccion_alt
        
        st.header("Análisis y Reporte de Control Oportuno")
        if resultado_final.startswith("ALTO"): st.error(f"## 🔴 RIESGO: {resultado_final}")
        elif resultado_final.startswith("MEDIO"): st.warning(f"## 🟠 RIESGO: {resultado_final}")
        else: st.success(f"## 🟢 RIESGO: {resultado_final}")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1: st.metric(label="Hemoglobina Medida (g/dL)", value=data_reporte['Hemoglobina_g_dL'])
        with col_res2: st.metric(label=f"Corrección por Altitud ({data_reporte['Altitud_m']}m)", value=f"+{abs(correccion_alt):.1f} g/dL")
        with col_res3: st.metric(label="Hemoglobina Corregida (g/dL)", value=f"**{hb_corregida:.1f}**", delta=f"Gravedad: {gravedad_anemia}")
        
        st.metric(label="Prob. de Alto Riesgo por IA", value=f"{prob_alto_riesgo:.2%}")
        
        st.subheader("📝 Sugerencias Personalizadas de Intervención Oportuna:")
        for sugerencia in sugerencias_finales: st.info(sugerencia.replace('|', '** | **'))
        
        st.markdown("---")
        try:
            pdf_data = generar_informe_pdf_fpdf(data_reporte, resultado_final, prob_alto_riesgo, sugerencias_finales, gravedad_anemia)
            st.download_button(label="⬇️ Descargar Informe de Recomendaciones Individual (PDF)", data=pdf_data, file_name=f'informe_riesgo_DNI_{data_reporte["DNI"]}_{datetime.date.today().isoformat()}.pdf', mime='application/pdf', type="secondary")
        except Exception as pdf_error: st.error(f"⚠️ Error al generar el PDF. Detalle: {pdf_error}")
        st.markdown("---")

def vista_monitoreo():
    st.title("📊 Monitoreo y Gestión de Alertas (Supabase)")
    st.markdown("---")
    st.header("1. Casos de Monitoreo Activo (Pendientes y En Seguimiento)")
    
    if get_supabase_client() is None:
        st.error("🛑 La gestión de alertas no está disponible. No se pudo establecer conexión con Supabase. Por favor, revise sus 'secrets' o la clave FALLBACK.")
        return

    df_monitoreo = obtener_alertas_pendientes_o_seguimiento()

    if df_monitoreo.empty:
        st.success("No hay casos de alto riesgo o críticos pendientes de seguimiento activo. ✅")
    else:
        st.info(f"Se encontraron **{len(df_monitoreo)}** casos que requieren acción inmediata o seguimiento activo.")
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)"]
        
        cols_to_display = ['DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION']
        if 'ID_DB' in df_monitoreo.columns:
             cols_to_display.insert(0, 'ID_DB')

        df_display = df_monitoreo[cols_to_display].copy()
        
        edited_df = st.data_editor(
            df_display,
            column_config={
                "Estado": st.column_config.SelectboxColumn("Estado de Gestión", options=opciones_estado, required=True),
                "Sugerencias": st.column_config.TextColumn("Sugerencias", width="large"),
                "ID_GESTION": None,
                "ID_DB": st.column_config.NumberColumn("ID de Registro", disabled=True)
            },
            hide_index=True,
            key="monitoreo_data_editor"
        )

        changes_detected = False
        for index, row in edited_df.iterrows():
            original_row = df_monitoreo.loc[index]
            if row['Estado'] != original_row['Estado']:
                success = actualizar_estado_alerta(row['DNI'], original_row['Fecha Alerta'], row['Estado'])
                if success:
                    st.toast(f"✅ Estado de DNI {row['DNI']} actualizado a '{row['Estado']}'", icon='✅')
                    changes_detected = True
                else:
                    st.toast(f"❌ Error al actualizar estado para DNI {row['DNI']}", icon='❌')
                
        if changes_detected:
            st.rerun()

    st.markdown("---")
    st.header("2. Historial Completo de Registros")

    df_historial = obtener_todos_los_registros()
    
    if not df_historial.empty:
        st.download_button(
            label="⬇️ Descargar Historial Completo (CSV)",
            data=df_historial.to_csv(index=False, sep=';').encode('utf-8'),
            file_name=f'historial_alertas_anemia_{datetime.date.today().isoformat()}.csv',
            mime='text/csv',
        )
        st.dataframe(df_historial)
    else:
        st.info("No hay registros en el historial.")

# ==============================================================================
# 6. VISTA DEL DASHBOARD ESTADÍSTICO
# ==============================================================================

def vista_dashboard():
    st.title("📊 Panel Estadístico de Alertas de Anemia")
    st.markdown("---")
    
    if get_supabase_client() is None:
        st.error("🛑 El dashboard no está disponible. No se pudo establecer conexión con Supabase.")
        return

    df_historial = obtener_todos_los_registros()

    if df_historial.empty:
        st.info("No hay datos de historial disponibles para generar el tablero.")
        if st.session_state.get('supabase_error_historial'):
             st.error(f"❌ Error al consultar el historial de registros (Supabase): {st.session_state.get('supabase_error_historial')}")
        return

    # Preparar datos: Contar por riesgo, región y estado
    df_riesgo = df_historial.groupby('Riesgo').size().reset_index(name='Conteo')
    df_estado = df_historial.groupby('Estado').size().reset_index(name='Conteo')
    
    df_region = df_historial[df_historial['Riesgo'].str.contains('ALTO RIESGO', na=False)].groupby('Region').size().reset_index(name='Casos de Alto Riesgo')
    
    df_historial['Fecha Alerta'] = pd.to_datetime(df_historial['Fecha Alerta'])
    df_tendencia = df_historial.set_index('Fecha Alerta').resample('M').size().reset_index(name='Alertas Registradas')
    
    # --- FILTROS ---
    st.sidebar.header("Filtros del Dashboard")
    regiones_disponibles = sorted(df_historial['Region'].unique())
    if regiones_disponibles:
        filtro_region = st.sidebar.multiselect("Filtrar por Región:", regiones_disponibles, default=regiones_disponibles)
        df_filtrado = df_historial[df_historial['Region'].isin(filtro_region)]
    else:
        df_filtrado = df_historial

    if df_filtrado.empty:
        st.warning("No hay datos para la selección actual de filtros.")
        return

    st.header("1. Visión General del Riesgo")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Riesgo (IA y Clínico)")
        fig_riesgo = px.pie(
            df_riesgo, 
            names='Riesgo', 
            values='Conteo', 
            title='Distribución por Nivel de Riesgo',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_riesgo.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_riesgo, use_container_width=True)

    with col2:
        st.subheader("Estado de Seguimiento de Casos")
        fig_estado = px.bar(
            df_estado,
            y='Conteo', 
            x='Estado', 
            title='Estado de Gestión de Alertas',
            color='Estado',
            color_discrete_map={
                'PENDIENTE (CLÍNICO URGENTE)': 'red',
                'PENDIENTE (IA/VULNERABILIDAD)': 'orange',
                'EN SEGUIMIENTO': 'blue',
                'RESUELTO': 'green',
                'REGISTRADO': 'gray',
                'CERRADO (NO APLICA)': 'purple'
            }
        )
        fig_estado.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_estado, use_container_width=True)

    st.markdown("---")
    st.header("2. Tendencias y Distribución Geográfica")
    
    st.subheader("Tendencia Mensual de Alertas")
    fig_tendencia = px.line(
        df_tendencia,
        x='Fecha Alerta',
        y='Alertas Registradas',
        title='Alertas Registradas por Mes',
        markers=True
    )
    fig_tendencia.update_layout(hovermode="x unified")
    st.plotly_chart(fig_tendencia, use_container_width=True)

    st.subheader("Casos de Alto Riesgo por Región (Top 10)")
    df_region_top = df_region.sort_values(by='Casos de Alto Riesgo', ascending=False).head(10)
    fig_region = px.bar(
        df_region_top,
        y='Region',
        x='Casos de Alto Riesgo',
        orientation='h',
        title='Regiones con Mayor Alto Riesgo',
        color='Casos de Alto Riesgo'
    )
    fig_region.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_region, use_container_width=True)

# ==============================================================================
# 7. CONFIGURACIÓN PRINCIPAL (SIDEBAR Y RUTAS)
# ==============================================================================

def main():
    client = get_supabase_client()
    
    with st.sidebar:
        st.title("🩸 Sistema de Alerta IA")
        st.markdown("---")
        seleccion = st.radio(
            "Ahora la vista:",
            ["Predicción y Reporte", "Monitoreo de Alertas", "Panel de control estadístico"]
        )
        st.markdown("---")
        st.markdown("### Estado del Sistema")
        if MODELO_ML: st.success("✅ Modelo ML Cargado")
        else: st.error("❌ Modelo ML Falló")
        if client: st.success("✅ Supabase Conectado")
        else: st.error("❌ Supabase Desconectado")
        
    if seleccion == "Predicción y Reporte":
        vista_prediccion()
    elif seleccion == "Monitoreo de Alertas":
        vista_monitoreo()
    elif seleccion == "Panel de control estadístico":
        vista_dashboard()

if __name__ == "__main__":
    main()
