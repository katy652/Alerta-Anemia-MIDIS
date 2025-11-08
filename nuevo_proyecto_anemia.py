import streamlit as st
import pandas as pd
import joblib
import unidecode
from supabase import create_client, Client
import datetime
from fpdf import FPDF
import base64
import requests
import io
import json
import re
import os

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELO
# ==============================================================================

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
COLUMNS_FILENAME = "modelo_columns.joblib" 

# ===================================================================
# CONFIGURACIÓN Y CLAVES DE SUPABASE
# ===================================================================

SUPABASE_TABLE = "alertas"

# ===================================================================
# GESTIÓN DE LA BASE DE DATOS (SUPABASE) - FUNCIÓN DE CONEXIÓN ROBUSTA
# ===================================================================
@st.cache_resource
def get_supabase_client():
    """Inicializa y retorna el cliente de Supabase."""
    
    # URL obtenida de la configuración de tu proyecto
    FALLBACK_URL = "https://kwsuszkolbejvliniqgd.supabase.co" 
    FALLBACK_KEY = "TU_CLAVE_API_ANON_AQUI" # <-- REEMPLAZAR AQUÍ CON TU CLAVE REAL

    url, key = None, None

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except KeyError:
        url = FALLBACK_URL
        key = FALLBACK_KEY
        if key == "TU_CLAVE_API_ANON_AQUI":
            st.error("❌ ERROR: La clave FALLBACK de Supabase no fue reemplazada. Funcionalidad DB Deshabilitada.")
            return None

    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"❌ Error al inicializar Supabase: {e}")
        return None

# ===================================================================
# CARGA DE ACTIVOS DE MACHINE LEARNING (SOLO CARGA LOCAL)
# ===================================================================
@st.cache_resource
def load_model_components():
    """Carga los activos de ML directamente desde archivos locales."""
    modelo = None

    # 1️⃣ Cargar columnas
    try:
        model_columns = joblib.load(COLUMNS_FILENAME)
        st.success("✅ Activos de columna cargados exitosamente.")
    except FileNotFoundError:
        st.error(f"❌ CRÍTICO: No se encontró el archivo '{COLUMNS_FILENAME}'. La IA está deshabilitada.")
        return None, None
    except Exception as e:
        st.error(f"❌ ERROR al cargar las columnas: {e}")
        return None, None

    # 2️⃣ Cargar modelo
    # ESTO REQUIERE QUE EL ARCHIVO 'modelo_anemia.joblib' ESTÉ EN EL MISMO DIRECTORIO
    # (Soluciona el error 60 de descarga)
    try:
        model = joblib.load(MODEL_FILENAME)
        st.success("✅ Modelo de IA cargado correctamente desde almacenamiento local.")
        return model, model_columns
    except FileNotFoundError:
        st.error(f"❌ CRÍTICO: No se encontró el archivo '{MODEL_FILENAME}'. La predicción de IA está deshabilitada.")
        return None, model_columns
    except Exception as e:
        st.error(f"❌ ERROR al cargar el modelo local: {e}")
        st.warning("⚠️ La predicción de IA está temporalmente deshabilitada.")
        return None, model_columns


MODELO_ML, MODELO_COLUMNS = load_model_components()

RISK_MAPPING = {0: "BAJO RIESGO", 1: "MEDIO RIESGO", 2: "ALTO RIESGO"}

# ==============================================================================
# 2. LÓGICA DE NEGOCIO Y PREDICCIÓN (Funciones)
# ==============================================================================

def corregir_hemoglobina_por_altitud(hemoglobina_medida, altitud_m):
    """Aplica la corrección de Hemoglobina según la altitud (OMS, 2011)."""
    if altitud_m < 1000:
        correccion = 0.0
    elif altitud_m < 2000:
        correccion = 0.2
    elif altitud_m < 3000:
        correccion = 0.5
    elif altitud_m < 4000:
        correccion = 0.8
    elif altitud_m < 5000:
        correccion = 1.3
    else: # >= 5000
        correccion = 1.9
        
    return hemoglobina_medida - correccion, correccion

def limpiar_texto(texto):
    if pd.isna(texto): return 'desconocido'
    return unidecode.unidecode(str(texto).strip().lower())

def clasificar_anemia_clinica(hemoglobina_g_dL, edad_meses, altitud_m):
    """
    Clasifica la anemia según la Hb, edad (umbrales clínicos estándar) y altitud.
    Retorna la gravedad clínica, el umbral base, la Hb corregida y la corrección aplicada.
    """
    
    # 1. Aplicar Corrección por Altitud
    hb_corregida, correccion = corregir_hemoglobina_por_altitud(hemoglobina_g_dL, altitud_m)
    
    # 2. Definir Umbral por Edad
    umbral = 0
    if edad_meses >= 6 and edad_meses <= 59: umbral = 11.0 # 6 meses a 5 años
    elif edad_meses >= 60 and edad_meses <= 144: umbral = 11.5 # 5 años a 12 años
    else: umbral = 12.0 # Resto
    
    # 3. Clasificar con Hb Corregida
    if hb_corregida < UMBRAL_SEVERA: return "SEVERA", umbral, hb_corregida, correccion
    elif hb_corregida < UMBRAL_MODERADA: return "MODERADA", umbral, hb_corregida, correccion
    elif hb_corregida < umbral: return "LEVE", umbral, hb_corregida, correccion
    else: return "NO ANEMIA", umbral, hb_corregida, correccion

def preprocess_data_for_ml(data_raw, model_columns):
    """Prepara los datos crudos para el modelo de ML (One-Hot Encoding)."""
    data_ml = {'Hemoglobina_g_dL': data_raw['Hemoglobina_g_dL'], 'Edad_meses': data_raw['Edad_meses'], 'Altitud_m': data_raw['Altitud_m'], 'Ingreso_Familiar_Soles': data_raw['Ingreso_Familiar_Soles'], 'Nro_Hijos': data_raw['Nro_Hijos']}
    df_pred = pd.DataFrame([data_ml])
    categorical_cols = ['Sexo', 'Region', 'Area', 'Clima', 'Nivel_Educacion_Madre', 'Programa_QaliWarma', 'Programa_Juntos', 'Programa_VasoLeche', 'Suplemento_Hierro']
    for col in categorical_cols:
        if col in data_raw: df_pred[col] = limpiar_texto(data_raw[col])
        
    df_encoded = pd.get_dummies(df_pred)
    missing_cols = set(model_columns) - set(df_encoded.columns)
    for c in missing_cols: df_encoded[c] = 0
    
    df_final = df_encoded[model_columns]
    df_final = df_final.astype({col: 'float64' for col in df_final.columns})
    return df_final

def predict_risk_ml(data_raw):
    """Realiza la predicción del riesgo usando el modelo de Machine Learning."""
    if MODELO_ML is None or MODELO_COLUMNS is None:
        return 0.5, "RIESGO INDEFINIDO (IA DESHABILITADA)"
    try:
        X_df = preprocess_data_for_ml(data_raw, MODELO_COLUMNS)
        resultado_clase = MODELO_ML.predict(X_df)[0]
        prob_riesgo_array = MODELO_ML.predict_proba(X_df)[0]
        prob_alto_riesgo = prob_riesgo_array[2] 
        resultado_texto = RISK_MAPPING.get(resultado_clase, "RIESGO INDEFINIDO")
        return prob_alto_riesgo, resultado_texto
    except Exception as e:
        st.error(f"Fallo en el motor de IA durante la predicción: {e}")
        return 0.5, "ERROR: Fallo en el motor de IA"

def generar_sugerencias(data, resultado_final, gravedad_anemia):
    """Genera una lista de sugerencias basadas en el diagnóstico y factores de riesgo."""
    sugerencias_raw = []
    
    if gravedad_anemia == 'SEVERA':
        sugerencias_raw.append("🚨🚨 EMERGENCIA SEVERA | Traslado inmediato a Hospital/Centro de Salud de mayor complejidad y posible transfusión.")
    elif gravedad_anemia == 'MODERADA':
        sugerencias_raw.append("⚠️ ATENCIÓN INMEDIATA (Moderada) | Derivación urgente al Puesto de Salud más cercano para evaluación y dosis de ataque de suplemento.")
        
    if not gravedad_anemia in ['SEVERA', 'MODERADA']:
        if resultado_final.startswith("ALTO"):
            sugerencias_raw.append(f"⚠️ Alerta por Vulnerabilidad (IA) | Se requiere seguimiento clínico reforzado y monitoreo por el alto riesgo detectado.")
            if data['Hemoglobina_g_dL'] < UMBRAL_HEMOGLOBINA_ANEMIA:
                sugerencias_raw.append("💊 Anemia Leve Confirmada | Priorizar la entrega y garantizar el consumo diario de suplementos de hierro.")
        
        if data['Altitud_m'] > 2500:
            sugerencias_raw.append("🍲 Riesgo Ambiental (Altura) | Priorizar alimentos con alta absorción de hierro.")
        if data['Ingreso_Familiar_Soles'] < 1000:
            sugerencias_raw.append("💰 Riesgo Socioeconómico | Reforzar la inclusión en Programas Sociales.")
        if data['Nivel_Educacion_Madre'] in ['Primaria', 'Inicial']:
            sugerencias_raw.append("📚 Capacitación | Ofrecer talleres nutricionales dirigidos a la madre/cuidador.")
            
    if resultado_final.startswith("MEDIO"):
        sugerencias_raw.append("✅ Monitoreo Reforzado | Mantener el seguimiento de rutina y reforzar la educación nutricional.")
    elif resultado_final.startswith("BAJO") and not sugerencias_raw:
        sugerencias_raw.append("✅ Control Preventivo | Mantener el seguimiento de rutina y los hábitos saludables.")
            
    if not sugerencias_raw:
        sugerencias_raw.append("✨ Recomendaciones Generales | Asegurar una dieta variada y el consumo de alimentos con vitamina C.")
        
    sugerencias_limpias = []
    for sug in sugerencias_raw:
        sug_stripped = sug.replace('**', '').replace('*', '').replace('<b>', '').replace('</b>', '').strip()
        sugerencias_limpias.append(unidecode.unidecode(sug_stripped))
        
    return list(set(sugerencias_limpias)) 


# ==============================================================================
# 3. GESTIÓN DE LA BASE DE DATOS (SUPABASE) - FUNCIONES CORREGIDAS
# ==============================================================================

def safe_json_to_text_display(json_str):
    if isinstance(json_str, str) and json_str.strip() and json_str.startswith('['):
        try:
            sug_list = json.loads(json_str)
            sug_display = []
            for sug in sug_list:
                sug_markdown = sug.replace('|', ' | ')
                sug_display.append(sug_markdown)
            return "\n".join(sug_display)
        except json.JSONDecodeError:
            return "**ERROR: Datos de sugerencia corruptos.**"
    return "No hay sugerencias registradas."

def rename_and_process_df(response_data):
    """Procesa los datos de respuesta de Supabase a un DataFrame legible."""
    if response_data:
        df = pd.DataFrame(response_data)
        # 🛑 CORRECCIÓN: Se elimina 'id' de la lista de renombrado ya que no existe en la tabla 'alertas'
        df = df.rename(columns={'dni': 'DNI', 'nombre_apellido': 'Nombre', 'edad_meses': 'Edad (meses)', 'hemoglobina_g_dL': 'Hb Inicial', 'riesgo': 'Riesgo', 'fecha_alerta': 'Fecha Alerta', 'estado': 'Estado', 'sugerencias': 'Sugerencias'})
        
        # Agregamos una columna visible para el ID de actualización, usando el DNI
        df['ID_GESTION'] = df['DNI'].astype(str) + '_' + df['Fecha Alerta'].astype(str)
        
        df['Sugerencias'] = df['Sugerencias'].apply(safe_json_to_text_display)
        return df
    return pd.DataFrame()

@st.cache_data
def obtener_alertas_pendientes_o_seguimiento():
    """Obtiene registros marcados para monitoreo activo."""
    supabase = get_supabase_client()
    if not supabase: return pd.DataFrame()

    try:
        # 🛑 CORRECCIÓN: Se elimina el ordenamiento por 'id' que causaba el error
        response = supabase.table(SUPABASE_TABLE).select('*').in_('estado', ['PENDIENTE (CLÍNICO URGENTE)', 'PENDIENTE (IA/VULNERABILIDAD)', 'EN SEGUIMIENTO']).order('fecha_alerta', desc=True).execute()
        return rename_and_process_df(response.data)

    except Exception as e:
        st.error(f"❌ Error al consultar alertas de monitoreo: {e}") 
        return pd.DataFrame()

@st.cache_data
def obtener_todos_los_registros():
    """Obtiene todo el historial de registros."""
    supabase = get_supabase_client()
    if not supabase: return pd.DataFrame()

    try:
        # 🛑 CORRECCIÓN: Se elimina el ordenamiento por 'id'
        response = supabase.table(SUPABASE_TABLE).select('*').order('fecha_alerta', desc=True).execute()
        return rename_and_process_df(response.data)

    except Exception as e:
        st.error(f"❌ Error al consultar el historial de registros: {e}")
        return pd.DataFrame()

def actualizar_estado_alerta(dni, fecha_alerta, nuevo_estado):
    """
    Actualiza el estado de una alerta usando DNI y Fecha de Alerta como clave compuesta.
    """
    supabase = get_supabase_client()
    if not supabase: return False
    try:
        # 🛑 CORRECCIÓN: Se usa DNI y fecha para actualizar el registro.
        supabase.table(SUPABASE_TABLE).update({'estado': nuevo_estado}).eq('dni', dni).eq('fecha_alerta', fecha_alerta).execute()
        obtener_alertas_pendientes_o_seguimiento.clear()
        obtener_todos_los_registros.clear()
        return True
    except Exception as e:
        st.error(f"❌ Error al actualizar estado en Supabase: {e}")
        return False

def registrar_alerta_db(data_alerta):
    """Registra un nuevo caso en la base de datos."""
    supabase = get_supabase_client()
    if not supabase:
        st.error("No se pudo registrar: La conexión a Supabase falló.")
        return False
    try:
        
        if 'SEVERA' in data_alerta['gravedad_anemia'] or 'MODERADA' in data_alerta['gravedad_anemia']: estado = 'PENDIENTE (CLÍNICO URGENTE)'
        elif data_alerta['riesgo'].startswith("ALTO RIESGO") and not data_alerta['riesgo'].startswith("ALTO RIESGO (Alerta Clínica"): estado = 'PENDIENTE (IA/VULNERABILIDAD)'
        else: estado = 'REGISTRADO'
        
        fecha_registro = datetime.date.today().isoformat()

        data = {
            'dni': data_alerta['DNI'],
            'nombre_apellido': data_alerta['Nombre_Apellido'],
            'edad_meses': data_alerta['Edad_meses'],
            'hemoglobina_g_dL': data_alerta['Hemoglobina_g_dL'],
            'riesgo': data_alerta['riesgo'],
            'fecha_alerta': fecha_registro,
            'estado': estado,
            'sugerencias': json.dumps(data_alerta['sugerencias'])
        }

        supabase.table(SUPABASE_TABLE).insert(data).execute()

        obtener_alertas_pendientes_o_seguimiento.clear()
        obtener_todos_los_registros.clear()

        if estado.startswith('PENDIENTE'):
            st.info(f"✅ Caso registrado para **Monitoreo Activo** (Supabase). DNI: **{data_alerta['DNI']}**. Estado: **{estado}**.")
        else:
            st.info(f"✅ Caso registrado para **Control Estadístico** (Supabase). DNI: **{data_alerta['DNI']}**. Estado: **REGISTRADO**.")
        return True
    except Exception as e:
        st.error(f"❌ Error al registrar en Supabase. Mensaje: {e}")
        return False

# ==============================================================================
# 4. GENERACIÓN DE INFORME PDF (Funciones)
# ==============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, unidecode.unidecode('INFORME PERSONALIZADO DE RIESGO DE ANEMIA'), 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, 'Ministerio de Desarrollo e Inclusion Social (MIDIS)', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(165, 42, 42)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(2)

def generar_informe_pdf_fpdf(data, resultado_final, prob_riesgo, sugerencias, gravedad_anemia):
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.chapter_title('I. DATOS DEL CASO')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"DNI del Paciente: {data['DNI']}", 0, 1)
    pdf.cell(0, 5, f"Nombre: {data['Nombre_Apellido']}", 0, 1)
    pdf.cell(0, 5, f"Fecha de Analisis: {datetime.date.today().isoformat()}", 0, 1)
    pdf.ln(5)

    pdf.chapter_title('II. CLASIFICACION DE RIESGO')
    if resultado_final.startswith("ALTO"): pdf.set_text_color(255, 0, 0)
    elif resultado_final.startswith("MEDIO"): pdf.set_text_color(255, 140, 0)
    else: pdf.set_text_color(0, 128, 0)
    resultado_texto = f"RIESGO HÍBRIDO: {unidecode.unidecode(resultado_final)}"
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, resultado_texto, 0, 1)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Gravedad Clínica (Hb Corregida): {gravedad_anemia} ({data['Hemoglobina_g_dL']} g/dL)", 0, 1)
    pdf.cell(0, 5, f"Prob. de Alto Riesgo por IA: {prob_riesgo:.2%}", 0, 1)
    pdf.ln(5)

    pdf.chapter_title('III. PLAN DE INTERVENCION PERSONALIZADO')
    pdf.set_font('Arial', '', 10)
    for sug in sugerencias:
        final_text = sug.replace('|', ' - ').replace('🚨🚨', '[EMERGENCIA]').replace('🔴', '[CRITICO]').replace('⚠️', '[ALERTA]').replace('💊', '[Suplemento]').replace('🍲', '[Dieta]').replace('💰', '[Social]').replace('👶', '[Edad]').replace('✅', '[Ok]').replace('📚', '[Educacion]').replace('✨', '[General]')
        pdf.set_fill_color(240, 240, 240)
        pdf.multi_cell(0, 6, f"- {final_text}", 0, 'L')
        pdf.ln(1)

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, "--- Fin del Informe ---", 0, 1, 'C')

    return bytes(pdf.output(dest='S'))

# ==============================================================================
# 5. VISTAS DE LA APLICACIÓN (STREAMLIT UI)
# ==============================================================================

def vista_prediccion():
    st.title("📝 Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.2 Híbrida con Corrección de Altitud)")
    st.markdown("---")

    if MODELO_COLUMNS is None:
        st.error(f"❌ El formulario está deshabilitado. No se pudo cargar los archivos necesarios. Revise los errores críticos de arriba.")
        return

    if MODELO_ML is None:
        st.warning("⚠️ El motor de Predicción de IA no está disponible. Solo se realizarán la **Clasificación Clínica** y la **Generación de PDF**.")

    # 🛑 LISTA FINAL DE REGIONES DE PERÚ (25 Regiones: 24 Dptos + Callao)
    REGIONES_PERU = [
        "LIMA (Metropolitana y Provincia)", "CALLAO (Provincia Constitucional)", 
        # Costa Norte y Centro
        "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ÁNCASH (Costa)", "ICA", 
        # Sierra/Andes (Alta y Media)
        "PUNO (Sierra Alta)", "HUANCAVELICA (Sierra Alta)", "CUSCO (Andes)", 
        "JUNÍN (Andes)", "AYACUCHO", "APURÍMAC", "HUÁNUCO", "PASCO", 
        "CAJAMARCA", 
        # Sur (Mayormente Sierra y Costa)
        "AREQUIPA", "MOQUEGUA", "TACNA", 
        # Selva (Amazonía)
        "LORETO", "AMAZONAS", "SAN MARTÍN", "UCAYALI", "MADRE DE DIOS", 
        # Otros
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
        col_h, col_e, col_a = st.columns(3)
        with col_h: hemoglobina = st.number_input("Hemoglobina (g/dL) - CRÍTICO", min_value=5.0, max_value=18.0, value=10.5, step=0.1)
        with col_e: edad_meses = st.slider("Edad (meses)", min_value=12, max_value=60, value=36)
        # ⚠️ Nota: El Altitud_m es crucial para la corrección de Hb
        with col_a: altitud = st.number_input("Altitud (metros s.n.m.) - CLAVE", min_value=0, max_value=5000, value=1500, step=10) 
        st.markdown("---")
        
        st.subheader("2. Factores Socioeconómicos y Contextuales")
        col_r, col_c, col_ed = st.columns(3)
        with col_r: region = st.selectbox("Región", options=REGIONES_PERU) # 🛑 Uso de lista completa de regiones
        with col_c: clima = st.selectbox("Clima Predominante", options=['Templado andino', 'Frío andino', 'Cálido seco', 'Otro'])
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
            
            data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'Altitud_m': altitud, 'Sexo': sexo, 'Region': region, 'Area': area, 'Clima': clima, 'Ingreso_Familiar_Soles': ingreso_familiar, 'Nivel_Educacion_Madre': educacion_madre, 'Nro_Hijos': nro_hijos, 'Programa_QaliWarma': qali_warma, 'Programa_Juntos': juntos, 'Programa_VasoLeche': vaso_leche, 'Suplemento_Hierro': suplemento_hierro}

            # 🛑 Llamada a la función corregida: Clasificación Clínica con ajuste por altitud
            gravedad_anemia, umbral_clinico, hb_corregida, correccion_alt = clasificar_anemia_clinica(hemoglobina, edad_meses, altitud)
            prob_alto_riesgo, resultado_ml = predict_risk_ml(data)

            if gravedad_anemia in ['SEVERA', 'MODERADA']:
                resultado_final = f"ALTO RIESGO (Alerta Clínica - {gravedad_anemia})"
            elif resultado_ml.startswith("ALTO RIESGO"):
                resultado_final = f"ALTO RIESGO (Predicción ML - Anemia {gravedad_anemia})"
            else:
                resultado_final = resultado_ml

            sugerencias_finales = generar_sugerencias(data, resultado_final, gravedad_anemia)
            alerta_data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'riesgo': resultado_final, 'gravedad_anemia': gravedad_anemia, 'sugerencias': sugerencias_finales}

            # Intenta registrar en DB
            registrar_alerta_db(alerta_data)

            # Guardar resultados en session_state y recargar
            st.session_state.resultado = resultado_final
            st.session_state.prob_alto_riesgo = prob_alto_riesgo
            st.session_state.gravedad_anemia = gravedad_anemia
            st.session_state.sugerencias_finales = sugerencias_finales
            st.session_state.data_reporte = data
            st.session_state.hb_corregida = hb_corregida
            st.session_state.correccion_alt = correccion_alt
            st.session_state.prediction_done = True
            st.rerun()

    # Mostrar resultados después de la predicción
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
        with col_res2: st.metric(label=f"Corrección por Altitud ({data_reporte['Altitud_m']}m)", value=f"-{correccion_alt:.1f} g/dL")
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
        st.error("🛑 La gestión de alertas no está disponible. No se pudo establecer conexión con Supabase. Por favor, revise sus 'secrets'.")
        return

    df_monitoreo = obtener_alertas_pendientes_o_seguimiento()

    if df_monitoreo.empty:
        st.success("No hay casos de alto riesgo o críticos pendientes de seguimiento activo. ✅")
    else:
        st.info(f"Se encontraron **{len(df_monitoreo)}** casos que requieren acción inmediata o seguimiento activo.")
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)"]
        
        df_display = df_monitoreo[['DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION']].copy()
        
        edited_df = st.data_editor(
            df_display,
            column_config={
                "Estado": st.column_config.SelectboxColumn("Estado de Gestión", options=opciones_estado, required=True), 
                "Sugerencias": st.column_config.TextColumn("Sugerencias", width="large"),
                "ID_GESTION": None # Ocultar la clave compuesta
            },
            hide_index=True, num_rows="fixed", use_container_width=True
        )

        if st.button("Guardar Cambios de Estado", type="primary"):
            cambios_guardados = 0
            for original_row in df_monitoreo.itertuples():
                # Encontrar la fila editada usando el DNI (asumiendo que DNI es único en el dataframe filtrado)
                edited_row = edited_df[edited_df['DNI'] == original_row.DNI].iloc[0]
                
                if original_row.Estado != edited_row['Estado']:
                    # Usamos DNI y Fecha Alerta para la actualización
                    # La fecha de alerta está en el índice 6 de la tupla de nombres generados por .itertuples()
                    if actualizar_estado_alerta(original_row.DNI, original_row._6, edited_row['Estado']): 
                        st.success(f"Estado del DNI **{original_row.DNI}** (Fecha: {original_row._6}) actualizado a **{edited_row['Estado']}**.")
                        cambios_guardados += 1
            if cambios_guardados > 0:
                st.info(f"Se actualizaron {cambios_guardados} registros. Recargando la vista...")
                st.rerun()
            else: st.warning("No se detectaron cambios de estado para guardar.")

        st.markdown("---")
        st.header("2. Reporte Histórico de Registros")
        df_reporte = obtener_todos_los_registros()

        if not df_reporte.empty:
            st.dataframe(df_reporte[['DNI', 'Nombre', 'Edad (meses)', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado']], use_container_width=True, hide_index=True)
            @st.cache_data
            def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
            csv = convert_df_to_csv(df_reporte)
            st.download_button(label="⬇️ Descargar Reporte Completo (CSV)", data=csv, file_name=f'reporte_historico_alertas_{datetime.date.today().isoformat()}.csv', mime='text/csv')
        else: st.info("No hay registros históricos en la base de datos.")


# ==============================================================================
# 6. ESTRUCTURA DE LA APP (NAVEGACIÓN)
# ==============================================================================

st.sidebar.title("🩸 Menú MIDIS Anemia")
st.sidebar.markdown("---")
opcion_seleccionada = st.sidebar.radio(
    "Selecciona una vista:",
    ["📝 Generar Informe (Predicción)", "📊 Monitoreo y Reportes"]
)
st.sidebar.markdown("---")
st.sidebar.info("App Híbrida v2.2 (Clínica + IA)")

if opcion_seleccionada == "📝 Generar Informe (Predicción)":
    vista_prediccion()
elif opcion_seleccionada == "📊 Monitoreo y Reportes":
    vista_monitoreo()
