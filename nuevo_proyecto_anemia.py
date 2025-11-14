import streamlit as st
# Necesitas instalar 'fpdf', 'unidecode', 'pandas', 'plotly', y opcionalmente 'twilio'
# pip install streamlit fpdf unidecode pandas plotly twilio
from fpdf import FPDF
import unidecode
import datetime
import pandas as pd
import plotly.express as px
import random
# Importar la lógica del modelo mock, no el modelo real.
try:
    # Intenta importar Twilio Client, que se usará si las credenciales son reales
    from twilio.rest import Client
    TWILIO_CLIENT_AVAILABLE = True
except ImportError:
    # Si la librería no está instalada, no es un problema para la simulación
    TWILIO_CLIENT_AVAILABLE = False


# ==============================================================================
# 0. CONFIGURACIÓN DE PÁGINA Y VARIABLES GLOBALES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Alerta Anemia MIDIS")

# Simulación de las variables de entorno que proveería el sistema
# Estas variables son NECESARIAS para una conexión real a Firebase
APP_ID = "midis-anemia-app" # Simulación de __app_id
USER_ID = "admin-user-001" # Simulación de auth.currentUser.uid

MODELO_COLUMNS = ['Hemoglobina_g_dL', 'Edad_meses', 'Altitud_m', 'Area_Rural', 'Clima_Frio', 'Clima_Templado', 'Nivel_Educacion_Madre_Sin_Nivel', 'Ingreso_Familiar_Soles', 'Nro_Hijos', 'Programa_QaliWarma_Si', 'Programa_Juntos_Si', 'Programa_VasoLeche_Si', 'Suplemento_Hierro_No']
MODELO_ML = "Mock Model Loaded" # Simula que el modelo ha cargado correctamente


# ------------------------------------------------------------------------------
# FIREBASE CLIENT SIMULADO (Persistencia solo en SESIÓN)
# ------------------------------------------------------------------------------
class FirestoreSessionClient:
    """
    Cliente que simula la interacción con Firebase Firestore.
    Almacena los datos ÚNICAMENTE en la memoria de la sesión (st.session_state).
    Para una persistencia REAL (entre recargas), se necesita una conexión real 
    a Firebase Firestore con las credenciales adecuadas.
    """
    def __init__(self, app_id, user_id):
        self.app_id = app_id
        self.user_id = user_id
        # La colección pública simulada de Firestore: /artifacts/{appId}/public/data/alertas_anemia
        self.collection_path = f"/artifacts/{self.app_id}/public/data/alertas_anemia" 
        
        if 'FIRESTORE_RECORDS' not in st.session_state:
            st.session_state.FIRESTORE_RECORDS = [] 
        if 'MOCK_ID_COUNTER' not in st.session_state:
            st.session_state.MOCK_ID_COUNTER = 1
        self.is_connected = True # Siempre True en la simulación

    def insert(self, data):
        # Simula la inserción en la colección pública
        record = data.copy()
        record['ID_DB'] = st.session_state.MOCK_ID_COUNTER # ID numérico fácil de ver
        record['Fecha Alerta'] = datetime.datetime.now().isoformat()
        # ID_GESTION simula el ID del documento de Firestore
        record['ID_GESTION'] = f"doc_{st.session_state.MOCK_ID_COUNTER}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        record['Usuario_Registro'] = self.user_id # Quién registró
        
        st.session_state.FIRESTORE_RECORDS.append(record)
        st.session_state.MOCK_ID_COUNTER += 1
        return True, record['ID_GESTION']

    def update(self, id_gestion, nuevo_estado):
        # Simula la actualización por el ID_GESTION (ID del documento)
        for i, record in enumerate(st.session_state.FIRESTORE_RECORDS):
            if record.get('ID_GESTION') == id_gestion:
                st.session_state.FIRESTORE_RECORDS[i]['Estado'] = nuevo_estado
                return True
        return False

    def get_record_count(self):
        return len(st.session_state.FIRESTORE_RECORDS)

    def get_all_records(self):
        # Obtiene todos los registros, asegura el tipo y devuelve un DataFrame
        df = pd.DataFrame(st.session_state.FIRESTORE_RECORDS)
        
        if df.empty:
            return pd.DataFrame() 
        
        try:
            if 'Hb Inicial' in df.columns:
                df['Hb Inicial'] = pd.to_numeric(df['Hb Inicial'], errors='coerce')
            if 'ID_DB' in df.columns:
                df['ID_DB'] = pd.to_numeric(df['ID_DB'], errors='coerce', downcast='integer')
            if 'Fecha Alerta' in df.columns:
                df['Fecha Alerta'] = pd.to_datetime(df['Fecha Alerta'], errors='coerce')
            
            for col in ['Estado', 'Riesgo', 'Nombre', 'DNI', 'ID_GESTION']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    
            # Ordenar por ID_DB (descendente)
            if 'ID_DB' in df.columns:
                 return df.sort_values(by='ID_DB', ascending=False)
            
        except Exception as e:
            st.error(f"Error al procesar tipos de columna en Mock DB: {e}")
            return pd.DataFrame() 
            
        return df


# Inicializar el cliente simulado de Firestore
DB_CLIENT = FirestoreSessionClient(app_id=APP_ID, user_id=USER_ID)

# ==============================================================================
# 2. FUNCIONES DE SOPORTE (Altitud, Clima, DB Mock)
# ==============================================================================

def get_altitud_por_region(region):
    # Diccionario de altitudes promedio para la corrección de Hb (Valores simplificados)
    altitudes = {
        "PUNO (Sierra Alta)": 3820,
        "HUANCAVELICA (Sierra Alta)": 3676,
        "PASCO": 4330,
        "JUNÍN (Andes)": 3271,
        "CUSCO (Andes)": 3399,
        "AYACUCHO": 2761,
        "APURÍMAC": 2900,
        "CAJAMARCA": 2750,
        "AREQUIPA": 2335,
        "MOQUEGUA": 1410,
        "TACNA": 562,
        "HUÁNUCO": 1894,
        "ÁNCASH (Costa)": 50,
        "LIMA (Metropolitana y Provincia)": 150,
        "CALLAO (Provincia Constitucional)": 10,
        "PIURA": 30, "LAMBAYEQUE": 50, "LA LIBERTAD": 150, "ICA": 406, "TUMBES": 50,
        "LORETO": 106, "AMAZONAS": 500, "SAN MARTÍN": 500, "UCAYALI": 154, "MADRE DE DIOS": 200,
        "OTRO / NO ESPECIFICADO": 150
    }
    return altitudes.get(region, 150)

def get_clima_por_region(region):
    # Clasificación simplificada de clima para la variable ML
    if 'SIERRA ALTA' in region.upper() or 'PUNO' in region.upper() or 'PASCO' in region.upper() or 'HUANCAVELICA' in region.upper():
        return "FRÍO"
    elif 'ANDES' in region.upper() or 'AYACUCHO' in region.upper() or 'CAJAMARCA' in region.upper():
        return "TEMPLADO"
    elif 'LORETO' in region.upper() or 'UCAYALI' in region.upper() or 'AMAZONAS' in region.upper() or 'MADRE DE DIOS' in region.upper() or 'SAN MARTÍN' in region.upper():
        return "CÁLIDO/HÚMEDO"
    else: # Costa y Lima/Callao
        return "CÁLIDO/SECO"

def registrar_alerta_db(alerta_data):
    # Prepara el objeto para inserción (simulada en la sesión)
    data_to_insert = {
        'DNI': alerta_data['DNI'],
        'Nombre': alerta_data['Nombre_Apellido'],
        'Hb Inicial': alerta_data['Hemoglobina_g_dL'],
        'Riesgo': alerta_data['riesgo'],
        'Gravedad': alerta_data['gravedad_anemia'],
        'Region': alerta_data['Region'],
        'Estado': 'REGISTRADO', # Estado inicial
        'Sugerencias': ' | '.join(alerta_data['sugerencias']),
    }
    
    # 🛑 Uso del cliente simulado de Firestore
    success, new_doc_id = DB_CLIENT.insert(data_to_insert)
    
    if success:
        st.success(f"✅ Caso de {alerta_data['Nombre_Apellido']} registrado en la DB de Sesión con ID de Documento (Mock): {new_doc_id}")
        return True
    else:
        st.warning("⚠️ No se pudo registrar el caso en la DB de Sesión.")
        return False

# ==============================================================================
# 3. FUNCIONES DE CORE LOGIC (Clasificación Clínica e IA)
# ==============================================================================

def clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_m):
    # Factor de corrección por altitud (según CDC/OMS para Hb)
    # Corrección para ALTITUD (suma a Hb) = 0.3 * (Altitud en km)
    correccion_alt = 0.3 * (altitud_m / 1000)
    hb_corregida = hemoglobina + correccion_alt
    
    # Umbrales (Hb corregida, g/dL) para niños 12–59 meses
    umbral_anemia = 11.0 # < 11.0 es anemia
    umbral_moderada = 10.0 # < 10.0 es moderada
    umbral_severa = 7.0 # < 7.0 es severa
    
    # Gravedad
    if hb_corregida < umbral_severa:
        gravedad = "SEVERA"
    elif hb_corregida < umbral_moderada:
        gravedad = "MODERADA"
    elif hb_corregida < umbral_anemia:
        gravedad = "LEVE"
    else:
        gravedad = "NO ANÉMICO"
        
    return gravedad, umbral_anemia, hb_corregida, correccion_alt

def predict_risk_ml(data):
    # --- MOCK / SIMULACIÓN DE MODELO ML ---
    gravedad_anemia, _, _, _ = clasificar_anemia_clinica(data['Hemoglobina_g_dL'], data['Edad_meses'], data['Altitud_m'])
    
    prob_base = 0.1 # Riesgo inicial
    
    # Factores de aumento de riesgo (por IA simulada)
    if data['Area'] == 'Rural': prob_base += 0.15
    if data['Nivel_Educacion_Madre'] in ['Sin Nivel', 'Inicial', 'Primaria']: prob_base += 0.2
    if data['Ingreso_Familiar_Soles'] < 1000: prob_base += 0.25
    if data['Nro_Hijos'] >= 4: prob_base += 0.1
    if data['Suplemento_Hierro'] == 'No': prob_base += 0.15
    
    # Ajuste por gravedad clínica (dominante en el sistema híbrido)
    if gravedad_anemia == 'SEVERA':
        prob_base = 0.99
    elif gravedad_anemia == 'MODERADA':
        prob_base = max(prob_base, 0.75)
    elif gravedad_anemia == 'LEVE':
        prob_base = max(prob_base, 0.45)
        
    prob_riesgo = min(0.99, prob_base + random.uniform(-0.05, 0.05))
    
    if prob_riesgo >= 0.7:
        resultado_ml = "ALTO RIESGO (Predicción ML)"
    elif prob_riesgo >= 0.4:
        resultado_ml = "MEDIO RIESGO (Predicción ML)"
    else:
        resultado_ml = "BAJO RIESGO (Predicción ML)"
        
    return prob_riesgo, resultado_ml

def generar_sugerencias(data, resultado_final, gravedad_anemia):
    sugerencias = []
    
    # 1. Sugerencias Clínicas (Prioridad Alta)
    if gravedad_anemia == 'SEVERA':
        sugerencias.append("🚨🚨 Requerimiento Inmediato: Hospitalización y Transfusión de Sangre si la indicación clínica lo amerita. Contacto Urgente con UCI Pediátrica. | CRÍTICO | Atención Hospitalaria")
    elif gravedad_anemia == 'MODERADA':
        sugerencias.append("🔴 Seguimiento Clínico Urgente: Dosis terapéutica de Hierro por 6 meses y reevaluación mensual de Hemoglobina. Consulta con Hematología. | CRÍTICO | Suplementación Reforzada")
    elif gravedad_anemia == 'LEVE':
        sugerencias.append("⚠️ Suplementación Inmediata: Dosis profiláctica o terapéutica inicial de Hierro por 4 meses. Control en 30 días. | ALERTA | Suplementación")
    else:
        sugerencias.append("✅ Vigilancia Activa: El valor corregido de Hb es óptimo. Continuar con chequeos regulares y prevención primaria. | Ok | Preventivo")

    # 2. Sugerencias de Suplementación y Dieta
    if data['Suplemento_Hierro'] == 'No':
        sugerencias.append("💊 Suplementación: Iniciar o asegurar la adherencia al suplemento de Hierro (gotas/jarabe) según la edad (MINSA). | Suplemento")
    if data['Edad_meses'] < 24:
        sugerencias.append("👶 Edad Crítica: Reforzar la alimentación complementaria rica en hierro hemo (sangrecita, hígado, bazo) debido a la edad vulnerable (6 a 24 meses). | Dieta | Edad")
        
    sugerencias.append("🍲 Nutrición: Incluir alimentos fortificados y menús ricos en hierro y vitamina C (para absorción). Énfasis en proteínas de origen animal. | Dieta")

    # 3. Sugerencias Socioeconómicas/Contextuales (IA)
    if data['Ingreso_Familiar_Soles'] < 1000:
        sugerencias.append("💰 Apoyo Social: Evaluar la elegibilidad para programas de apoyo económico (Juntos) o alimentario (Vaso de Leche, Qali Warma) si no está inscrito. | Social | Económico")
        
    if data['Area'] == 'Rural':
        sugerencias.append("📚 Educación: Sesiones educativas sobre preparación de alimentos ricos en hierro, higiene y desparasitación adaptadas al contexto rural. | Educación | Contextual")
        
    if data['Nivel_Educacion_Madre'] in ['Primaria', 'Sin Nivel']:
        sugerencias.append("📚 Intervención: Materiales educativos con lenguaje simple y demostraciones prácticas de cocina/higiene. | Educación | Vulnerabilidad")
        
    # 4. Sugerencias Geográficas
    if data['Clima'] == 'FRÍO':
        sugerencias.append("✨ Clima Frío: Reforzar la vigilancia de infecciones respiratorias agudas (IRAs), ya que el frío aumenta el gasto energético y el riesgo nutricional. | General | Contextual")
        
    sugerencias.insert(0, f"Diagnóstico Híbrido: {unidecode.unidecode(resultado_final)}")
    
    return sugerencias

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
        self.cell(0, 10, unidecode.unidecode(title), 0, 1, 'L')
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
    pdf.cell(0, 5, f"Celular: {data['Celular']}", 0, 1) 
    pdf.cell(0, 5, f"Fecha de Analisis: {datetime.date.today().isoformat()}", 0, 1)
    pdf.ln(5)

    pdf.chapter_title('II. CLASIFICACION DE RIESGO')
    if resultado_final.startswith("ALTO"): pdf.set_text_color(255, 0, 0)
    elif resultado_final.startswith("MEDIO"): pdf.set_text_color(255, 140, 0)
    else: pdf.set_text_color(0, 128, 0)
    # unidecode se usa para evitar problemas con tildes en fpdf
    resultado_texto = f"RIESGO HÍBRIDO: {unidecode.unidecode(resultado_final)}"
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, resultado_texto, 0, 1)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Gravedad Clinica (Hb Corregida): {unidecode.unidecode(gravedad_anemia)} ({data['Hemoglobina_g_dL']} g/dL)", 0, 1)
    pdf.cell(0, 5, f"Prob. de Alto Riesgo por IA: {prob_riesgo:.2%}", 0, 1)
    pdf.ln(5)

    pdf.chapter_title('III. PLAN DE INTERVENCION PERSONALIZADO')
    pdf.set_font('Arial', '', 10)
    for sug in sugerencias:
        # Reemplazar íconos por texto para compatibilidad con fpdf
        final_text = sug.replace('|', ' - ').replace('🚨🚨', '[EMERGENCIA]').replace('🔴', '[CRITICO]').replace('⚠️', '[ALERTA]').replace('💊', '[Suplemento]').replace('🍲', '[Dieta]').replace('💰', '[Social]').replace('👶', '[Edad]').replace('✅', '[Ok]').replace('📚', '[Educacion]').replace('✨', '[General]')
        final_text = unidecode.unidecode(final_text) # Aplicar unidecode al texto final
        pdf.set_fill_color(240, 240, 240)
        pdf.multi_cell(0, 6, f"- {final_text}", 0, 'L')
        pdf.ln(1)

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, "--- Fin del Informe ---", 0, 1, 'C')

    # Devolver bytes del PDF
    return bytes(pdf.output(dest='S'))

# ==============================================================================
# 5. INTEGRACIÓN DE ALERTA POR SMS (TWILIO REAL O SIMULADO)
# ==============================================================================

def enviar_alerta_sms_twilio(celular, nombre, dni, riesgo, gravedad):
    """
    Función que gestiona el envío de una alerta por SMS (Simulada o Real).
    """
    # 🛑 INSTRUCCIONES: Reemplace estas variables con sus claves reales de Twilio.
    # Si las deja con los valores 'ACxxx', se ejecutará la SIMULACIÓN.
    ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    AUTH_TOKEN = "your_auth_token"
    TWILIO_NUMBER = "+15017122661"  # Tu número Twilio asignado

    mensaje = f"ALERTA MIDIS: Caso {nombre} (DNI {dni}) clasificado como {riesgo} y Gravedad {gravedad}. REQUIERE ACCIÓN URGENTE. Reporte en PDF adjunto."
    
    # -------------------------------------------------------------------------
    # LÓGICA DE ENVÍO REAL vs. SIMULACIÓN
    # -------------------------------------------------------------------------
    
    if ACCOUNT_SID.startswith("ACx") or not TWILIO_CLIENT_AVAILABLE:
        # Simulación si no hay credenciales reales o si Twilio no está instalado
        st.info(f"📲 Alerta SMS **SIMULADA** enviada al número: {celular}. \n\n*Mensaje:* {mensaje}")
        if not TWILIO_CLIENT_AVAILABLE:
            st.warning("⚠️ Twilio no está instalado (`pip install twilio`). Solo se puede realizar la simulación.")
        return True
    
    # -------------------------------------------------------------------------
    # LÓGICA DE ENVÍO REAL 
    # -------------------------------------------------------------------------
    else:
        try:
            client = Client(ACCOUNT_SID, AUTH_TOKEN)
            client.messages.create(
                to=celular,
                from_=TWILIO_NUMBER,
                body=mensaje
            )
            st.success(f"✅ Alerta SMS **REAL** enviada con éxito a {celular}.")
            return True
        except Exception as e:
            st.error(f"❌ ERROR: No se pudo enviar el SMS real. Detalle: {e}")
            return False


# ==============================================================================
# 6. VISTAS DE LA APLICACIÓN (STREAMLIT UI)
# ==============================================================================

def vista_prediccion():
    st.title("📝 Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.5 Altitud y Clima Automatizados)")
    st.markdown("---")

    if MODELO_COLUMNS is None:
        st.error(f"❌ El formulario está deshabilitado. No se pudo cargar los archivos necesarios. Revise los errores críticos de arriba.")
        return

    # Mensaje de advertencia si la IA no carga
    if MODELO_ML is None:
        st.warning("⚠️ El motor de Predicción de IA no está disponible. Solo se realizarán la **Clasificación Clínica** y la **Generación de PDF**.")

    # 🛑 LISTA FINAL DE REGIONES DE PERÚ (25 Regiones: 24 Dptos + Callao)
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
        col_dni, col_nombre, col_celular = st.columns(3)
        with col_dni: dni = st.text_input("DNI del Paciente", max_chars=8, placeholder="Solo 8 dígitos")
        with col_nombre: nombre = st.text_input("Nombre y Apellido", placeholder="Ej: Ana Torres")
        with col_celular: celular = st.text_input("Celular de Contacto (Ej: +519XXXXXXXX)", max_chars=15, placeholder="+51 9XXXXXXXX")
        st.markdown("---")
        
        st.subheader("1. Factores Clínicos y Demográficos Clave")
        col_h, col_e, col_r = st.columns(3)
        with col_h: hemoglobina = st.number_input("Hemoglobina (g/dL) - CRÍTICO", min_value=5.0, max_value=18.0, value=10.5, step=0.1)
        with col_e: edad_meses = st.slider("Edad (meses)", min_value=12, max_value=60, value=36)
        with col_r: region = st.selectbox("Región (Define Altitud y Clima)", options=REGIONES_PERU)
        
        # 🛑 Altitud se calcula automáticamente
        altitud_calculada = get_altitud_por_region(region)
        st.info(f"📍 Altitud asignada automáticamente para **{region}**: **{altitud_calculada} msnm** (Usada para la corrección de Hemoglobina).")
        st.markdown("---")
        
        st.subheader("2. Factores Socioeconómicos y Contextuales")
        
        # 🛑 Clima se calcula automáticamente
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
        
        predict_button = st.form_submit_button("GENERAR INFORME PERSONALIZADO, REGISTRAR CASO Y ENVIAR ALERTA", type="primary", use_container_width=True)
        st.markdown("---")

        if predict_button:
            if not dni or len(dni) != 8: st.error("Por favor, ingrese un DNI válido de 8 dígitos."); return
            if not nombre: st.error("Por favor, ingrese un nombre."); return
            if not celular: st.error("Por favor, ingrese un número de celular de contacto (ej: +519XXXXXXXX)."); return
            
            # Altitud y Clima usan los valores calculados/asignados
            data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'Altitud_m': altitud_calculada, 'Sexo': sexo, 'Region': region, 'Area': area, 'Clima': clima, 'Ingreso_Familiar_Soles': ingreso_familiar, 'Nivel_Educacion_Madre': educacion_madre, 'Nro_Hijos': nro_hijos, 'Programa_QaliWarma': qali_warma, 'Programa_Juntos': juntos, 'Programa_VasoLeche': vaso_leche, 'Suplemento_Hierro': suplemento_hierro, 'Celular': celular}

            # Clasificación Clínica con ajuste por altitud automática
            gravedad_anemia, umbral_clinico, hb_corregida, correccion_alt = clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_calculada)
            
            # 🛑 Ejecutar el Mock de IA
            prob_alto_riesgo, resultado_ml = predict_risk_ml(data)

            # Lógica Híbrida de Riesgo
            if gravedad_anemia in ['SEVERA', 'MODERADA']:
                resultado_final = f"ALTO RIESGO (Alerta Clínica - {gravedad_anemia})"
            elif resultado_ml.startswith("ALTO RIESGO"):
                resultado_final = f"ALTO RIESGO (Predicción ML - Anemia {gravedad_anemia})"
            else:
                resultado_final = resultado_ml

            sugerencias_finales = generar_sugerencias(data, resultado_final, gravedad_anemia)
            
            # Pasamos la Region para que se guarde en la DB
            alerta_data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'riesgo': resultado_final, 'gravedad_anemia': gravedad_anemia, 'sugerencias': sugerencias_finales, 'Region': region}

            # Intenta registrar en DB (Mock persistente de SESIÓN)
            registrar_alerta_db(alerta_data)
            
            # Intenta enviar alerta por celular
            enviar_alerta_sms_twilio(celular, nombre, dni, resultado_final, gravedad_anemia)

            # Guardar resultados en session_state y recargar
            st.session_state.resultado = resultado_final
            st.session_state.prob_alto_riesgo = prob_alto_riesgo
            st.session_state.gravedad_anemia = gravedad_anemia
            st.session_state.sugerencias_finales = sugerencias_finales
            st.session_state.data_reporte = data 
            st.session_state.hb_corregida = hb_corregida
            st.session_state.correccion_alt = correccion_alt
            st.session_state.prediction_done = True
            # No usamos st.rerun() aquí.

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
        
        with col_res2: st.metric(label=f"Corrección por Altitud ({data_reporte['Altitud_m']}m)", value=f"+{correccion_alt:.1f} g/dL")
        
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

# ==============================================================================
# 7. VISTAS DE MONITOREO Y DASHBOARD
# ==============================================================================

def obtener_alertas_pendientes_o_seguimiento():
    # Consulta (simulada) para obtener registros activos
    df = DB_CLIENT.get_all_records()
    
    if df.empty:
        return pd.DataFrame()
        
    df_filtered = df[~df['Estado'].isin(['RESUELTO', 'CERRADO (NO APLICA)'])].copy()
        
    if not df_filtered.empty:
        # CORRECCIÓN: Usar 'ID_DB' si existe
        sort_col = 'ID_DB' if 'ID_DB' in df_filtered.columns else df_filtered.index.name if df_filtered.index.name else df_filtered.columns[0]
        # Aseguramos que 'ID_DB' es la clave de ordenamiento
        try:
            return df_filtered.sort_values(by='ID_DB', ascending=True).reset_index(drop=True)
        except KeyError:
            st.error("Error de datos: No se encontró la columna 'ID_DB' para ordenar alertas activas.")
            return df_filtered.reset_index(drop=True)
    return pd.DataFrame()

def actualizar_estado_alerta(id_gestion, nuevo_estado):
    # Simula la actualización en el cliente de sesión (que se comporta como Firestore)
    return DB_CLIENT.update(id_gestion, nuevo_estado)

def obtener_todos_los_registros():
    # Retorna los registros ordenados descendentemente por ID_DB (más nuevos primero)
    return DB_CLIENT.get_all_records()

def vista_monitoreo():
    st.title("📊 Monitoreo y Gestión de Alertas")
    st.caption(f"📍 **Colección Firestore Simulada:** `{DB_CLIENT.collection_path}`")
    st.info("🚨 **ATENCIÓN:** Los datos se guardan solo **durante su sesión** (persistencia de memoria). Para guardado PERMANENTE, migrar a React/Angular con Firebase Firestore real.")
    st.markdown("---")
    st.header("1. Casos de Monitoreo Activo (Pendientes y En Seguimiento)")
    
    if not DB_CLIENT.is_connected:
        st.error("🛑 La gestión de alertas no está disponible. No se pudo establecer conexión con el cliente de base de datos.")
        return

    df_monitoreo = obtener_alertas_pendientes_o_seguimiento()

    if df_monitoreo.empty:
        st.success("No hay casos de alto riesgo o críticos pendientes de seguimiento activo. ✅")
    else:
        st.info(f"Se encontraron **{len(df_monitoreo)}** casos que requieren acción inmediata o seguimiento activo.")
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)", "REGISTRADO"]
        
        # Columnas a mostrar en el editor
        cols_to_display = ['ID_DB', 'DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION', 'Usuario_Registro']
        cols_to_display = [col for col in cols_to_display if col in df_monitoreo.columns]
        
        df_display = df_monitoreo[cols_to_display].copy()
        
        # Ocultar ID_GESTION y Usuario_Registro
        column_config = {
            "Estado": st.column_config.SelectboxColumn("Estado de Gestión", options=opciones_estado, required=True),
            "Sugerencias": st.column_config.TextColumn("Sugerencias", width="large"),
            "ID_GESTION": None, 
            "Usuario_Registro": None,
            "ID_DB": st.column_config.NumberColumn("ID de Registro", disabled=True),
            "Fecha Alerta": st.column_config.DatetimeColumn("Fecha Alerta", format="YYYY-MM-DD HH:mm:ss", disabled=True),
            "Hb Inicial": st.column_config.NumberColumn("Hb Inicial (g/dL)", format="%.1f", disabled=True),
            "DNI": st.column_config.TextColumn("DNI", disabled=True),
            "Nombre": st.column_config.TextColumn("Nombre", disabled=True),
            "Riesgo": st.column_config.TextColumn("Riesgo", disabled=True),
        }
        
        try:
            edited_df = st.data_editor(
                df_display,
                column_config=column_config,
                hide_index=True,
                key="monitoreo_data_editor"
            )

            # Lógica de guardado
            changes_detected = False
            if not df_monitoreo.empty:
                # Iterar sobre las filas editadas
                for index, edited_row in edited_df.iterrows():
                    # Usar el ID_GESTION de la fila editada (que no se modifica en el editor)
                    id_gestion = edited_row['ID_GESTION'] 
                    
                    # Buscar la fila original usando el ID_GESTION
                    original_row = df_monitoreo[df_monitoreo['ID_GESTION'] == id_gestion].iloc[0]

                    if edited_row['Estado'] != original_row['Estado']:
                        success = actualizar_estado_alerta(id_gestion, edited_row['Estado'])
                        if success:
                            st.toast(f"✅ Estado de DNI {edited_row['DNI']} actualizado a '{edited_row['Estado']}'", icon='✅')
                            changes_detected = True
                        else:
                            st.toast(f"❌ Error al actualizar estado para DNI {edited_row['DNI']}", icon='❌')
                            
            if changes_detected:
                # Recargar datos después de la actualización exitosa
                st.rerun()
                
        except Exception as e:
            st.error(f"Error en el editor de datos (st.data_editor). Detalle: {e}")
            st.dataframe(df_display) # Mostrar el DataFrame crudo para debug.


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

def vista_dashboard():
    st.title("📊 Panel Estadístico de Alertas")
    st.caption(f"Datos de la Colección Simulada: `{DB_CLIENT.collection_path}`")
    st.markdown("---")
    
    if not DB_CLIENT.is_connected:
        st.error("🛑 El dashboard no está disponible. No se pudo establecer conexión con el cliente de base de datos.")
        return

    df_historial = obtener_todos_los_registros()

    if df_historial.empty:
        st.info("No hay datos de historial disponibles para generar el tablero.")
        return

    # Preparar datos: Contar por riesgo, región y estado
    df_riesgo = df_historial.groupby('Riesgo').size().reset_index(name='Conteo')
    df_estado = df_historial.groupby('Estado').size().reset_index(name='Conteo')
    
    # Filtrar solo casos de ALTO RIESGO para análisis geográfico
    df_region = df_historial[df_historial['Riesgo'].str.contains('ALTO RIESGO', na=False)].groupby('Region').size().reset_index(name='Casos de Alto Riesgo')
    
    # Asegurarse de que las fechas sean datetime para series temporales
    df_historial['Fecha Alerta'] = pd.to_datetime(df_historial['Fecha Alerta'])
    
    # Agrupamos por mes para la tendencia
    if not df_historial.empty:
        # Usamos el set_index solo si el DataFrame no está vacío
        df_tendencia = df_historial.set_index('Fecha Alerta').resample('M').size().reset_index(name='Alertas Registradas')
    else:
        df_tendencia = pd.DataFrame(columns=['Fecha Alerta', 'Alertas Registradas'])


    # --- FILTROS ---
    st.sidebar.header("Filtros del Dashboard")
    regiones_disponibles = sorted(df_historial['Region'].unique())
    # Usar el filtro solo si hay regiones disponibles
    if regiones_disponibles:
        filtro_region = st.sidebar.multiselect("Filtrar por Región:", regiones_disponibles, default=regiones_disponibles)
        df_filtrado = df_historial[df_historial['Region'].isin(filtro_region)]
    else:
        df_filtrado = df_historial

    if df_filtrado.empty:
        st.warning("No hay datos para la selección actual de filtros.")
        return

    st.header("1. Visión General del Riesgo")
    
    # 1.1 Gráfico de Distribución de Riesgo (Columna 1)
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

    # 1.2 Gráfico de Casos por Estado de Gestión (Columna 2)
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
    
    # 2.1 Gráfico de Tendencia Mensual (Ancho Completo)
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

    # 2.2 Gráfico de Casos de Alto Riesgo por Región (Ancho Completo)
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
    fig_region.update_yaxes(autorange="reversed") # Para que el mayor esté arriba
    st.plotly_chart(fig_region, use_container_width=True)


# ==============================================================================
# 8. CONFIGURACIÓN PRINCIPAL (SIDEBAR Y RUTAS)
# ==============================================================================

def main():
    with st.sidebar:
        st.title("🩸 Sistema de Alerta IA")
        st.markdown("---")
        seleccion = st.radio(
            "Ahora la vista:",
            ["Predicción y Reporte", "Monitoreo de Alertas", "Panel de control estadístico"]
        )
        st.markdown("---")
        # Mostrar el estado del modelo y Supabase en la barra lateral
        st.markdown("### Estado del Sistema")
        if MODELO_ML: st.success("✅ Modelo ML Cargado (Lógica Simulada Activa)")
        else: st.error("❌ Modelo ML Falló")
        
        # Conexión a DB (simulada)
        if DB_CLIENT.is_connected: 
            st.success("✅ DB Conectada (Mock de Firestore)")
            st.warning("⚠️ Los datos **NO PERSISTEN** si recarga la página. Lea la nota importante.")
        else: st.error("❌ DB Desconectada")
        
        # Muestra el contador de registros
        num_registros = DB_CLIENT.get_record_count()
        st.info(f"💾 Alertas en Sesión: **{num_registros}**")

        st.markdown("---")
        # Muestra la advertencia si Twilio no está disponible (para el envío real)
        st.markdown("### Estado de SMS")
        if not TWILIO_CLIENT_AVAILABLE:
            st.warning("⚠️ Módulo Twilio NO detectado. SMS solo en MODO SIMULACIÓN.")
        else:
             st.info("✅ Módulo Twilio detectado.")
        
    if seleccion == "Predicción y Reporte":
        vista_prediccion()
    elif seleccion == "Monitoreo de Alertas":
        vista_monitoreo()
    elif seleccion == "Panel de control estadístico":
        vista_dashboard()

if __name__ == "__main__":
    main()
