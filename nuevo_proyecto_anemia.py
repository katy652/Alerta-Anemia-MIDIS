# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from fpdf import FPDF
from unidecode import unidecode
import plotly.express as px

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y CARGA DE RECURSOS (Mocks para Entorno sin Archivos)
# ==============================================================================

# 🛑 SIMULACIÓN DE CARGA DE MODELO Y COLUMNAS
try:
    # Simula la carga de un modelo ML y sus columnas de características
    MODELO_ML = True # Se simula que el modelo está cargado (cambiar a None si falla)
    MODELO_COLUMNS = [
        'Hemoglobina_g_dL', 'Edad_meses', 'Altitud_m', 'Sexo', 'Area', 'Ingreso_Familiar_Soles',
        'Nivel_Educacion_Madre', 'Nro_Hijos', 'Programa_QaliWarma', 'Programa_Juntos',
        'Programa_VasoLeche', 'Suplemento_Hierro', 'Clima_Costa', 'Clima_Sierra', 'Clima_Selva'
    ]
except Exception as e:
    st.error(f"Error crítico al simular carga de recursos: {e}")
    MODELO_ML = None
    MODELO_COLUMNS = None


# 🛑 SIMULACIÓN DE CONEXIÓN A SUPABASE (In-memory Mock Database)
if 'mock_db_df' not in st.session_state:
    st.session_state.mock_db_df = pd.DataFrame(columns=['ID_DB', 'DNI', 'Nombre', 'Hb Inicial', 'Edad_meses', 'Riesgo', 'Gravedad', 'Fecha Alerta', 'Estado', 'Sugerencias', 'Region'])
    st.session_state.mock_db_df['ID_DB'] = st.session_state.mock_db_df['ID_DB'].astype('int')
    st.session_state.mock_db_df['Estado'] = st.session_state.mock_db_df['Estado'].astype('object')
    st.session_state.mock_db_df['Fecha Alerta'] = pd.to_datetime(st.session_state.mock_db_df['Fecha Alerta'])

def get_supabase_client():
    """Simula la conexión al cliente de Supabase. Siempre retorna True para el mock."""
    return True

def registrar_alerta_db(data):
    """Registra el caso en la base de datos simulada."""
    try:
        df = st.session_state.mock_db_df
        new_id = df['ID_DB'].max() + 1 if not df.empty else 1
        
        # Determinar el estado inicial
        initial_state = 'REGISTRADO'
        if data['gravedad_anemia'] in ['SEVERA', 'MODERADA']:
            initial_state = 'PENDIENTE (CLÍNICO URGENTE)'
        elif data['riesgo'].startswith('ALTO RIESGO'):
             initial_state = 'PENDIENTE (IA/VULNERABILIDAD)'

        new_row = {
            'ID_DB': new_id,
            'DNI': data['DNI'],
            'Nombre': data['Nombre_Apellido'],
            'Hb Inicial': data['Hemoglobina_g_dL'],
            'Edad_meses': data['Edad_meses'],
            'Riesgo': data['riesgo'],
            'Gravedad': data['gravedad_anemia'],
            'Fecha Alerta': datetime.date.today(),
            'Estado': initial_state,
            'Sugerencias': "\n".join(data['sugerencias']),
            'Region': data['Region']
        }
        
        # Prevenir duplicados basados en DNI y fecha
        is_duplicate = ((df['DNI'] == new_row['DNI']) & (df['Fecha Alerta'] == new_row['Fecha Alerta'])).any()
        if is_duplicate:
            st.warning(f"El DNI {new_row['DNI']} ya fue registrado hoy. No se registrará duplicado.")
            return True

        st.session_state.mock_db_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.toast(f"✅ Caso DNI {data['DNI']} registrado en la base de datos simulada (ID: {new_id}).", icon='💾')
        return True
    except Exception as e:
        st.session_state['supabase_error_historial'] = str(e)
        st.error(f"Error al registrar caso en mock DB: {e}")
        return False

def obtener_alertas_pendientes_o_seguimiento():
    """Obtiene registros con estado PENDIENTE o EN SEGUIMIENTO."""
    df = st.session_state.mock_db_df.copy()
    if df.empty:
        return df

    # Crear una ID compuesta para edición (DNI_Fecha)
    df['Fecha Alerta Str'] = df['Fecha Alerta'].dt.strftime('%Y-%m-%d')
    df['ID_GESTION'] = df['DNI'].astype(str) + '_' + df['Fecha Alerta Str']

    # Filtrar por estados activos
    estados_activos = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO"]
    df_filtered = df[df['Estado'].isin(estados_activos)].sort_values(by='Fecha Alerta', ascending=False)
    
    return df_filtered.rename(columns={'Nombre': 'Nombre', 'Riesgo': 'Riesgo', 'Hb Inicial': 'Hb Inicial', 'Fecha Alerta Str': 'Fecha Alerta'})

def actualizar_estado_alerta(dni, fecha_alerta, nuevo_estado):
    """Actualiza el estado de una alerta en la base de datos simulada."""
    try:
        df = st.session_state.mock_db_df
        # Asegurarse de que la columna de fecha sea compatible para la comparación
        fecha_alerta = pd.to_datetime(fecha_alerta).date()
        
        match_index = df[(df['DNI'] == dni) & (df['Fecha Alerta'].dt.date == fecha_alerta)].index
        
        if not match_index.empty:
            st.session_state.mock_db_df.loc[match_index, 'Estado'] = nuevo_estado
            return True
        return False
    except Exception:
        return False

def obtener_todos_los_registros():
    """Obtiene todos los registros de la base de datos simulada."""
    df = st.session_state.mock_db_df.copy()
    if df.empty:
        return df
    
    df['Fecha Alerta'] = df['Fecha Alerta'].dt.strftime('%Y-%m-%d')
    return df.sort_values(by='ID_DB', ascending=False)


# ==============================================================================
# 2. FUNCIONES DE UTILDAD (Clasificación, Altitud, Clima)
# ==============================================================================

def get_altitud_por_region(region):
    """Asigna una altitud representativa (en msnm) por región para corrección de Hb."""
    # Valores representativos para fines de simulación
    altitudes = {
        "LIMA (Metropolitana y Provincia)": 160, "CALLAO (Provincia Constitucional)": 50,
        "PIURA": 80, "LAMBAYEQUE": 100, "LA LIBERTAD": 150, "ICA": 400, "TUMBES": 50, "ÁNCASH (Costa)": 150,
        "HUÁNUCO": 1900, "JUNÍN (Andes)": 3200, "CUSCO (Andes)": 3400, "AYACUCHO": 2700, "APURÍMAC": 2300,
        "CAJAMARCA": 2600, "AREQUIPA": 2300, "MOQUEGUA": 1400, "TACNA": 560,
        "PUNO (Sierra Alta)": 3800, "HUANCAVELICA (Sierra Alta)": 3600, "PASCO": 4300,
        "LORETO": 100, "AMAZONAS": 400, "SAN MARTÍN": 300, "UCAYALI": 150, "MADRE DE DIOS": 200,
        "OTRO / NO ESPECIFICADO": 500 # Default para una corrección mínima
    }
    return altitudes.get(region, 500)

def get_clima_por_region(region):
    """Asigna el clima predominante por región para fines de la IA."""
    clima_map = {
        "COSTA": ["LIMA (Metropolitana y Provincia)", "CALLAO (Provincia Constitucional)", "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ICA", "TUMBES", "ÁNCASH (Costa)", "MOQUEGUA", "TACNA"],
        "SIERRA": ["HUÁNUCO", "JUNÍN (Andes)", "CUSCO (Andes)", "AYACUCHO", "APURÍMAC", "CAJAMARCA", "AREQUIPA", "PUNO (Sierra Alta)", "HUANCAVELICA (Sierra Alta)", "PASCO"],
        "SELVA": ["LORETO", "AMAZONAS", "SAN MARTÍN", "UCAYALI", "MADRE DE DIOS"]
    }
    
    for clima, regiones in clima_map.items():
        if region in regiones:
            return clima
    return "OTRO"

def clasificar_anemia_clinica(hemoglobina_medida, edad_meses, altitud_m):
    """
    Clasifica la anemia basada en el protocolo clínico de la OMS, ajustando por altitud.
    
    Parámetros:
    - hemoglobina_medida (float): Valor de Hb medida en g/dL.
    - edad_meses (int): Edad del niño en meses.
    - altitud_m (int): Altitud en metros sobre el nivel del mar (msnm).
    
    Retorna: gravedad (str), umbral_normal (float), hb_corregida (float), correccion (float)
    """
    
    # 1. Corrección por Altitud (OMS / CDC)
    # Se utiliza la fórmula más común para ajustes en Perú: 
    # Corrección = 0.032 * (altitud_m * 0.00328) - 0.21 (Simplificada para msnm a: 0.0032 * Altitud)
    # Se usa la tabla de la OMS que es más precisa (e.g. 1000m: 0.2; 2000m: 0.7; 3000m: 1.3; 4000m: 2.1)
    
    # Interpolación lineal simple entre los puntos de la tabla OMS (Simplificado)
    if altitud_m < 1000:
        correccion = 0
    elif altitud_m < 2000:
        correccion = 0.2 + (altitud_m - 1000) / 1000 * (0.7 - 0.2)
    elif altitud_m < 3000:
        correccion = 0.7 + (altitud_m - 2000) / 1000 * (1.3 - 0.7)
    elif altitud_m < 4000:
        correccion = 1.3 + (altitud_m - 3000) / 1000 * (2.1 - 1.3)
    else: # > 4000m
        correccion = 2.1 + (altitud_m - 4000) / 1000 * 0.2 # Valor base más incremento

    # El valor de corrección siempre es positivo y se SUMA al valor medido para obtener el valor corregido.
    # El umbral de anemia SE RESTA en la práctica. Aquí usamos la corrección en el valor de Hb.
    hb_corregida = hemoglobina_medida + correccion
    
    # 2. Umbrales de Anemia (OMS)
    if 6 <= edad_meses <= 59: # Niños de 6 a 59 meses
        umbral_normal = 11.0
        # Clasificación para Hb CORREGIDA
        if hb_corregida < 7.0:
            gravedad = "SEVERA"
        elif hb_corregida < 10.0:
            gravedad = "MODERADA"
        elif hb_corregida < 11.0:
            gravedad = "LEVE"
        else:
            gravedad = "NORMAL"
    # Otras categorías (no se usan en el formulario, pero se mantienen por robustez)
    # elif 60 <= edad_meses <= 144: # Niños de 5 a 11 años
    #     umbral_normal = 11.5
    # elif 144 < edad_meses: # Adolescentes / Adultos
    #     umbral_normal = 12.0
    else: # Fuera de rango del estudio (12-60 meses)
        umbral_normal = 11.0 # Usar el de niños por defecto
        gravedad = "NORMAL (Fuera de Rango)"


    # Se retorna el umbral (solo para referencia), el valor corregido y el valor de la corrección aplicado
    # La correccion es el valor que SE SUMA, pero en la UI se muestra como el valor que SE RESTA al umbral.
    # Para mostrar en la UI la corrección como un valor RESTADO al Umbral, se retorna el negativo del valor de correccion.
    return gravedad, umbral_normal, hb_corregida, -correccion # Retornar la corrección como valor negativo

def predict_risk_ml(data):
    """Simula la predicción de riesgo de anemia usando un modelo ML (reglas simples)."""
    if not MODELO_ML:
        return 0.0, "CLASIFICACIÓN NO DISPONIBLE (Modelo IA Ausente)"

    # SIMULACIÓN DE PESOS: Se asigna probabilidad basada en factores de riesgo.
    probabilidad = 0.0 
    riesgos_activos = 0
    
    # Factores de Hemoglobina (Hb) y Edad
    hb = data['Hemoglobina_g_dL']
    edad = data['Edad_meses']
    
    if hb < 10.0:
        probabilidad += 0.40 # Peso alto para Hb baja
        riesgos_activos += 1
    elif hb < 11.5:
        probabilidad += 0.15 # Peso medio para Hb cerca del límite
        riesgos_activos += 1
        
    if edad <= 24: # Edad crítica (niños pequeños son más vulnerables)
        probabilidad += 0.20
        riesgos_activos += 1
        
    # Factores Socioeconómicos
    if data['Ingreso_Familiar_Soles'] < 1500.0:
        probabilidad += 0.15
        riesgos_activos += 1
        
    if data['Area'] == 'Rural':
        probabilidad += 0.10
        riesgos_activos += 1
        
    if data['Nivel_Educacion_Madre'] in ['Inicial', 'Sin Nivel']:
        probabilidad += 0.10
        riesgos_activos += 1
        
    if data['Nro_Hijos'] > 3:
        probabilidad += 0.05
        riesgos_activos += 1
        
    # Factores de Intervención (la ausencia aumenta el riesgo)
    if data['Suplemento_Hierro'] == 'No':
        probabilidad += 0.15
        riesgos_activos += 1
        
    if data['Programa_Juntos'] == 'No':
        probabilidad += 0.05
        
    if data['Altitud_m'] > 3000:
        probabilidad += 0.10
        
    # Normalizar la probabilidad (simulación)
    probabilidad = min(probabilidad, 0.99)
    
    # Clasificación final IA
    if probabilidad >= 0.50:
        resultado_ml = "ALTO RIESGO (Vulnerabilidad ML)"
    elif probabilidad >= 0.25:
        resultado_ml = "MEDIO RIESGO (Vulnerabilidad Moderada)"
    else:
        resultado_ml = "BAJO RIESGO (Control Recomendado)"
        
    return probabilidad, resultado_ml


def generar_sugerencias(data, resultado_final, gravedad_anemia):
    """Genera una lista de sugerencias personalizadas."""
    sugerencias = []
    
    # Sugerencias por Gravedad Clínica (Prioridad Máxima)
    if gravedad_anemia == "SEVERA":
        sugerencias.append("🚨🚨 Alerta Máxima: Derivación INMEDIATA al centro de salud para manejo urgente y transfusión si es necesario.")
    elif gravedad_anemia == "MODERADA":
        sugerencias.append("🔴 Crítico: Iniciar tratamiento con sulfato ferroso (gotas/jarabe) o complejo polivitamínico con hierro. Control y seguimiento estricto en 7 días.")
    elif gravedad_anemia == "LEVE":
        sugerencias.append("⚠️ Alerta: Reforzar el consumo de suplementos de hierro (micronutrientes o gotas) y ajustar la dieta.")

    # Sugerencias por Suplementación
    if data['Suplemento_Hierro'] == 'No':
        if gravedad_anemia in ['NORMAL', 'LEVE']:
            sugerencias.append("💊 Suplemento | Es **IMPRESCINDIBLE** iniciar la suplementación preventiva con Hierro (micronutrientes) según la edad del niño. Coordinar con puesto de salud.")
    else:
        sugerencias.append("💊 Suplemento | **Reforzar** la adherencia y correcta administración del suplemento de hierro. Evaluar la dosis si la Hb no mejora.")
        
    # Sugerencias por Dieta
    if gravedad_anemia != "SEVERA":
        sugerencias.append("🍲 Dieta | Promover el consumo diario de alimentos ricos en hierro hemo (sangrecita, hígado de pollo o res, pescado oscuro) y Vitamina C para mejorar la absorción.")
        sugerencias.append("🍲 Dieta | Evitar el consumo de lácteos y bebidas ricas en taninos (té, café) cerca de las comidas con hierro.")

    # Sugerencias por Edad
    if data['Edad_meses'] < 24:
        sugerencias.append("👶 Edad | Prioridad en la consejería nutricional para niños menores de 2 años. Énfasis en la alimentación complementaria rica en hierro.")

    # Sugerencias por Factores Sociales
    if data['Ingreso_Familiar_Soles'] < 1500.0 or data['Nivel_Educacion_Madre'] in ['Inicial', 'Sin Nivel']:
        sugerencias.append("💰 Social | Coordinar con programas sociales (Juntos) o apoyo nutricional (Vaso de Leche/Qali Warma) para asegurar la seguridad alimentaria.")
    
    if data['Area'] == 'Rural':
        sugerencias.append("📚 Educación | Estrategias de educación en salud adaptadas al contexto rural sobre higiene y prevención de parásitos, que afectan la absorción de hierro.")
        
    # Sugerencia de cierre
    if not sugerencias:
        sugerencias.append("✅ Ok | El caso está en riesgo bajo y con Hb normal/levemente baja. Continuar con monitoreo mensual y suplementación preventiva (micronutrientes).")
        
    sugerencias.append("✨ General | Se recomienda un control de Hemoglobina en un plazo de 30 días para evaluar la respuesta a la intervención.")
        
    return sugerencias

# ==============================================================================
# 4. GENERACIÓN DE INFORME PDF (Funciones)
# ==============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, unidecode('INFORME PERSONALIZADO DE RIESGO DE ANEMIA'), 0, 1, 'C')
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
    resultado_texto = f"RIESGO HÍBRIDO: {unidecode(resultado_final)}"
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, resultado_texto, 0, 1)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Gravedad Clinica (Hb Corregida): {gravedad_anemia} ({data['Hemoglobina_g_dL']} g/dL)", 0, 1)
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

    # Devolver los bytes del PDF
    return bytes(pdf.output(dest='S'))

# ==============================================================================
# 5. VISTAS DE LA APLICACIÓN (STREAMLIT UI)
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
        col_dni, col_nombre = st.columns(2)
        with col_dni: dni = st.text_input("DNI del Paciente", max_chars=8, placeholder="Solo 8 dígitos")
        with col_nombre: nombre = st.text_input("Nombre y Apellido", placeholder="Ej: Ana Torres")
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
        
        predict_button = st.form_submit_button("GENERAR INFORME PERSONALIZADO Y REGISTRAR CASO", type="primary", use_container_width=True)
        st.markdown("---")

        if predict_button:
            if not dni or len(dni) != 8 or not dni.isdigit(): st.error("Por favor, ingrese un DNI válido de 8 dígitos numéricos."); return
            if not nombre: st.error("Por favor, ingrese un nombre."); return
            
            # Altitud y Clima usan los valores calculados/asignados
            data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'Altitud_m': altitud_calculada, 'Sexo': sexo, 'Region': region, 'Area': area, 'Clima': clima, 'Ingreso_Familiar_Soles': ingreso_familiar, 'Nivel_Educacion_Madre': educacion_madre, 'Nro_Hijos': nro_hijos, 'Programa_QaliWarma': qali_warma, 'Programa_Juntos': juntos, 'Programa_VasoLeche': vaso_leche, 'Suplemento_Hierro': suplemento_hierro}

            # Clasificación Clínica con ajuste por altitud automática
            gravedad_anemia, umbral_clinico, hb_corregida, correccion_alt = clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_calculada)
            prob_alto_riesgo, resultado_ml = predict_risk_ml(data)

            if gravedad_anemia in ['SEVERA', 'MODERADA']:
                resultado_final = f"ALTO RIESGO (Alerta Clínica - {gravedad_anemia})"
            elif resultado_ml.startswith("ALTO RIESGO"):
                resultado_final = f"ALTO RIESGO (Predicción ML - Anemia {gravedad_anemia})"
            elif resultado_ml.startswith("MEDIO RIESGO") and gravedad_anemia in ['NORMAL', 'LEVE']:
                resultado_final = f"MEDIO RIESGO (Vulnerabilidad ML - Anemia {gravedad_anemia})"
            else:
                resultado_final = resultado_ml

            sugerencias_finales = generar_sugerencias(data, resultado_final, gravedad_anemia)
            # Pasamos la Region para que se guarde en la DB
            alerta_data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'riesgo': resultado_final, 'gravedad_anemia': gravedad_anemia, 'sugerencias': sugerencias_finales, 'Region': region}

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
        
        # Corrección del formato de la corrección de altitud
        # La corrección_alt es un valor negativo que indica lo que se debe restar al umbral.
        # En la UI se muestra como -|valor|.
        with col_res2: st.metric(label=f"Corrección por Altitud ({data_reporte['Altitud_m']}m)", value=f"{correccion_alt:.1f} g/dL")
        
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
    st.title("📊 Monitoreo y Gestión de Alertas (Supabase Simulada)")
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
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)", "REGISTRADO"]
        
        # Usamos ID_DB si existe (después de la migración SQL), si no, usamos la clave compuesta
        cols_to_display = ['DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION']
        if 'ID_DB' in df_monitoreo.columns:
             cols_to_display.insert(0, 'ID_DB')

        # Se renombran las columnas para la edición
        df_display = df_monitoreo[cols_to_display].copy()
        
        edited_df = st.data_editor(
            df_display,
            column_config={
                "Estado": st.column_config.SelectboxColumn("Estado de Gestión", options=opciones_estado, required=True),
                "Sugerencias": st.column_config.TextColumn("Sugerencias", width="large"),
                "ID_GESTION": None, # Ocultar la clave compuesta
                "ID_DB": st.column_config.NumberColumn("ID de Registro", disabled=True)
            },
            hide_index=True,
            key="monitoreo_data_editor"
        )

        # Lógica de guardado
        changes_detected = False
        # Comparar el DataFrame editado con el original
        for index, row in edited_df.iterrows():
            original_row = df_monitoreo.loc[index]
            
            # Solo actualizar si el estado ha cambiado
            if row['Estado'] != original_row['Estado']:
                # Usamos DNI y Fecha Alerta como clave (simulada)
                success = actualizar_estado_alerta(row['DNI'], original_row['Fecha Alerta'], row['Estado'])
                if success:
                    st.toast(f"✅ Estado de DNI {row['DNI']} actualizado a '{row['Estado']}'", icon='✅')
                    changes_detected = True
                else:
                    st.toast(f"❌ Error al actualizar estado para DNI {row['DNI']}", icon='❌')
                
        if changes_detected:
            # Recargar datos después de la actualización exitosa
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
        # Se muestra el error de la DB aquí para claridad
        if st.session_state.get('supabase_error_historial'):
             st.error(f"❌ Error al consultar el historial de registros (Supabase): {st.session_state.get('supabase_error_historial')}")
        return

    # Preparar datos: Contar por riesgo, región y estado
    df_riesgo = df_historial.groupby('Riesgo').size().reset_index(name='Conteo')
    df_estado = df_historial.groupby('Estado').size().reset_index(name='Conteo')
    
    # Filtrar solo casos de ALTO RIESGO para análisis geográfico
    df_region = df_historial[df_historial['Riesgo'].str.contains('ALTO RIESGO', na=False)].groupby('Region').size().reset_index(name='Casos de Alto Riesgo')
    
    # Asegurarse de que las fechas sean datetime para series temporales
    df_historial['Fecha Alerta'] = pd.to_datetime(df_historial['Fecha Alerta'])
    # Resamplear y agrupar por mes ('M')
    df_tendencia = df_historial.set_index('Fecha Alerta').resample('M').size().reset_index(name='Alertas Registradas')
    
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
# 7. CONFIGURACIÓN PRINCIPAL (SIDEBAR Y RUTAS)
# ==============================================================================

def main():
    # Se llama a la conexión de Supabase para mostrar el estado en el sidebar
    client = get_supabase_client()
    
    st.set_page_config(layout="wide", page_title="Sistema de Alerta IA Anemia")

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
        if MODELO_ML: st.success("✅ Modelo ML Cargado")
        else: st.error("❌ Modelo ML Falló")
        if client: st.success("✅ Supabase Conectado (Mock)")
        else: st.error("❌ Supabase Desconectado")
        
    if seleccion == "Predicción y Reporte":
        vista_prediccion()
    elif seleccion == "Monitoreo de Alertas":
        vista_monitoreo()
    elif seleccion == "Panel de control estadístico":
        vista_dashboard()

if __name__ == "__main__":
    main()
