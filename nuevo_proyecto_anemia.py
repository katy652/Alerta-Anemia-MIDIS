## ==============================================================================
# 0. LIBRERÍAS E IMPORTACIONES DE MOCK/PLACEHOLDER
# ==============================================================================
import streamlit as st
import pandas as pd
import datetime
from fpdf import FPDF as FPDF_lib # Alias para evitar conflicto con la clase PDF
import unidecode
import plotly.express as px
import numpy as np # Necesario para la simulación de lógica del modelo ML

# --- MOCK: Variables y Componentes No Incluidos en el Snippet ---

# Mock del modelo ML y las columnas esperadas
MODELO_ML = True # Mock: Asume que el modelo se cargó correctamente (True para simular activo)
MODELO_COLUMNS = ['Hemoglobina_g_dL', 'Edad_meses', 'Altitud_m', 'Sexo_Femenino', 'Sexo_Masculino', 'Area_Rural', 'Area_Urbana', 'Clima_Andino_Alto', 'Clima_Costa_Baja', 'Clima_Selva_Media', 'Ingreso_Familiar_Soles', 'Nivel_Educacion_Madre_Inicial', 'Nivel_Educacion_Madre_Primaria', 'Nivel_Educacion_Madre_Secundaria', 'Nivel_Educacion_Madre_Superior_Tecnica', 'Nivel_Educacion_Madre_Universitaria', 'Nivel_Educacion_Madre_Sin_Nivel', 'Nro_Hijos', 'Programa_QaliWarma_No', 'Programa_QaliWarma_Sí', 'Programa_Juntos_No', 'Programa_Juntos_Sí', 'Programa_VasoLeche_No', 'Programa_VasoLeche_Sí', 'Suplemento_Hierro_No', 'Suplemento_Hierro_Sí']

# --- MOCK: Funciones de Base de Datos (Supabase) ---

def get_supabase_client():
    # Mock: Simula el estado de la conexión a Supabase
    return True # Simula una conexión exitosa

def registrar_alerta_db(data):
    # Mock: Simula el registro en la base de datos (Supabase)
    if get_supabase_client():
        st.toast(f"✅ Caso DNI {data['DNI']} registrado/actualizado en DB (Mock).", icon='💾')
        # Simula la persistencia al actualizar el mock
        if 'alerta_data_storage' not in st.session_state:
            st.session_state.alerta_data_storage = []
        
        # Crear ID de gestión único basado en DNI y fecha actual (para el mock)
        id_gestion = f"{data['DNI']}_{datetime.date.today().isoformat()}"

        # Simular una nueva entrada para el historial (solo si es nuevo o se debe actualizar)
        new_record = {
            'ID_DB': np.random.randint(1000, 9999), # ID aleatorio para mock
            'DNI': data['DNI'],
            'Nombre': data['Nombre_Apellido'],
            'Hb Inicial': data['Hemoglobina_g_dL'],
            'Riesgo': data['riesgo'],
            'Fecha Alerta': datetime.date.today().isoformat(),
            'Estado': 'PENDIENTE (IA/VULNERABILIDAD)' if 'ALTO RIESGO' in data['riesgo'] or 'MEDIO RIESGO' in data['riesgo'] else 'REGISTRADO',
            'Sugerencias': ' | '.join(data['sugerencias']),
            'ID_GESTION': id_gestion,
            'Region': data['Region']
        }
        
        # Eliminar registros antiguos con el mismo DNI/Fecha para simular UPDATE
        st.session_state.alerta_data_storage = [
            r for r in st.session_state.alerta_data_storage 
            if r['DNI'] != data['DNI'] or r['Fecha Alerta'] != new_record['Fecha Alerta']
        ]
        
        st.session_state.alerta_data_storage.append(new_record)
        return True
    else:
        st.toast(f"❌ Falló el registro de caso DNI {data['DNI']} (DB Desconectada - Mock).", icon='❌')
        return False

def obtener_alertas_pendientes_o_seguimiento():
    # Mock: Retorna un DataFrame de ejemplo para el monitoreo
    if 'alerta_data_storage' not in st.session_state or not st.session_state.alerta_data_storage:
        # Datos iniciales si la simulación de registro aún no ha ocurrido
        data = {
            'ID_DB': [101, 102, 103],
            'DNI': ['78901234', '12345678', '99887766'],
            'Nombre': ['Juan Perez', 'Maria Lopez', 'Carlos Soto'],
            'Hb Inicial': [9.5, 10.8, 8.0],
            'Riesgo': ['ALTO RIESGO (Alerta Clínica - SEVERA)', 'RIESGO MEDIO (Vulnerabilidad ML)', 'ALTO RIESGO (Predicción ML - MODERADA)'],
            'Fecha Alerta': [datetime.date(2025, 10, 1).isoformat(), datetime.date(2025, 10, 5).isoformat(), datetime.date(2025, 10, 10).isoformat()],
            'Estado': ['PENDIENTE (CLÍNICO URGENTE)', 'EN SEGUIMIENTO', 'PENDIENTE (IA/VULNERABILIDAD)'],
            'Sugerencias': ['🚨🚨 Necesita transfusión | PRIORIDAD CLÍNICA', '💊 Suplemento | 🍲 Dieta | REVISAR ADHERENCIA', '🔴 CRITICO | 📚 Educación | VULNERABILIDAD EDUCATIVA'],
            'ID_GESTION': ['78901234_2025-10-01', '12345678_2025-10-05', '99887766_2025-10-10'],
            'Region': ['PUNO (Sierra Alta)', 'LIMA (Metropolitana y Provincia)', 'JUNÍN (Andes)']
        }
        df = pd.DataFrame(data)
        st.session_state.alerta_data_storage = df.to_dict('records') # Inicializar el mock storage
    
    # Filtrar solo los estados activos
    df_storage = pd.DataFrame(st.session_state.alerta_data_storage)
    df_monitoreo = df_storage[df_storage['Estado'].isin(['PENDIENTE (CLÍNICO URGENTE)', 'PENDIENTE (IA/VULNERABILIDAD)', 'EN SEGUIMIENTO'])].copy()
    
    # Conversión de lista de sugerencias a string para la visualización si se usó el registro
    if df_monitoreo['Sugerencias'].apply(lambda x: isinstance(x, list)).any():
         df_monitoreo['Sugerencias'] = df_monitoreo['Sugerencias'].apply(lambda x: ' | '.join(x) if isinstance(x, list) else x)

    return df_monitoreo.reset_index(drop=True)

def actualizar_estado_alerta(dni, fecha_alerta, nuevo_estado):
    # Mock: Simula la actualización del estado en el session_state
    if 'alerta_data_storage' in st.session_state:
        for i, record in enumerate(st.session_state.alerta_data_storage):
            if record['DNI'] == dni and record['Fecha Alerta'] == fecha_alerta:
                st.session_state.alerta_data_storage[i]['Estado'] = nuevo_estado
                return True
    return False # Siempre exitoso en el mock

def obtener_todos_los_registros():
    # Mock: Retorna un DataFrame completo de ejemplo para el historial y dashboard
    if 'alerta_data_storage' not in st.session_state:
        # Llama a la función de monitoreo para inicializar el storage si es necesario
        df_monitoreo_inicial = obtener_alertas_pendientes_o_seguimiento()
        df_base = df_monitoreo_inicial
    else:
        df_base = pd.DataFrame(st.session_state.alerta_data_storage)

    # Añadir registros resueltos de ejemplo (solo si no están ya en el storage)
    df_resuelto_ejemplo = pd.DataFrame({
        'ID_DB': [104, 105, 106, 107],
        'DNI': ['11112222', '33334444', '55556666', '77778888'],
        'Nombre': ['Laura Gomez', 'Pedro Flores', 'Sofia Torres', 'Ricardo Diaz'],
        'Hb Inicial': [12.5, 13.0, 11.2, 9.8],
        'Riesgo': ['RIESGO BAJO', 'RIESGO MEDIO (Vulnerabilidad ML)', 'RIESGO BAJO', 'ALTO RIESGO (Alerta Clínica - MODERADA)'],
        'Fecha Alerta': [datetime.date(2025, 9, 15).isoformat(), datetime.date(2025, 8, 20).isoformat(), datetime.date(2025, 10, 1).isoformat(), datetime.date(2025, 11, 10).isoformat()],
        'Estado': ['RESUELTO', 'CERRADO (NO APLICA)', 'REGISTRADO', 'PENDIENTE (CLÍNICO URGENTE)'],
        'Sugerencias': ['✅ Ok', '💰 Social | 👶 Edad', '✅ Ok', '🔴 CRITICO'],
        'ID_GESTION': ['11112222_2025-09-15', '33334444_2025-08-20', '55556666_2025-10-01', '77778888_2025-11-10'],
        'Region': ['ICA', 'LORETO', 'AREQUIPA', 'PUNO (Sierra Alta)']
    })

    # Concatenar todos los datos, asegurándose de que no haya duplicados basados en ID_GESTION o DNI+Fecha
    df_historial = pd.concat([df_base, df_resuelto_ejemplo], ignore_index=True).drop_duplicates(subset=['DNI', 'Fecha Alerta'], keep='last')
    
    # Conversión de lista de sugerencias a string para la visualización
    if df_historial['Sugerencias'].apply(lambda x: isinstance(x, list)).any():
        df_historial['Sugerencias'] = df_historial['Sugerencias'].apply(lambda x: ' | '.join(x) if isinstance(x, list) else x)
        
    return df_historial.sort_values(by='Fecha Alerta', ascending=False).reset_index(drop=True)

# --- MOCK: Funciones de Cálculo de Altitud/Clima/Clasificación ---

def get_altitud_por_region(region):
    if 'PUNO' in region or 'HUANCAVELICA' in region: return 4000
    if 'JUNÍN' in region or 'CUSCO' in region or 'HUÁNUCO' in region or 'PASCO' in region: return 3000
    if 'LIMA' in region or 'CALLAO' in region or 'ICA' in region or 'PIURA' in region: return 150
    if 'LORETO' in region or 'UCAYALI' in region or 'MADRE DE DIOS' in region: return 500
    return 2000 # Valor por defecto

def get_clima_por_region(region):
    altitud = get_altitud_por_region(region)
    if altitud >= 3500: return "Andino Alto (Frio Extremo)"
    if altitud >= 1500: return "Andino Medio (Templado/Frio)"
    if altitud < 1500 and ('LORETO' in region or 'UCAYALI' in region or 'AMAZONAS' in region or 'SAN MARTÍN' in region or 'MADRE DE DIOS' in region): return "Selva Media/Baja (Cálido Húmedo)"
    return "Costa/Urbano (Cálido/Seco)"

def clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_m):
    # 1. Corrección por Altitud (Ejemplo simplificado según normativas internacionales)
    if altitud_m < 1000: correccion_alt = 0.0
    elif altitud_m < 2000: correccion_alt = -0.3
    elif altitud_m < 3000: correccion_alt = -0.8
    elif altitud_m < 4000: correccion_alt = -1.5
    else: correccion_alt = -2.0 # Altitudes muy altas

    hb_corregida = hemoglobina + correccion_alt
    hb_corregida = max(hb_corregida, 5.0)

    # 2. Determinación del Umbral (OMS para 6 a 59 meses)
    # Se utiliza el umbral de 11.0 g/dL para este rango de edad (6 a 59 meses)
    umbral_clinico = 11.0

    # 3. Clasificación de Gravedad (OMS para 6-59 meses)
    if hb_corregida < 7.0: gravedad_anemia = "SEVERA"
    elif hb_corregida < 10.0: gravedad_anemia = "MODERADA"
    elif hb_corregida < umbral_clinico: gravedad_anemia = "LEVE"
    else: gravedad_anemia = "NORMAL"

    return gravedad_anemia, umbral_clinico, hb_corregida, correccion_alt

# --- MOCK: Funciones de Predicción ML y Sugerencias ---

def predict_risk_ml(data):
    # Mock: Simula la predicción del modelo de Machine Learning
    if MODELO_ML is None:
        return 0.0, "RIESGO BAJO (ML no disponible)"

    hemoglobina = data['Hemoglobina_g_dL']
    altitud_m = data['Altitud_m']

    # La probabilidad de riesgo es inversamente proporcional a la Hb y directamente a la altitud
    base_risk = 1.0 - (hemoglobina / 14.0)
    altitud_boost = altitud_m / 4000.0 * 0.2

    prob_riesgo = min(1.0, base_risk + altitud_boost)
    
    # Ajuste por factores sociales (más hijos, menos ingreso, menos educación = más riesgo)
    if data['Nro_Hijos'] > 3: prob_riesgo += 0.05
    if data['Ingreso_Familiar_Soles'] < 1000: prob_riesgo += 0.10
    if data['Nivel_Educacion_Madre'] in ['Inicial', 'Sin Nivel']: prob_riesgo += 0.10
    if data['Area'] == 'Rural': prob_riesgo += 0.05
    if data['Suplemento_Hierro'] == 'No': prob_riesgo += 0.10

    prob_riesgo = np.clip(prob_riesgo, 0.01, 0.99)

    # Clasificación ML (Umbral Alto Riesgo > 0.7)
    if prob_riesgo >= 0.70:
        resultado_ml = "ALTO RIESGO (Vulnerabilidad ML)"
    elif prob_riesgo >= 0.40:
        resultado_ml = "MEDIO RIESGO (Vulnerabilidad ML)"
    else:
        resultado_ml = "RIESGO BAJO"

    return prob_riesgo, resultado_ml

def generar_sugerencias(data, resultado_final, gravedad_anemia):
    sugerencias = []

    # Sugerencias Clínicas
    if gravedad_anemia == "SEVERA":
        sugerencias.append("🚨🚨 TRATAMIENTO URGENTE: Referir inmediatamente a un centro de salud para evaluación y posible transfusión sanguínea. | PRIORIDAD CLÍNICA")
    elif gravedad_anemia == "MODERADA":
        sugerencias.append("🔴 INTERVENCIÓN CRÍTICA: Iniciar tratamiento intensivo con suplementos de hierro terapéuticos bajo supervisión médica inmediata. | SEGUIMIENTO CERCANO")
    elif gravedad_anemia == "LEVE":
        sugerencias.append("⚠️ MONITOREO Y PREVENCIÓN: Reforzar la suplementación de hierro preventiva y asegurar un seguimiento en 3 meses. | PREVENCIÓN")
    else:
        sugerencias.append("✅ Hemoglobina en rango normal. Continuar con medidas preventivas de salud y nutrición. | CONTINUIDAD")
    
    # Sugerencias por Suplementación
    if data['Suplemento_Hierro'] == 'No' and gravedad_anemia != "NORMAL":
        sugerencias.append("💊 SUPLEMENTACIÓN URGENTE: El paciente NO está recibiendo suplementos. Es crucial iniciar el esquema apropiado (sulfato ferroso, multimicronutrientes). | FALTA DE ACCESO")
    elif data['Suplemento_Hierro'] == 'Sí' and gravedad_anemia != "NORMAL":
        sugerencias.append("💊 ADHERENCIA: Investigar la adherencia o absorción del suplemento de hierro. Es posible que la dosis o la ingesta sean inadecuadas. | REVISAR ADHERENCIA")

    # Sugerencias Socioeconómicas y Contextuales
    if data['Nivel_Educacion_Madre'] in ["Inicial", "Sin Nivel", "Primaria"]:
        sugerencias.append("📚 EDUCACIÓN NUTRICIONAL: Priorizar sesiones de educación para la madre/cuidador sobre preparación de alimentos ricos en hierro y la importancia de la adherencia al tratamiento. | VULNERABILIDAD EDUCATIVA")
    
    if data['Ingreso_Familiar_Soles'] < 1500 or data['Programa_Juntos'] == 'No':
        sugerencias.append("💰 APOYO SOCIAL: Evaluar la elegibilidad para programas de transferencia condicionada (Juntos) o apoyo nutricional adicional, dada la baja capacidad económica. | VULNERABILIDAD ECONÓMICA")

    if data['Area'] == 'Rural':
        sugerencias.append("🍲 ENFOQUE RURAL: Promover huertos familiares o acceso a alimentos frescos locales. Considerar la dificultad de acceso a servicios de salud. | CONTEXTO GEOGRÁFICO")

    return sugerencias

# ==============================================================================
# 4. GENERACIÓN DE INFORME PDF (Funciones)
# ==============================================================================

class PDF(FPDF_lib):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, unidecode.unidecode('INFORME PERSONALIZADO DE RIESGO DE ANEMIA'), 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, unidecode.unidecode('Ministerio de Desarrollo e Inclusion Social (MIDIS)'), 0, 1, 'C')
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
    pdf.cell(0, 5, unidecode.unidecode(f"Nombre: {data['Nombre_Apellido']}"), 0, 1)
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
    pdf.cell(0, 5, unidecode.unidecode(f"Gravedad Clinica (Hb Corregida): {gravedad_anemia} ({data['Hemoglobina_g_dL']} g/dL)"), 0, 1)
    pdf.cell(0, 5, f"Prob. de Alto Riesgo por IA: {prob_riesgo:.2%}", 0, 1)
    pdf.ln(5)

    pdf.chapter_title('III. PLAN DE INTERVENCION PERSONALIZADO')
    pdf.set_font('Arial', '', 10)
    for sug in sugerencias:
        final_text = sug.replace('|', ' - ').replace('🚨🚨', '[EMERGENCIA]').replace('🔴', '[CRITICO]').replace('⚠️', '[ALERTA]').replace('💊', '[Suplemento]').replace('🍲', '[Dieta]').replace('💰', '[Social]').replace('👶', '[Edad]').replace('✅', '[Ok]').replace('📚', '[Educacion]').replace('✨', '[General]')
        # Aplicar unidecode después del reemplazo para manejar acentos en el texto de las sugerencias
        final_text = unidecode.unidecode(final_text) 
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
    # Inicialización de session_state para hb_corregida y correccion_alt
    if 'hb_corregida' not in st.session_state: st.session_state.hb_corregida = 0.0
    if 'correccion_alt' not in st.session_state: st.session_state.correccion_alt = 0.0
    if 'prediction_done' not in st.session_state: st.session_state.prediction_done = False
    
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
        "CAJAMARCA", "AREQUIPA", "MOQUEGUE", "TACNA",
        "PUNO (Sierra Alta)", "HUANCAVELICA (Sierra Alta)", "PASCO",
        "LORETO", "AMAZONAS", "SAN MARTÍN", "UCAYALI", "MADRE DE DIOS",
        "OTRO / NO ESPECIFICADO"
    ]

    with st.form("formulario_prediccion"):
        st.subheader("0. Datos de Identificación y Contacto")
        col_dni, col_nombre = st.columns(2)
        with col_dni: dni = st.text_input("DNI del Paciente", max_chars=8, placeholder="Solo 8 dígitos", key="dni_input")
        with col_nombre: nombre = st.text_input("Nombre y Apellido", placeholder="Ej: Ana Torres", key="nombre_input")
        st.markdown("---")
        
        st.subheader("1. Factores Clínicos y Demográficos Clave")
        col_h, col_e, col_r = st.columns(3)
        with col_h: hemoglobina = st.number_input("Hemoglobina (g/dL) - CRÍTICO", min_value=5.0, max_value=18.0, value=10.5, step=0.1, key="hb_input")
        with col_e: edad_meses = st.slider("Edad (meses)", min_value=12, max_value=60, value=36, key="edad_input")
        with col_r: region = st.selectbox("Región (Define Altitud y Clima)", options=REGIONES_PERU, key="region_input")
        
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
            
        with col_ed: educacion_madre = st.selectbox("Nivel Educ. Madre", options=["Secundaria", "Primaria", "Superior Técnica", "Universitaria", "Inicial", "Sin Nivel"], key="educacion_input")
        
        col_hijos, col_ing, col_area, col_s = st.columns(4)
        with col_hijos: nro_hijos = st.number_input("Nro. de Hijos en el Hogar", min_value=1, max_value=15, value=2, key="hijos_input")
        with col_ing: ingreso_familiar = st.number_input("Ingreso Familiar (Soles/mes)", min_value=0.0, max_value=5000.0, value=1800.0, step=10.0, key="ingreso_input")
        with col_area: area = st.selectbox("Área de Residencia", options=['Urbana', 'Rural'], key="area_input")
        with col_s: sexo = st.selectbox("Sexo", options=["Femenino", "Masculino"], key="sexo_input")
        st.markdown("---")
        
        st.subheader("3. Acceso a Programas y Servicios")
        col_q, col_j, col_v, col_hierro = st.columns(4)
        with col_q: qali_warma = st.radio("Programa Qali Warma", options=["No", "Sí"], horizontal=True, key="qw_input")
        with col_j: juntos = st.radio("Programa Juntos", options=["No", "Sí"], horizontal=True, key="juntos_input")
        with col_v: vaso_leche = st.radio("Programa Vaso de Leche", options=["No", "Sí"], horizontal=True, key="vl_input")
        with col_hierro: suplemento_hierro = st.radio("Recibe Suplemento de Hierro", options=["No", "Sí"], horizontal=True, key="hierro_input")
        st.markdown("---")
        
        predict_button = st.form_submit_button("GENERAR INFORME PERSONALIZADO Y REGISTRAR CASO", type="primary", use_container_width=True)
        st.markdown("---")

        if predict_button:
            if not dni or len(dni) != 8: st.error("Por favor, ingrese un DNI válido de 8 dígitos."); return
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
            elif resultado_ml.startswith("MEDIO RIESGO") and gravedad_anemia == "LEVE":
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
        
        # correccion_alt es un valor negativo o cero que representa el ajuste. Se muestra con el signo.
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
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)", "REGISTRADO"]
        
        # Usamos ID_DB si existe (después de la migración SQL), si no, usamos la clave compuesta
        cols_to_display = ['DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION', 'Region']
        if 'ID_DB' in df_monitoreo.columns:
            cols_to_display.insert(0, 'ID_DB')

        df_display = df_monitoreo[[col for col in cols_to_display if col in df_monitoreo.columns]].copy()
        
        # Configuración de columnas para data_editor
        column_config = {
            "Estado": st.column_config.SelectboxColumn("Estado de Gestión", options=opciones_estado, required=True),
            "Sugerencias": st.column_config.TextColumn("Sugerencias", width="large"),
            "ID_GESTION": None, # Ocultar la clave compuesta
            "Region": st.column_config.TextColumn("Región", disabled=True),
            "DNI": st.column_config.TextColumn("DNI", disabled=True)
        }
        if 'ID_DB' in df_display.columns:
            column_config["ID_DB"] = st.column_config.NumberColumn("ID de Registro", disabled=True)
            
        edited_df = st.data_editor(
            df_display,
            column_config=column_config,
            hide_index=True,
            key="monitoreo_data_editor"
        )

        # Lógica de guardado
        changes_detected = False
        if "monitoreo_data_editor" in st.session_state:
            # Detectar cambios solo en el campo 'Estado' del data_editor
            current_df = st.session_state["monitoreo_data_editor"]["edited_rows"]
            
            for index, row_changes in current_df.items():
                if 'Estado' in row_changes:
                    # Obtenemos el registro original por índice para obtener la clave compuesta
                    original_row = df_monitoreo.loc[index]
                    dni = original_row['DNI']
                    fecha_alerta = original_row['Fecha Alerta'] # Asumiendo que esta fecha es parte de la clave única
                    nuevo_estado = row_changes['Estado']
                    
                    success = actualizar_estado_alerta(dni, fecha_alerta, nuevo_estado)
                    
                    if success:
                        st.toast(f"✅ Estado de DNI {dni} actualizado a '{nuevo_estado}'", icon='✅')
                        changes_detected = True
                    else:
                        st.toast(f"❌ Error al actualizar estado para DNI {dni}", icon='❌')
        
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
        return

    # Preparar datos: Contar por riesgo, región y estado
    df_riesgo = df_historial.groupby('Riesgo').size().reset_index(name='Conteo')
    df_estado = df_historial.groupby('Estado').size().reset_index(name='Conteo')
    
    # Filtrar solo casos de ALTO RIESGO para análisis geográfico
    df_region = df_historial[df_historial['Riesgo'].str.contains('ALTO RIESGO', na=False)].groupby('Region').size().reset_index(name='Casos de Alto Riesgo')
    
    # Asegurarse de que las fechas sean datetime para series temporales
    try:
        df_historial['Fecha Alerta'] = pd.to_datetime(df_historial['Fecha Alerta'])
        # Contar por mes y año
        df_historial['AñoMes'] = df_historial['Fecha Alerta'].dt.to_period('M')
        df_tendencia = df_historial.groupby('AñoMes').size().reset_index(name='Alertas Registradas')
        df_tendencia['Fecha Alerta'] = df_tendencia['AñoMes'].astype(str)
        df_tendencia.drop(columns=['AñoMes'], inplace=True)
    except Exception as e:
        st.warning(f"⚠️ Error al procesar fechas para la tendencia: {e}. Mostrando solo datos de resumen.")
        df_tendencia = pd.DataFrame({'Fecha Alerta': [], 'Alertas Registradas': []})
        
    # --- FILTROS ---
    st.sidebar.header("Filtros del Dashboard")
    regiones_disponibles = sorted(df_historial['Region'].unique())
    # Usar el filtro solo si hay regiones disponibles
    if regiones_disponibles and len(regiones_disponibles) > 0:
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
        
        # Recalcular conteo de riesgo para el filtro
        df_riesgo_filtrado = df_filtrado.groupby('Riesgo').size().reset_index(name='Conteo')

        fig_riesgo = px.pie(
            df_riesgo_filtrado, 
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
        
        # Recalcular conteo de estado para el filtro
        df_estado_filtrado = df_filtrado.groupby('Estado').size().reset_index(name='Conteo')

        fig_estado = px.bar(
            df_estado_filtrado,
            y='Conteo', 
            x='Estado', 
            title='Estado de Gestión de Alertas',
            color='Estado',
            color_discrete_map={
                'PENDIENTE (CLÍNICO URGENTE)': '#e43a3a', 
                'PENDIENTE (IA/VULNERABILIDAD)': '#ffa500', 
                'EN SEGUIMIENTO': '#4169e1', 
                'RESUELTO': '#228b22', 
                'REGISTRADO': '#a9a9a9', 
                'CERRADO (NO APLICA)': '#8a2be2' 
            }
        )
        fig_estado.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_estado, use_container_width=True)

    st.markdown("---")
    st.header("2. Tendencias y Distribución Geográfica")
    
    # 2.1 Gráfico de Tendencia Mensual (Ancho Completo)
    if not df_tendencia.empty:
        st.subheader("Tendencia Mensual de Alertas")
        
        # Si hay filtro, se recalcula la tendencia con df_filtrado
        if len(regiones_disponibles) > 0 and len(filtro_region) < len(regiones_disponibles):
            df_tendencia_filtrado = df_filtrado.groupby('AñoMes').size().reset_index(name='Alertas Registradas')
            df_tendencia_filtrado['Fecha Alerta'] = df_tendencia_filtrado['AñoMes'].astype(str)
            df_tendencia_filtrado.drop(columns=['AñoMes'], inplace=True)
            data_tendencia = df_tendencia_filtrado
        else:
            data_tendencia = df_tendencia
            
        fig_tendencia = px.line(
            data_tendencia,
            x='Fecha Alerta',
            y='Alertas Registradas',
            title='Alertas Registradas por Mes',
            markers=True
        )
        fig_tendencia.update_layout(hovermode="x unified")
        st.plotly_chart(fig_tendencia, use_container_width=True)
    else:
        st.info("No hay datos suficientes para mostrar la tendencia mensual.")

    # 2.2 Gráfico de Casos de Alto Riesgo por Región (Ancho Completo)
    st.subheader("Casos de Alto Riesgo por Región (Top 10)")
    
    # Recalcular alto riesgo por región usando df_filtrado
    df_region_filtrado = df_filtrado[df_filtrado['Riesgo'].str.contains('ALTO RIESGO', na=False)].groupby('Region').size().reset_index(name='Casos de Alto Riesgo')
    df_region_top = df_region_filtrado.sort_values(by='Casos de Alto Riesgo', ascending=False).head(10)
    
    if not df_region_top.empty:
        fig_region = px.bar(
            df_region_top,
            y='Region',
            x='Casos de Alto Riesgo',
            orientation='h',
            title='Regiones con Mayor Alto Riesgo',
            color='Casos de Alto Riesgo',
            color_continuous_scale=px.colors.sequential.Sunset
        )
        fig_region.update_yaxes(autorange="reversed") # Para que el mayor esté arriba
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("No hay casos de Alto Riesgo para analizar geográficamente.")

# ==============================================================================
# 7. CONFIGURACIÓN PRINCIPAL (SIDEBAR Y RUTAS)
# ==============================================================================

def main():
    # Configuración inicial de la página de Streamlit
    st.set_page_config(layout="wide", page_title="Sistema de Alerta IA Anemia", page_icon="🩸")

    # Se llama a la conexión de Supabase para mostrar el estado en el sidebar
    client = get_supabase_client()
    
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
        if MODELO_ML: st.success("✅ Modelo ML Cargado (Mock)")
        else: st.warning("⚠️ Modelo ML Inactivo")

        if client: st.success("✅ Conexión DB Activa (Mock)")
        else: st.error("❌ Conexión DB Fallida (Mock)")
        
    # Lógica de enrutamiento
    if seleccion == "Predicción y Reporte":
        vista_prediccion()
    elif seleccion == "Monitoreo de Alertas":
        vista_monitoreo()
    elif seleccion == "Panel de control estadístico":
        vista_dashboard()

if __name__ == '__main__':
    main()
