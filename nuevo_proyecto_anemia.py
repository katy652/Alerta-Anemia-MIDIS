import streamlit as st
# Necesitas instalar 'fpdf' (o 'fpdf2' si usas versiones recientes, pero la FPDF original es más compatible con el código legado) y 'unidecode'
# pip install streamlit fpdf unidecode pandas plotly
from fpdf import FPDF
import unidecode
import datetime
import pandas as pd
import plotly.express as px
import random

# ==============================================================================
# 1. CONFIGURACIÓN Y VARIABLES GLOBALES (MOCK/PLACEHOLDER)
# ==============================================================================

# Simulación de la carga del modelo ML y las columnas esperadas
# En un entorno real, estos serían cargados desde un archivo (ej: joblib)
MODELO_COLUMNS = ['Hemoglobina_g_dL', 'Edad_meses', 'Altitud_m', 'Area_Rural', 'Clima_Frio', 'Clima_Templado', 'Nivel_Educacion_Madre_Sin_Nivel', 'Ingreso_Familiar_Soles', 'Nro_Hijos', 'Programa_QaliWarma_Si', 'Programa_Juntos_Si', 'Programa_VasoLeche_Si', 'Suplemento_Hierro_No']
MODELO_ML = "Mock Model Loaded" # Simula que el modelo ha cargado correctamente

# Simulación de la conexión a Supabase
SUPABASE_CLIENT = True # Lo establecemos a True para que el dashboard funcione
MOCK_DB_RECORDS = [] # Almacén de datos simulado
MOCK_ID_COUNTER = 1

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

def get_supabase_client():
    # Simula la conexión a Supabase
    return SUPABASE_CLIENT

def registrar_alerta_db(alerta_data):
    global MOCK_DB_RECORDS, MOCK_ID_COUNTER
    if not get_supabase_client():
        st.warning("⚠️ SIMULACIÓN: No se pudo conectar a la DB. No se registró el caso.")
        return False
        
    fecha_alerta = datetime.datetime.now().isoformat()
    record = {
        'ID_DB': MOCK_ID_COUNTER,
        'DNI': alerta_data['DNI'],
        'Nombre': alerta_data['Nombre_Apellido'],
        'Hb Inicial': alerta_data['Hemoglobina_g_dL'],
        'Riesgo': alerta_data['riesgo'],
        'Gravedad': alerta_data['gravedad_anemia'],
        'Region': alerta_data['Region'],
        'Fecha Alerta': fecha_alerta,
        'Estado': 'REGISTRADO', # Estado inicial
        'Sugerencias': ' | '.join(alerta_data['sugerencias']),
        'ID_GESTION': f"{alerta_data['DNI']}_{fecha_alerta}",
    }
    MOCK_DB_RECORDS.append(record)
    MOCK_ID_COUNTER += 1
    st.success(f"✅ SIMULACIÓN: Caso de {alerta_data['Nombre_Apellido']} registrado con ID {record['ID_DB']}.")
    return True

def obtener_alertas_pendientes_o_seguimiento():
    if not get_supabase_client():
        return pd.DataFrame()
        
    df = pd.DataFrame(MOCK_DB_RECORDS)
    if df.empty:
        return df
        
    # Se simula la consulta para obtener registros activos
    df_filtrado = df[~df['Estado'].isin(['RESUELTO', 'CERRADO (NO APLICA)'])]
    return df_filtrado.reset_index(drop=True)

def actualizar_estado_alerta(dni, fecha_alerta, nuevo_estado):
    global MOCK_DB_RECORDS
    if not get_supabase_client():
        return False
        
    target_id = f"{dni}_{fecha_alerta}"
    for i, record in enumerate(MOCK_DB_RECORDS):
        if record.get('ID_GESTION') == target_id:
            MOCK_DB_RECORDS[i]['Estado'] = nuevo_estado
            return True
    return False

def obtener_todos_los_registros():
    if not get_supabase_client():
        st.session_state['supabase_error_historial'] = "Conexión a Supabase simulada fallida."
        return pd.DataFrame()
        
    df = pd.DataFrame(MOCK_DB_RECORDS)
    if not df.empty:
        # Aseguramos que los tipos sean correctos para el dashboard
        df['Hb Inicial'] = pd.to_numeric(df['Hb Inicial'])
        df['Fecha Alerta'] = pd.to_datetime(df['Fecha Alerta'])
    return df

# ==============================================================================
# 3. FUNCIONES DE CORE LOGIC (Clasificación Clínica e IA)
# ==============================================================================

def clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_m):
    # Factor de corrección por altitud (según CDC/OMS para Hb)
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
    
    # Factores de aumento de riesgo (por IA)
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
            if not dni or len(dni) != 8: st.error("Por favor, ingrese un DNI válido de 8 dígitos."); return
            if not nombre: st.error("Por favor, ingrese un nombre."); return
            
            # Altitud y Clima usan los valores calculados/asignados
            data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'Altitud_m': altitud_calculada, 'Sexo': sexo, 'Region': region, 'Area': area, 'Clima': clima, 'Ingreso_Familiar_Soles': ingreso_familiar, 'Nivel_Educacion_Madre': educacion_madre, 'Nro_Hijos': nro_hijos, 'Programa_QaliWarma': qali_warma, 'Programa_Juntos': juntos, 'Programa_VasoLeche': vaso_leche, 'Suplemento_Hierro': suplemento_hierro}

            # Clasificación Clínica con ajuste por altitud automática
            gravedad_anemia, umbral_clinico, hb_corregida, correccion_alt = clasificar_anemia_clinica(hemoglobina, edad_meses, altitud_calculada)
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

            # Intenta registrar en DB (Mock)
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
        with col_res2: st.metric(label=f"Corrección por Altitud ({data_reporte['Altitud_m']}m)", value=f"-{abs(correccion_alt):.1f} g/dL")
        
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
    st.title("📊 Monitoreo y Gestión de Alertas (Supabase - SIMULACIÓN)")
    st.markdown("---")
    st.header("1. Casos de Monitoreo Activo (Pendientes y En Seguimiento)")
    
    if get_supabase_client() is None:
        st.error("🛑 La gestión de alertas no está disponible. No se pudo establecer conexión con Supabase.")
        return

    df_monitoreo = obtener_alertas_pendientes_o_seguimiento()

    if df_monitoreo.empty:
        st.success("No hay casos de alto riesgo o críticos pendientes de seguimiento activo. ✅")
    else:
        st.info(f"Se encontraron **{len(df_monitoreo)}** casos que requieren acción inmediata o seguimiento activo.")
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)", "REGISTRADO"]
        
        # Usamos ID_DB si existe (después de la migración SQL), si no, usamos la clave compuesta
        cols_to_display = ['DNI', 'Nombre', 'Hb Inicial', 'Riesgo', 'Fecha Alerta', 'Estado', 'Sugerencias', 'ID_GESTION', 'ID_DB']
        # Nos aseguramos de que solo se muestren las columnas que existen en el DataFrame
        cols_to_display = [col for col in cols_to_display if col in df_monitoreo.columns]
        
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
        if not df_monitoreo.empty:
            for index, row in edited_df.iterrows():
                original_row = df_monitoreo.loc[index]
                # Verificamos si el índice existe en el DataFrame original
                if index in df_monitoreo.index and row['Estado'] != original_row['Estado']:
                    # Usamos DNI y Fecha Alerta como clave de Supabase (Simulada)
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
    st.title("📊 Panel Estadístico de Alertas de Anemia (SIMULACIÓN)")
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
    
    # Filtrar solo casos de ALTO RIESGO para análisis geográfico
    df_region = df_historial[df_historial['Riesgo'].str.contains('ALTO RIESGO', na=False)].groupby('Region').size().reset_index(name='Casos de Alto Riesgo')
    
    # Asegurarse de que las fechas sean datetime para series temporales
    df_historial['Fecha Alerta'] = pd.to_datetime(df_historial['Fecha Alerta'])
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
    
    # Configuración de página (solo si la app se ejecuta por primera vez)
    # st.set_page_config(layout="wide") 

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
        if client: st.success("✅ Supabase Conectado (Simulación)")
        else: st.error("❌ Supabase Desconectado (Simulación)")
        
    if seleccion == "Predicción y Reporte":
        vista_prediccion()
    elif seleccion == "Monitoreo de Alertas":
        vista_monitoreo()
    elif seleccion == "Panel de control estadístico":
        vista_dashboard()

if __name__ == "__main__":
    main()
