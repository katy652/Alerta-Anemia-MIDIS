# app.py (versión corregida y completa basada en tu código)
import streamlit as st
import pandas as pd
import joblib
import unidecode
from supabase import create_client, Client
import datetime
from fpdf import FPDF
import base64
import json
import gdown
import os
import io
import re
import requests
import traceback

st.set_page_config(page_title="Alerta de Riesgo de Anemia (IA)", page_icon="🩸", layout="wide")

# === UMBRALES ===
UMBRAL_SEVERA = 7.0
UMBRAL_MODERADA = 9.0
UMBRAL_HEMOGLOBINA_ANEMIA = 11.0

# === ML ===
MODELO_URL = "https://drive.google.com/uc?export=download&id=1vij71K2DtTHEc1seEOqeYk-fV2AQNfBK"
COLUMNS_FILENAME = "modelo_columns.joblib"

SUPABASE_TABLE = "alertas"

# -------------------------
# Inicialización segura del cliente Supabase (NO en import)
# -------------------------
def get_supabase_credentials():
    """Obtiene credenciales, prioriza st.secrets (Streamlit Cloud)."""
    url = None
    key = None
    try:
        if hasattr(st, "secrets") and "SUPABASE_URL" in st.secrets:
            url = st.secrets.get("SUPABASE_URL")
            key = st.secrets.get("SUPABASE_KEY")
        # fallback a variables de entorno (local)
        if not url:
            url = os.environ.get("SUPABASE_URL")
        if not key:
            key = os.environ.get("SUPABASE_KEY")
    except Exception:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
    return url, key

def init_supabase_client_safe():
    """
    Inicializa el cliente pero controla errores con detalle.
    No lo llamamos en el import para evitar crash en arranque.
    """
    url, key = get_supabase_credentials()
    if not url or not key:
        st.error("⚠️ No se encontraron las credenciales de Supabase. Añade SUPABASE_URL y SUPABASE_KEY en Secrets (Streamlit Cloud) o en variables de entorno.")
        return None, "Credenciales no configuradas"
    try:
        client = create_client(url, key)
        # hacemos una prueba pequeña (sin lanzar) para capturar error DNS/SSL pronto
        # Nota: no hacemos queries pesadas
        return client, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Excepción creando cliente Supabase: {e}\n{tb}"

# -------------------------
# Diagnóstico de Conexión (botón)
# -------------------------
def diagnostico_conexion_ui():
    st.sidebar.markdown("### 🔎 Diagnóstico Supabase")
    if st.sidebar.button("Probar conexión a Supabase"):
        with st.spinner("Probando conexión..."):
            client, err = init_supabase_client_safe()
            if client:
                try:
                    resp = client.table(SUPABASE_TABLE).select("*").limit(1).execute()
                    if resp.error:
                        st.sidebar.error(f"❌ Conexión establecida pero error en SELECT: {resp.error.message}")
                    else:
                        st.sidebar.success("✅ Conexión y SELECT OK. Respuesta recibida.")
                        st.sidebar.write(resp.data if resp.data else "Tabla vacía (o sin permisos para ver datos).")
                except Exception as e:
                    st.sidebar.error(f"❌ Error ejecutando SELECT: {e}")
            else:
                st.sidebar.error(f"❌ No se pudo inicializar cliente. Detalle: {err}")

# -------------------------
# Carga de modelo y columnas (cache)
# -------------------------
@st.cache_resource
def load_model_components():
    try:
        model_columns = joblib.load(COLUMNS_FILENAME)
    except Exception as e:
        st.error(f"❌ ERROR: No se pudo cargar '{COLUMNS_FILENAME}': {e}")
        return None, None
    try:
        if not os.path.exists("modelo_anemia.joblib"):
            gdown.download(url=MODELO_URL, output="modelo_anemia.joblib", quiet=True, fuzzy=True)
        model = joblib.load("modelo_anemia.joblib")
        return model, model_columns
    except Exception as e:
        st.error(f"❌ ERROR al descargar/cargar el modelo: {e}")
        return None, None

MODELO_ML, MODELO_COLUMNS = load_model_components()
RISK_MAPPING = {0: "BAJO RIESGO", 1: "MEDIO RIESGO", 2: "ALTO RIESGO"}

# -------------------------
# Funciones de negocio (igual que tu versión original)
# -------------------------
def limpiar_texto(texto):
    if pd.isna(texto): return 'desconocido'
    return unidecode.unidecode(str(texto).strip().lower())

def clasificar_anemia_clinica(hemoglobina_g_dL, edad_meses):
    umbral = 0
    if 6 <= edad_meses <= 59: umbral = 11.0
    elif 60 <= edad_meses <= 144: umbral = 11.5
    else: umbral = 12.0
    if hemoglobina_g_dL < UMBRAL_SEVERA: return "SEVERA", umbral
    elif hemoglobina_g_dL < UMBRAL_MODERADA: return "MODERADA", umbral
    elif hemoglobina_g_dL < umbral: return "LEVE", umbral
    else: return "NO ANEMIA", umbral

def preprocess_data_for_ml(data_raw, model_columns):
    data_ml = {
        'Hemoglobina_g_dL': data_raw['Hemoglobina_g_dL'],
        'Edad_meses': data_raw['Edad_meses'],
        'Altitud_m': data_raw['Altitud_m'],
        'Ingreso_Familiar_Soles': data_raw['Ingreso_Familiar_Soles'],
        'Nro_Hijos': data_raw['Nro_Hijos']
    }
    df_pred = pd.DataFrame([data_ml])
    categorical_cols = ['Sexo', 'Region', 'Area', 'Clima', 'Nivel_Educacion_Madre', 'Programa_QaliWarma', 'Programa_Juntos', 'Programa_VasoLeche', 'Suplemento_Hierro']
    for col in categorical_cols:
        if col in data_raw:
            df_pred[col] = limpiar_texto(data_raw[col])
    df_encoded = pd.get_dummies(df_pred)
    missing_cols = set(model_columns) - set(df_encoded.columns)
    for c in missing_cols:
        df_encoded[c] = 0
    df_final = df_encoded[model_columns]
    df_final = df_final.astype({col: 'float64' for col in df_final.columns})
    return df_final

def predict_risk_ml(data_raw):
    if MODELO_ML is None or MODELO_COLUMNS is None:
        return 0.5, "ERROR: Modelo IA no disponible"
    try:
        X_df = preprocess_data_for_ml(data_raw, MODELO_COLUMNS)
        resultado_clase = MODELO_ML.predict(X_df)[0]
        prob_riesgo_array = MODELO_ML.predict_proba(X_df)[0]
        prob_alto_riesgo = prob_riesgo_array[2] if len(prob_riesgo_array) > 2 else prob_riesgo_array.max()
        resultado_texto = RISK_MAPPING.get(resultado_clase, "RIESGO INDEFINIDO")
        return prob_alto_riesgo, resultado_texto
    except Exception as e:
        return 0.5, f"ERROR: Fallo en el motor de IA: {e}"

# -------------------------
# Funciones Supabase (lectura/escritura) - usan cliente inicializado al momento de usar
# -------------------------
def registrar_alerta_db(data_alerta):
    supabase, err = init_supabase_client_safe()
    if supabase is None:
        st.error(f"❌ No se pudo registrar: {err}")
        return False
    try:
        if 'SEVERA' in data_alerta['gravedad_anemia'] or 'MODERADA' in data_alerta['gravedad_anemia']:
            estado = 'PENDIENTE (CLÍNICO URGENTE)'
        elif data_alerta['riesgo'].startswith("ALTO RIESGO"):
            estado = 'PENDIENTE (IA/VULNERABILIDAD)'
        else:
            estado = 'REGISTRADO'

        payload = {
            'dni': data_alerta['DNI'],
            'nombre_apellido': data_alerta['Nombre_Apellido'],
            'edad_meses': data_alerta['Edad_meses'],
            'hemoglobina_g_dL': data_alerta['Hemoglobina_g_dL'],
            'riesgo': data_alerta['riesgo'],
            'fecha_alerta': datetime.date.today().isoformat(),
            'estado': estado,
            'sugerencias': json.dumps(data_alerta['sugerencias'])
        }

        response = supabase.table(SUPABASE_TABLE).insert(payload).execute()
        if getattr(response, "error", None):
            st.error(f"❌ Error de API Supabase al insertar: {response.error.message}")
            return False
        # limpiar caches si existieran (decoradores los usan)
        try:
            obtener_alertas_pendientes_o_seguimiento.clear()
            obtener_todos_los_registros.clear()
        except Exception:
            pass
        st.info("✅ Caso registrado en Supabase.")
        return True
    except Exception as e:
        st.error(f"❌ Error al registrar en Supabase: {e}")
        return False

def fetch_data(query_condition=None):
    supabase, err = init_supabase_client_safe()
    if supabase is None:
        st.warning(f"⚠️ La conexión a Supabase no está activa: {err}")
        return pd.DataFrame()
    try:
        query = supabase.table(SUPABASE_TABLE).select('*').order('fecha_alerta', desc=True).order('id', desc=True)
        if query_condition:
            query = query.or_(query_condition)
        response = query.execute()
        if getattr(response, "error", None):
            st.error(f"❌ Error al obtener datos (SELECT): {response.error.message}")
            return pd.DataFrame()
        if response.data:
            df = pd.DataFrame(response.data)
            df = df.rename(columns={
                'id': 'ID',
                'dni': 'DNI',
                'nombre_apellido': 'Nombre',
                'edad_meses': 'Edad (meses)',
                'hemoglobina_g_dL': 'Hb Inicial',
                'riesgo': 'Riesgo',
                'fecha_alerta': 'Fecha Alerta',
                'estado': 'Estado',
                'sugerencias': 'Sugerencias'
            })
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Excepción al consultar Supabase: {e}")
        return pd.DataFrame()

@st.cache_data
def obtener_alertas_pendientes_o_seguimiento():
    query_condition = "estado.ilike.PENDIENTE%,estado.eq.EN SEGUIMIENTO"
    df = fetch_data(query_condition=query_condition)
    if not df.empty:
        df['Sugerencias'] = df['Sugerencias'].apply(lambda s: safe_json_to_text_display(s))
    return df

@st.cache_data
def obtener_todos_los_registros():
    df = fetch_data()
    if not df.empty:
        df['Sugerencias'] = df['Sugerencias'].apply(lambda s: safe_json_to_text_display(s))
    return df

def actualizar_estado_alerta(alerta_id, nuevo_estado):
    supabase, err = init_supabase_client_safe()
    if supabase is None:
        st.error(f"No se pudo actualizar: {err}")
        return False
    try:
        response = supabase.table(SUPABASE_TABLE).update({'estado': nuevo_estado}).eq('id', alerta_id).execute()
        if getattr(response, "error", None):
            st.error(f"Error de API Supabase al actualizar: {response.error.message}")
            return False
        try:
            obtener_alertas_pendientes_o_seguimiento.clear()
            obtener_todos_los_registros.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        st.error(f"Error al actualizar en Supabase: {e}")
        return False

def safe_json_to_text_display(json_str):
    if isinstance(json_str, str) and json_str.strip() and json_str.startswith('['):
        try:
            sug_list = json.loads(json_str)
            return "\n".join([s for s in sug_list])
        except Exception:
            return json_str
    return json_str or ""

# -------------------------
# PDF (tu versión)
# -------------------------
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
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, f"RIESGO HÍBRIDO: {unidecode.unidecode(resultado_final)}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Gravedad Clínica (Hb): {gravedad_anemia} ({data['Hemoglobina_g_dL']} g/dL)", 0, 1)
    pdf.cell(0, 5, f"Prob. de Alto Riesgo por IA: {prob_riesgo:.2%}", 0, 1)
    pdf.ln(5)
    pdf.chapter_title('III. PLAN DE INTERVENCION PERSONALIZADO')
    pdf.set_font('Arial', '', 10)
    for sug in sugerencias:
        final_text = sug.replace('|', ' - ')
        pdf.set_fill_color(240, 240, 240)
        pdf.multi_cell(0, 6, f"- {final_text}", 0, 'L')
        pdf.ln(1)
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, "--- Fin del Informe ---", 0, 1, 'C')
    return bytes(pdf.output(dest='S'))

# -------------------------
# Vistas (igual que tu estructura)
# -------------------------
def vista_prediccion():
    st.title("📝 Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.1 Híbrida)")
    st.markdown("---")
    if MODELO_ML is None:
        st.error("❌ El formulario está deshabilitado. No se pudo cargar el modelo de IA.")
        return
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
        with col_a: altitud = st.number_input("Altitud (metros s.n.m.)", min_value=0, max_value=5000, value=1500, step=10)
        st.markdown("---")
        st.subheader("2. Factores Socioeconómicos y Contextuales")
        col_r, col_c, col_ed = st.columns(3)
        with col_r: region = st.selectbox("Región", options=['Lima', 'Junín', 'Piura', 'Cusco', 'Arequipa', 'Otro'])
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
            gravedad_anemia, umbral_clinico = clasificar_anemia_clinica(hemoglobina, edad_meses)
            prob_alto_riesgo, resultado_ml = predict_risk_ml(data)
            if gravedad_anemia in ['SEVERA', 'MODERADA']: resultado_final = f"ALTO RIESGO (Alerta Clínica - {gravedad_anemia})"
            elif resultado_ml.startswith("ALTO RIESGO"): resultado_final = f"ALTO RIESGO (Predicción ML - Anemia {gravedad_anemia})"
            else: resultado_final = resultado_ml
            sugerencias_finales = generar_sugerencias(data, resultado_final, gravedad_anemia) if 'generar_sugerencias' in globals() else ["No hay sugerencias definidas."]
            alerta_data = {'DNI': dni, 'Nombre_Apellido': nombre, 'Hemoglobina_g_dL': hemoglobina, 'Edad_meses': edad_meses, 'riesgo': resultado_final, 'gravedad_anemia': gravedad_anemia, 'sugerencias': sugerencias_finales}
            registrar_alerta_db(alerta_data)
            st.session_state.resultado = resultado_final; st.session_state.prob_alto_riesgo = prob_alto_riesgo; st.session_state.gravedad_anemia = gravedad_anemia; st.session_state.sugerencias_finales = sugerencias_finales; st.session_state.data_reporte = data; st.session_state.prediction_done = True
            st.rerun()
    if st.session_state.prediction_done:
        resultado_final = st.session_state.resultado; prob_alto_riesgo = st.session_state.prob_alto_riesgo; gravedad_anemia = st.session_state.gravedad_anemia; sugerencias_finales = st.session_state.sugerencias_finales; data_reporte = st.session_state.data_reporte
        st.header("Análisis y Reporte de Control Oportuno")
        if resultado_final.startswith("ALTO"): st.error(f"## 🔴 RIESGO: {resultado_final}")
        elif resultado_final.startswith("MEDIO"): st.warning(f"## 🟠 RIESGO: {resultado_final}")
        else: st.success(f"## 🟢 RIESGO: {resultado_final}")
        col_res1, col_res2 = st.columns(2)
        with col_res1: st.metric(label="Clasificación Clínica (Gravedad Hb)", value=gravedad_anemia)
        with col_res2: st.metric(label="Prob. de Alto Riesgo por IA", value=f"{prob_alto_riesgo:.2%}")
        st.subheader("📝 Sugerencias Personalizadas de Intervención Oportuna:")
        for sugerencia in sugerencias_finales: st.info(sugerencia.replace('|', '** | **'))
        st.markdown("---")
        try:
            pdf_data = generar_informe_pdf_fpdf(data_reporte, resultado_final, prob_alto_riesgo, sugerencias_finales, gravedad_anemia)
            st.download_button(label="⬇️ Descargar Informe de Recomendaciones Individual (PDF)", data=pdf_data, file_name=f'informe_riesgo_DNI_{data_reporte["DNI"]}_{datetime.date.today().isoformat()}.pdf', mime='application/pdf', type="secondary")
        except Exception as pdf_error:
            st.error(f"⚠️ Error al generar el PDF. Detalle: {pdf_error}")

def vista_monitoreo():
    st.title("📊 Monitoreo y Gestión de Alertas (Supabase)")
    st.markdown("---")
    supabase, err = init_supabase_client_safe()
    if supabase is None:
        st.error(f"🛑 La gestión de alertas no está disponible. Detalle: {err}")
        return
    df_monitoreo = obtener_alertas_pendientes_o_seguimiento()
    if df_monitoreo.empty:
        st.success("No hay casos de alto riesgo o críticos pendientes de seguimiento activo. ✅")
    else:
        st.info(f"Se encontraron **{len(df_monitoreo)}** casos que requieren acción inmediata o seguimiento activo.")
        opciones_estado = ["PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)"]
        edited_df = st.data_editor(df_monitoreo, column_config={"Estado": st.column_config.SelectboxColumn("Estado de Gestión", options=opciones_estado, required=True), "Sugerencias": st.column_config.TextColumn("Sugerencias", width="large")}, hide_index=True, num_rows="fixed", use_container_width=True)
        if st.button("Guardar Cambios de Estado", type="primary"):
            cambios_guardados = 0
            for original_row in df_monitoreo.itertuples():
                edited_row = edited_df[edited_df['ID'] == original_row.ID].iloc[0]
                if original_row.Estado != edited_row['Estado']:
                    if actualizar_estado_alerta(original_row.ID, edited_row['Estado']):
                        st.success(f"Estado del DNI **{original_row.DNI}** (ID: {original_row.ID}) actualizado a **{edited_row['Estado']}**.")
                        cambios_guardados += 1
            if cambios_guardados > 0:
                st.info(f"Se actualizaron {cambios_guardados} registros. Recargando la vista...")
                st.rerun()
            else:
                st.warning("No se detectaron cambios de estado para guardar.")
    st.markdown("---")
    st.header("2. Reporte Histórico de Registros")
    df_reporte = obtener_todos_los_registros()
    if not df_reporte.empty:
        st.dataframe(df_reporte, use_container_width=True, hide_index=True)
        @st.cache_data
        def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
        csv = convert_df_to_csv(df_reporte)
        st.download_button(label="⬇️ Descargar Reporte Completo (CSV)", data=csv, file_name=f'reporte_historico_alertas_{datetime.date.today().isoformat()}.csv', mime='text/csv')
    else:
        st.info("No hay registros históricos en la base de datos.")

# -------------------------
# Navegación principal
# -------------------------
def main():
    st.sidebar.title("🩸 Menú MIDIS Anemia")
    diagnostico_conexion_ui()
    opcion_seleccionada = st.sidebar.radio("Selecciona una vista:", ["📝 Generar Informe (Predicción)", "📊 Monitoreo y Reportes"])
    if opcion_seleccionada == "📝 Generar Informe (Predicción)":
        vista_prediccion()
    elif opcion_seleccionada == "📊 Monitoreo y Reportes":
        vista_monitoreo()

if __name__ == "__main__":
    main()
