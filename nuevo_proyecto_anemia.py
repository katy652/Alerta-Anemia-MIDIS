import streamlit as st
import pandas as pd
import joblib
import unidecode
import datetime
import sqlite3
import json
import os
from fpdf import FPDF
from supabase import create_client, Client

# ==============================================================================
# 1. CONFIGURACIÓN DE LA CONEXIÓN SUPABASE
# ==============================================================================
@st.cache_resource
def init_supabase_client() -> Client:
    """Inicializa el cliente de Supabase usando st.secrets (para Streamlit Cloud) o entorno local."""
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        else:
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            st.error("❌ Credenciales de Supabase no encontradas. Define SUPABASE_URL y SUPABASE_KEY en Secrets.")
            return None

        supabase: Client = create_client(url, key)
        st.success("✅ Conexión con Supabase inicializada correctamente.")
        return supabase
    except Exception as e:
        st.error(f"❌ Error al conectar con Supabase: {e}")
        return None


# ==============================================================================
# 2. FUNCIÓN PARA GUARDAR ALERTAS EN SUPABASE
# ==============================================================================
def guardar_alerta_supabase(alerta: dict):
    """Inserta un registro en la tabla 'alertas' de Supabase."""
    supabase = init_supabase_client()
    if supabase is None:
        st.error("No se pudo conectar con Supabase.")
        return False

    try:
        response = supabase.table("alertas").insert(alerta).execute()
        if response.data:
            st.success("✅ Alerta registrada exitosamente en Supabase.")
            return True
        else:
            st.warning("⚠️ No se insertó el registro. Verifica la estructura de la tabla.")
            return False
    except Exception as e:
        st.error(f"❌ Error al guardar en Supabase: {e}")
        return False


# ==============================================================================
# 3. FUNCIÓN PARA GENERAR PDF
# ==============================================================================
def generar_pdf(nombre, edad, hemoglobina, riesgo, fecha, sugerencias):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Reporte de Alerta de Anemia", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Nombre y Apellido: {nombre}", ln=True)
    pdf.cell(0, 10, f"Edad: {edad} años", ln=True)
    pdf.cell(0, 10, f"Hemoglobina: {hemoglobina} g/dL", ln=True)
    pdf.cell(0, 10, f"Riesgo detectado: {riesgo}", ln=True)
    pdf.cell(0, 10, f"Fecha de evaluación: {fecha}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Sugerencias: {sugerencias}")
    filename = f"alerta_anemia_{nombre.replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename


# ==============================================================================
# 4. VISTA: PREDICCIÓN Y REGISTRO DE CASOS
# ==============================================================================
def vista_prediccion():
    st.header("📋 Evaluación y Registro de Riesgo de Anemia")

    with st.form("form_anemia"):
        nombre_apellido = st.text_input("Nombre y Apellido")
        edad = st.number_input("Edad (años)", min_value=0, max_value=12, value=2)
        hemoglobina = st.number_input("Nivel de Hemoglobina (g/dL)", min_value=0.0, max_value=20.0, value=10.5)
        dni = st.text_input("DNI del menor")
        submitted = st.form_submit_button("Evaluar Riesgo")

    if submitted:
        # Evaluación simple según umbral
        riesgo = "Alto" if hemoglobina < 11 else "Bajo"
        fecha_alerta = datetime.date.today().strftime("%Y-%m-%d")

        # Generar sugerencias
        if riesgo == "Alto":
            sugerencias = (
                "Prioridad Alta - Riesgo de Anemia: Derivar a establecimiento de salud para evaluación "
                "y tratamiento inmediato. Registrar seguimiento."
            )
        else:
            sugerencias = (
                "Control regular y mantener dieta rica en hierro, vitaminas y proteínas. "
                "Próximo control en 3 meses."
            )

        st.subheader("Resultado del Análisis")
        st.write(f"**Riesgo Detectado:** {riesgo}")
        st.write(f"**Sugerencias:** {sugerencias}")

        # Guardar alerta en Supabase
        alerta = {
            "dni": dni,
            "nombre_apellido": nombre_apellido,
            "edad": edad,
            "hemoglobina_g": hemoglobina,
            "riesgo": riesgo,
            "fecha_alerta": fecha_alerta,
            "estado": "Pendiente",
            "sugerencias": sugerencias
        }
        guardar_alerta_supabase(alerta)

        # Generar PDF
        pdf_file = generar_pdf(nombre_apellido, edad, hemoglobina, riesgo, fecha_alerta, sugerencias)
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Descargar Reporte PDF", f, file_name=pdf_file)


# ==============================================================================
# 5. VISTA: MONITOREO Y SEGUIMIENTO DE CASOS
# ==============================================================================
def vista_monitoreo():
    st.header("📊 Monitoreo y Seguimiento de Casos de Anemia")

    supabase = init_supabase_client()
    if supabase is None:
        st.error("No se pudo conectar con Supabase.")
        return

    try:
        data = supabase.table("alertas").select("*").execute()
        if not data.data:
            st.info("No hay registros de alertas aún.")
            return

        df = pd.DataFrame(data.data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Permitir actualización de estado
        st.markdown("### 🔄 Actualizar estado de atención")
        edited_df = st.data_editor(
            df[["id", "nombre_apellido", "riesgo", "estado"]],
            num_rows="dynamic",
            key="edicion_alertas"
        )

        if st.button("Guardar Cambios de Estado", type="primary"):
            cambios_guardados = 0
            for _, row in edited_df.iterrows():
                response = supabase.table("alertas").update({"estado": row["estado"]}).eq("id", row["id"]).execute()
                if response.data:
                    cambios_guardados += 1
            if cambios_guardados > 0:
                st.success(f"✅ Se actualizaron {cambios_guardados} registros correctamente.")
                st.rerun()
            else:
                st.warning("⚠️ No se realizaron cambios o hubo errores durante la actualización.")

    except Exception as e:
        st.error(f"❌ Error al recuperar registros: {e}")


# ==============================================================================
# 6. NAVEGACIÓN PRINCIPAL
# ==============================================================================
st.sidebar.title("🩸 Sistema de Monitoreo de Anemia (MIDIS - IA)")
opcion = st.sidebar.radio(
    "Selecciona una vista:",
    ["📋 Evaluación y Registro", "📊 Monitoreo de Casos"]
)

if opcion == "📋 Evaluación y Registro":
    vista_prediccion()
elif opcion == "📊 Monitoreo de Casos":
    vista_monitoreo()
