# =============================================================================
# 📊 SISTEMA DE DIAGNÓSTICO DE RIESGO DE ANEMIA (Versión IPv4 Fix)
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import os
import socket
from supabase import create_client, Client

# =============================================================================
# 🔐 CONFIGURACIÓN DE SUPABASE
# =============================================================================
@st.cache_resource
def init_supabase_client() -> Client:
    """Inicializa el cliente de Supabase con IPv4 forzado y validación de conexión."""
    try:
        # 1️⃣ Leer credenciales desde st.secrets o variables de entorno
        url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")

        if not url or not key:
            st.error("⚠️ ERROR: No se encontraron las credenciales de Supabase (URL o KEY).")
            return None

        # 2️⃣ Forzar uso de IPv4 (evita error Errno -2)
        domain = url.replace("https://", "").replace("/", "")
        try:
            ipv4_info = socket.getaddrinfo(domain, None, socket.AF_INET)
            ipv4_address = ipv4_info[0][4][0]
            ipv4_url = url.replace(domain, ipv4_address)
        except Exception as dns_error:
            st.warning(f"No se pudo resolver IPv4 ({dns_error}), usando URL original.")
            ipv4_url = url

        # 3️⃣ Crear cliente
        supabase: Client = create_client(ipv4_url, key)

        # 4️⃣ Prueba rápida de conexión
        try:
            supabase.table("alertas").select("dni").limit(1).execute()
            st.success("✅ Conexión a Supabase establecida correctamente (IPv4).")
            return supabase
        except Exception as api_e:
            st.error(f"❌ Error al conectar con Supabase REST API: {api_e}")
            return None

    except Exception as e:
        st.error(f"⚠️ Error inesperado al inicializar Supabase: {e}")
        return None


# =============================================================================
# ⚙️ CARGA DE MODELO DE PREDICCIÓN
# =============================================================================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("modelo_anemia.pkl")
        st.success("Modelo de predicción cargado correctamente.")
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


# =============================================================================
# 🩺 INTERFAZ PRINCIPAL DE LA APLICACIÓN
# =============================================================================
def main():
    st.title("🩸 Informe Personalizado y Diagnóstico de Riesgo de Anemia (v2.1 Híbrida)")
    
    # Inicializar conexión Supabase
    supabase = init_supabase_client()
    modelo = cargar_modelo()

    # Si no hay conexión o modelo, no continuar
    if supabase is None or modelo is None:
        st.stop()

    # Sección de entrada de datos
    st.header("1️⃣ Factores Clínicos y Demográficos")
    edad = st.number_input("Edad (años)", min_value=0, max_value=99, value=10)
    hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=0.0, max_value=25.0, value=12.5)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
    zona_rural = st.selectbox("¿Vive en zona rural?", ["Sí", "No"])
    agua_segura = st.selectbox("¿Tiene acceso a agua segura?", ["Sí", "No"])
    saneamiento = st.selectbox("¿Cuenta con saneamiento básico?", ["Sí", "No"])
    programa = st.selectbox("¿Recibe programa nutricional?", ["Sí", "No"])
    suplementos = st.selectbox("¿Recibió suplementos de hierro?", ["Sí", "No"])

    if st.button("🔍 Generar Diagnóstico"):
        try:
            entrada = pd.DataFrame({
                "Edad": [edad],
                "Hemoglobina": [hemoglobina],
                "Sexo": [sexo],
                "Zona_Rural": [zona_rural],
                "Agua_Segura": [agua_segura],
                "Saneamiento": [saneamiento],
                "Programa": [programa],
                "Suplementos": [suplementos]
            })

            # Realizar predicción
            prediccion = modelo.predict(entrada)[0]
            riesgo = "ALTO" if prediccion == 1 else "BAJO"

            st.success(f"🩸 Riesgo estimado de anemia: **{riesgo}**")

            # Guardar en Supabase (si la conexión existe)
            if supabase:
                supabase.table("alertas").insert({
                    "dni": "0000",
                    "edad": edad,
                    "hemoglobina": hemoglobina,
                    "riesgo": riesgo
                }).execute()
                st.info("Registro guardado en Supabase correctamente ✅")

        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")


if __name__ == "__main__":
    main()

