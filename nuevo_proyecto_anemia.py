# Simulación de la función que interactúa con datos y genera el error.
# El error original era: AttributeError: Can only use .dt accessor with datetime-like values

import pandas as pd
import streamlit as st
from datetime import datetime

# --- FUNCIONES DE EJEMPLO PARA ILUSTRAR LA CORRECCIÓN ---

def obtener_datos_simulados():
    """Simula la carga de datos donde la columna de fecha podría ser un string (object)."""
    data = {
        'ID_Paciente': [1, 2, 3],
        'Hemoglobina': [10.5, 12.0, 9.8],
        # Nota: La fecha es un string, lo que causa el error si se usa .dt directamente.
        'fecha Alerta Str': ['2025-10-01', '2025-10-05', '2025-10-10'] 
    }
    df = pd.DataFrame(data)
    return df

def obtener_alertas_pendientes_o_seguimiento_CORREGIDA(df_monitoreo):
    """
    Función corregida para manejar el error de tipo de datos.

    La corrección clave es: pd.to_datetime(df_monitoreo['fecha Alerta Str'])

    """
    
    st.info("Intentando obtener y formatear fechas...")
    
    # --- CORRECCIÓN CLAVE: CONVERTIR A TIPO DATETIME ---
    # Esto soluciona el error: "Can only use .dt accessor with datetime-like values"
    try:
        df_monitoreo['fecha Alerta Str'] = pd.to_datetime(df_monitoreo['fecha Alerta Str'], format='%Y-%m-%d', errors='coerce')
        st.success("Columna 'fecha Alerta Str' convertida correctamente a formato datetime.")
    except Exception as e:
        st.error(f"Error al convertir la columna de fecha: {e}")
        return pd.DataFrame() # Retorna vacío si falla la conversión

    # --- LÍNEA QUE CAUSABA EL ERROR ORIGINAL (Ahora funciona) ---
    # Usamos .dt.strftime('%d-%m-%Y') para formatear la fecha como se hacía en tu código.
    df_monitoreo['fecha Formateada'] = df_monitoreo['fecha Alerta Str'].dt.strftime('%d-%m-%Y')

    st.write("Datos con fecha formateada:")
    st.dataframe(df_monitoreo[['ID_Paciente', 'fecha Alerta Str', 'fecha Formateada']])
    
    return df_monitoreo

# --- SIMULACIÓN DE LA VISTA (Función main de Streamlit) ---

def vista_monitoreo_simulada():
    st.header("1. Casos de Monitoreo Activo (Pendientes y En Seguimiento) - (Simulación)")
    
    # 1. Obtener los datos (donde las fechas vienen como strings)
    df_monitoreo_simulado = obtener_datos_simulados()
    
    # 2. Llamar a la función con la CORRECCIÓN
    df_alertas = obtener_alertas_pendientes_o_seguimiento_CORREGIDA(df_monitoreo_simulado)
    
    if not df_alertas.empty:
        st.subheader("Resultado después del Formato")
        # En tu app real, aquí iría la lógica para mostrar las tablas, etc.
        st.dataframe(df_alertas)
    else:
        st.warning("No se pudieron cargar o procesar los datos de alerta.")
        
def main():
    st.set_page_config(layout="wide", page_title="Monitoreo de Anemia - FIX")
    st.title("✅ Corrección de ERROR - Predicción de Anemia")
    vista_monitoreo_simulada()

if __name__ == "__main__":
    main()
