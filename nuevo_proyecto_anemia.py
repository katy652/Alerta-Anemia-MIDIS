# ====================================================================
# SCRIPT PARA ENTRENAR Y GUARDAR LOS ARCHIVOS .JOBLIB
# Nombre del archivo: entrenar_modelo.py
# ====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import unidecode

def limpiar_texto(texto):
    """Función para normalizar texto categórico."""
    if pd.isna(texto):
        return 'desconocido'
    # Convierte a minúsculas, elimina tildes y espacios
    return unidecode.unidecode(str(texto).strip().lower())

# 1. Cargar datos
try:
    # Asegúrate de que este nombre coincida con tu archivo CSV
    df = pd.read_csv('data_anemia_final_midis.csv')
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print("ERROR: Asegúrate de que 'data_anemia_final_midis.csv' está en el mismo directorio.")
    exit()

# 2. Filtrado y Mapeo
# Filtrar solo casos con riesgo definido y mapear la variable objetivo a números
df = df[df['Riesgo_Anemia'].isin(['Alto', 'Medio', 'Bajo'])].copy()
risk_map = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
df['riesgo_target'] = df['Riesgo_Anemia'].map(risk_map)

# Seleccionar Features (Asegurando que coincidan con el formulario)
features = ['Edad_meses', 'Sexo', 'Region', 'Area', 'Clima', 'Altitud_m',
            'Programa_QaliWarma', 'Programa_Juntos', 'Programa_VasoLeche',
            'Suplemento_Hierro', 'Ingreso_Familiar_Soles', 
            'Nivel_Educacion_Madre', 'Nro_Hijos', 'Hemoglobina_g_dL']

df_features = df[features].copy()

# 3. Limpieza de texto categórico
cols_categoricas = ['Sexo', 'Region', 'Area', 'Clima', 'Nivel_Educacion_Madre', 
                    'Programa_QaliWarma', 'Programa_Juntos', 'Programa_VasoLeche', 
                    'Suplemento_Hierro']

for col in cols_categoricas:
    if col in df_features.columns:
        df_features[col] = df_features[col].apply(limpiar_texto)

# 4. One-Hot Encoding
df_encoded = pd.get_dummies(df_features, drop_first=True)

# 5. Entrenar el Modelo (RandomForest)
X = df_encoded
y = df['riesgo_target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Modelo de IA entrenado exitosamente.")

# 6. Guardar el Modelo y las Columnas (¡CRÍTICO!)
joblib.dump(model, 'modelo_anemia.joblib')
joblib.dump(X.columns.tolist(), 'modelo_columns.joblib')

print("Archivos 'modelo_anemia.joblib' y 'modelo_columns.joblib' creados y guardados.")