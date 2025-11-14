import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { 
  getFirestore, 
  doc, 
  collection, 
  addDoc, 
  updateDoc, 
  onSnapshot, 
  query, 
  where, 
  Timestamp 
} from 'firebase/firestore';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Info, AlertTriangle, CheckCircle, Save, Download, RefreshCw, Smartphone, BarChart as BarChartIcon } from 'lucide-react';
import { twilioClient } from './twilio_mock'; // Mock de Twilio para simular el envío.

// ==============================================================================
// 1. CONFIGURACIÓN INICIAL Y CLIENTE FIREBASE
// ==============================================================================

// Variables globales proporcionadas por el entorno (¡OBLIGATORIAS!)
const appId = typeof __app_id !== 'undefined' ? __app_id : 'midis-anemia-app-default';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};
const initialAuthToken = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;

// Mock de Twilio (si no se proporcionan credenciales reales)
const mockTwilio = {
  sendSMS: (to, body) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log(`[SMS SIMULADO ENVIADO] A: ${to} | Mensaje: ${body}`);
        resolve({ success: true, message: "Simulación de SMS exitosa." });
      }, 500);
    });
  }
};

// ==============================================================================
// 2. LÓGICA CORE Y MOCKS (Igual que en Python, pero en JS)
// ==============================================================================

const REGIONES_PERU = [
    "LIMA (Metropolitana y Provincia)", "CALLAO (Provincia Constitucional)",
    "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ICA", "TUMBES", "ÁNCASH (Costa)",
    "HUÁNUCO", "JUNÍN (Andes)", "CUSCO (Andes)", "AYACUCHO", "APURÍMAC",
    "CAJAMARCA", "AREQUIPA", "MOQUEGUA", "TACNA",
    "PUNO (Sierra Alta)", "HUANCAVELICA (Sierra Alta)", "PASCO",
    "LORETO", "AMAZONAS", "SAN MARTÍN", "UCAYALI", "MADRE DE DIOS",
    "OTRO / NO ESPECIFICADO"
];

const getAltitudPorRegion = (region) => {
    const altitudes = {
        "PUNO (Sierra Alta)": 3820, "HUANCAVELICA (Sierra Alta)": 3676, "PASCO": 4330,
        "JUNÍN (Andes)": 3271, "CUSCO (Andes)": 3399, "AYACUCHO": 2761,
        "APURÍMAC": 2900, "CAJAMARCA": 2750, "AREQUIPA": 2335,
        "MOQUEGUA": 1410, "TACNA": 562, "HUÁNUCO": 1894,
        "ÁNCASH (Costa)": 50, "LIMA (Metropolitana y Provincia)": 150,
        "CALLAO (Provincia Constitucional)": 10, "PIURA": 30, "LAMBAYEQUE": 50,
        "LA LIBERTAD": 150, "ICA": 406, "TUMBES": 50,
        "LORETO": 106, "AMAZONAS": 500, "SAN MARTÍN": 500, "UCAYALI": 154, "MADRE DE DIOS": 200,
        "OTRO / NO ESPECIFICADO": 150
    };
    return altitudes[region] || 150;
};

const getClimaPorRegion = (region) => {
    const r = region.toUpperCase();
    if (r.includes('SIERRA ALTA') || r.includes('PUNO') || r.includes('PASCO') || r.includes('HUANCAVELICA')) return "FRÍO";
    if (r.includes('ANDES') || r.includes('AYACUCHO') || r.includes('CAJAMARCA')) return "TEMPLADO";
    if (r.includes('LORETO') || r.includes('UCAYALI') || r.includes('AMAZONAS') || r.includes('MADRE DE DIOS') || r.includes('SAN MARTÍN')) return "CÁLIDO/HÚMEDO";
    return "CÁLIDO/SECO";
};

const clasificarAnemiaClinica = (hemoglobina, edad_meses, altitud_m) => {
    // Factor de corrección por altitud (suma a Hb) = 0.3 * (Altitud en km)
    const correccionAlt = 0.3 * (altitud_m / 1000);
    const hbCorregida = hemoglobina + correccionAlt;
    
    // Umbrales (Hb corregida, g/dL) para niños 12–59 meses
    const umbralAnemia = 11.0; 
    const umbralModerada = 10.0;
    const umbralSevera = 7.0;
    
    let gravedad = "NO ANÉMICO";
    if (hbCorregida < umbralSevera) gravedad = "SEVERA";
    else if (hbCorregida < umbralModerada) gravedad = "MODERADA";
    else if (hbCorregida < umbralAnemia) gravedad = "LEVE";
        
    return { gravedad, umbralAnemia, hbCorregida, correccionAlt };
};

const predictRiskML = (data, gravedad_anemia) => {
    // --- MOCK / SIMULACIÓN DE MODELO ML ---
    let probBase = 0.1; 
    
    // Factores de aumento de riesgo (por IA simulada)
    if (data.Area === 'Rural') probBase += 0.15;
    if (['Sin Nivel', 'Inicial', 'Primaria'].includes(data.Nivel_Educacion_Madre)) probBase += 0.2;
    if (data.Ingreso_Familiar_Soles < 1000) probBase += 0.25;
    if (data.Nro_Hijos >= 4) probBase += 0.1;
    if (data.Suplemento_Hierro === 'No') probBase += 0.15;
    
    // Ajuste por gravedad clínica (dominante en el sistema híbrido)
    if (gravedad_anemia === 'SEVERA') probBase = 0.99;
    else if (gravedad_anemia === 'MODERADA') probBase = Math.max(probBase, 0.75);
    else if (gravedad_anemia === 'LEVE') probBase = Math.max(probBase, 0.45);
        
    const probRiesgo = Math.min(0.99, probBase + (Math.random() * 0.1 - 0.05)); // +/- 5% random
    
    let resultadoML;
    if (probRiesgo >= 0.7) resultadoML = "ALTO RIESGO (Predicción ML)";
    else if (probRiesgo >= 0.4) resultadoML = "MEDIO RIESGO (Predicción ML)";
    else resultadoML = "BAJO RIESGO (Predicción ML)";
        
    return { probRiesgo, resultadoML };
};

const generarSugerencias = (data, resultadoFinal, gravedadAnemia) => {
    const sugerencias = [];
    
    // 1. Sugerencias Clínicas
    if (gravedadAnemia === 'SEVERA') {
        sugerencias.push("🚨🚨 Requerimiento Inmediato: Hospitalización y Transfusión de Sangre si la indicación clínica lo amerita. Contacto Urgente con UCI Pediátrica. | CRÍTICO | Atención Hospitalaria");
    } else if (gravedadAnemia === 'MODERADA') {
        sugerencias.push("🔴 Seguimiento Clínico Urgente: Dosis terapéutica de Hierro por 6 meses y reevaluación mensual de Hemoglobina. Consulta con Hematología. | CRÍTICO | Suplementación Reforzada");
    } else if (gravedadAnemia === 'LEVE') {
        sugerencias.push("⚠️ Suplementación Inmediata: Dosis profiláctica o terapéutica inicial de Hierro por 4 meses. Control en 30 días. | ALERTA | Suplementación");
    } else {
        sugerencias.push("✅ Vigilancia Activa: El valor corregido de Hb es óptimo. Continuar con chequeos regulares y prevención primaria. | Ok | Preventivo");
    }

    // 2. Suplementación y Dieta
    if (data.Suplemento_Hierro === 'No') {
        sugerencias.push("💊 Suplementación: Iniciar o asegurar la adherencia al suplemento de Hierro (gotas/jarabe) según la edad (MINSA). | Suplemento");
    }
    if (data.Edad_meses < 24) {
        sugerencias.push("👶 Edad Crítica: Reforzar la alimentación complementaria rica en hierro hemo (sangrecita, hígado, bazo) debido a la edad vulnerable (6 a 24 meses). | Dieta | Edad");
    }
    sugerencias.push("🍲 Nutrición: Incluir alimentos fortificados y menús ricos en hierro y vitamina C (para absorción). Énfasis en proteínas de origen animal. | Dieta");

    // 3. Socioeconómicas/Contextuales
    if (data.Ingreso_Familiar_Soles < 1000) {
        sugerencias.push("💰 Apoyo Social: Evaluar la elegibilidad para programas de apoyo económico (Juntos) o alimentario (Vaso de Leche, Qali Warma) si no está inscrito. | Social | Económico");
    }
    if (data.Area === 'Rural') {
        sugerencias.push("📚 Educación: Sesiones educativas sobre preparación de alimentos ricos en hierro, higiene y desparasitación adaptadas al contexto rural. | Educación | Contextual");
    }
    if (['Primaria', 'Sin Nivel'].includes(data.Nivel_Educacion_Madre)) {
        sugerencias.push("📚 Intervención: Materiales educativos con lenguaje simple y demostraciones prácticas de cocina/higiene. | Educación | Vulnerabilidad");
    }
    
    // 4. Geográficas
    if (data.Clima === 'FRÍO') {
        sugerencias.push("✨ Clima Frío: Reforzar la vigilancia de infecciones respiratorias agudas (IRAs), ya que el frío aumenta el gasto energético y el riesgo nutricional. | General | Contextual");
    }
    
    sugerencias.unshift(`Diagnóstico Híbrido: ${resultadoFinal}`);
    
    return sugerencias;
};

// ==============================================================================
// 3. COMPONENTE PRINCIPAL (APP)
// ==============================================================================

const App = () => {
    // Estados de Firebase y Autenticación
    const [db, setDb] = useState(null);
    const [auth, setAuth] = useState(null);
    const [userId, setUserId] = useState(null);
    const [isAuthReady, setIsAuthReady] = useState(false);
    
    // Estados de Datos y UI
    const [alertas, setAlertas] = useState([]);
    const [view, setView] = useState('Predicción y Reporte');
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState(null);
    const [error, setError] = useState(null);
    const [predictionResult, setPredictionResult] = useState(null);

    // ------------------------------------
    // Efecto 1: Inicialización de Firebase y Autenticación
    // ------------------------------------
    useEffect(() => {
        if (!firebaseConfig || Object.keys(firebaseConfig).length === 0) {
            setError("Error Crítico: La configuración de Firebase no está disponible. No se puede garantizar la persistencia de datos.");
            setLoading(false);
            return;
        }

        try {
            const app = initializeApp(firebaseConfig);
            const firestoreDb = getFirestore(app);
            const firebaseAuth = getAuth(app);
            setDb(firestoreDb);
            setAuth(firebaseAuth);

            // 1. Configurar Listener de Auth
            const unsubscribe = onAuthStateChanged(firebaseAuth, async (user) => {
                if (user) {
                    setUserId(user.uid);
                } else {
                    // Si no hay usuario, intenta iniciar sesión anónimamente
                    try {
                        await signInAnonymously(firebaseAuth);
                        // El listener onAuthStateChanged se disparará de nuevo con el usuario anónimo
                    } catch (e) {
                        console.error("Fallo al iniciar sesión anónimamente:", e);
                        // Si falla la anónima, usamos un ID aleatorio (solo para contexto de ruta de DB)
                        setUserId(crypto.randomUUID());
                    }
                }
                setIsAuthReady(true);
                setLoading(false);
            });

            // 2. Intentar usar el token de autenticación personalizado (para usuarios logueados)
            if (initialAuthToken) {
                signInWithCustomToken(firebaseAuth, initialAuthToken).catch(e => {
                    console.error("Fallo al iniciar sesión con Custom Token:", e);
                    // Si el token falla, el listener de onAuthStateChanged intentará anónimo.
                });
            }

            return () => unsubscribe();
        } catch (e) {
            setError(`Error al inicializar Firebase: ${e.message}`);
            setLoading(false);
        }
    }, [initialAuthToken]);
    
    // ------------------------------------
    // Efecto 2: Listener de Firestore para obtener las alertas
    // ------------------------------------
    useEffect(() => {
        if (!db || !isAuthReady) return; // Esperar a que Firebase y Auth estén listos

        // Usamos la colección pública para que todos los usuarios vean los casos
        const collectionPath = `artifacts/${appId}/public/data/alertas_anemia`;
        const alertasRef = collection(db, collectionPath);
        
        // Queremos todas las alertas, ordenadas por fecha (más reciente primero)
        const q = query(alertasRef); 

        const unsubscribe = onSnapshot(q, (snapshot) => {
            const fetchedAlerts = [];
            snapshot.forEach((doc) => {
                const data = doc.data();
                // Convertir Timestamp a Date/ISO string para uso en React
                const fechaAlerta = data['Fecha Alerta'] instanceof Timestamp ? data['Fecha Alerta'].toDate().toISOString() : data['Fecha Alerta'];
                fetchedAlerts.push({
                    id: doc.id, // ¡ID de documento de Firestore para las actualizaciones!
                    ...data,
                    'Fecha Alerta': fechaAlerta,
                });
            });
            // Ordenar por fecha de alerta (más nuevo primero)
            fetchedAlerts.sort((a, b) => new Date(b['Fecha Alerta']) - new Date(a['Fecha Alerta']));
            setAlertas(fetchedAlerts);
        }, (err) => {
            setError(`Error al obtener alertas de Firestore: ${err.message}`);
        });

        return () => unsubscribe();
    }, [db, isAuthReady]);

    // ------------------------------------
    // Manejadores de CRUD (Create, Update)
    // ------------------------------------
    
    const registrarAlertaDB = async (alertaData) => {
        if (!db || !userId) return setError("Base de datos no disponible o usuario no autenticado.");
        
        const dataToInsert = {
            DNI: alertaData.DNI,
            Nombre: alertaData.Nombre_Apellido,
            'Hb Inicial': alertaData.Hemoglobina_g_dL,
            Riesgo: alertaData.riesgo,
            Gravedad: alertaData.gravedad_anemia,
            Region: alertaData.Region,
            Estado: 'REGISTRADO', // Estado inicial
            Sugerencias: alertaData.sugerencias.join(' | '),
            'Fecha Alerta': Timestamp.fromDate(new Date()),
            Usuario_Registro: userId,
        };
        
        try {
            const collectionPath = `artifacts/${appId}/public/data/alertas_anemia`;
            const docRef = await addDoc(collection(db, collectionPath), dataToInsert);
            setMessage({ type: 'success', text: `✅ Caso de ${alertaData.Nombre_Apellido} registrado permanentemente en Firestore con ID: ${docRef.id}` });
            return docRef.id;
        } catch (e) {
            setError(`❌ Error al registrar en Firestore: ${e.message}`);
            return null;
        }
    };
    
    const actualizarEstadoAlerta = async (docId, nuevoEstado) => {
        if (!db) return false;
        
        try {
            const collectionPath = `artifacts/${appId}/public/data/alertas_anemia`;
            const docRef = doc(db, collectionPath, docId);
            await updateDoc(docRef, { Estado: nuevoEstado });
            return true;
        } catch (e) {
            setError(`❌ Error al actualizar estado en Firestore (ID: ${docId}): ${e.message}`);
            return false;
        }
    };

    // ------------------------------------
    // Enviar Alerta (Twilio Mock)
    // ------------------------------------
    
    const enviarAlertaSMS = (celular, nombre, dni, riesgo, gravedad) => {
        const mensaje = `ALERTA MIDIS: Caso ${nombre} (DNI ${dni}) clasificado como ${riesgo} y Gravedad ${gravedad}. REQUIERE ACCIÓN URGENTE. Monitoreo: ${window.location.href}`;
        
        // Usamos el Mock, ya que no podemos garantizar Twilio real en este entorno
        mockTwilio.sendSMS(celular, mensaje)
            .then(() => setMessage({ type: 'info', text: `📲 Alerta SMS SIMULADA enviada al número: ${celular}.` }))
            .catch((e) => setError(`❌ Fallo en simulación SMS: ${e.message}`));
    };
    
    // ------------------------------------
    // Renderizado Común
    // ------------------------------------
    
    const renderMessage = () => {
        if (error) return (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg shadow-md mb-6" role="alert">
                <p className="font-bold">Error:</p>
                <p>{error}</p>
            </div>
        );
        if (message) return (
            <div className={`bg-${message.type}-100 border-l-4 border-${message.type === 'success' ? 'green' : message.type === 'info' ? 'blue' : 'yellow'}-500 text-${message.type === 'success' ? 'green' : message.type === 'info' ? 'blue' : 'yellow'}-700 p-4 rounded-lg shadow-md mb-6`} role="alert">
                <p>{message.text}</p>
            </div>
        );
        return null;
    };
    
    if (loading) {
        return <div className="flex justify-center items-center h-screen text-xl text-gray-600">
            <RefreshCw className="animate-spin mr-2" /> Cargando servicios de autenticación y base de datos...
        </div>;
    }

    // ==============================================================================
    // 4. VISTAS
    // ==============================================================================

    // --- VISTA: Predicción y Reporte ---
    const PredictionView = () => {
        const [formData, setFormData] = useState({
            DNI: '', Nombre_Apellido: '', Celular: '', Hemoglobina_g_dL: 10.5, Edad_meses: 36, Region: REGIONES_PERU[0],
            Ingreso_Familiar_Soles: 1800.0, Nro_Hijos: 2, Nivel_Educacion_Madre: "Secundaria", Area: "Urbana", Suplemento_Hierro: "No",
            Programa_QaliWarma: "No", Programa_Juntos: "No", Programa_VasoLeche: "No", Sexo: "Femenino"
        });

        const handleInputChange = (e) => {
            const { name, value, type } = e.target;
            const finalValue = (type === 'number' || name === 'Hemoglobina_g_dL' || name === 'Edad_meses' || name === 'Ingreso_Familiar_Soles' || name === 'Nro_Hijos')
                ? (type === 'number' ? parseFloat(value) : value)
                : value;
            setFormData(prev => ({ ...prev, [name]: finalValue }));
        };

        const handleSubmit = async (e) => {
            e.preventDefault();
            setError(null);
            setMessage(null);
            
            if (!formData.DNI || formData.DNI.length !== 8) return setError("Ingrese un DNI válido de 8 dígitos.");
            if (!formData.Nombre_Apellido) return setError("Ingrese un nombre y apellido.");
            if (!formData.Celular) return setError("Ingrese un número de celular de contacto.");

            const altitudCalculada = getAltitudPorRegion(formData.Region);
            const climaCalculado = getClimaPorRegion(formData.Region);

            const data = { ...formData, Altitud_m: altitudCalculada, Clima: climaCalculado };

            const { gravedad: gravedadAnemia, hbCorregida, correccionAlt } = clasificarAnemiaClinica(data.Hemoglobina_g_dL, data.Edad_meses, altitudCalculada);
            const { probRiesgo, resultadoML } = predictRiskML(data, gravedadAnemia);

            let resultadoFinal;
            if (['SEVERA', 'MODERADA'].includes(gravedadAnemia)) {
                resultadoFinal = `ALTO RIESGO (Alerta Clínica - ${gravedadAnemia})`;
            } else if (resultadoML.startsWith("ALTO RIESGO")) {
                resultadoFinal = `ALTO RIESGO (Predicción ML - Anemia ${gravedadAnemia})`;
            } else {
                resultadoFinal = resultadoML;
            }

            const sugerenciasFinales = generarSugerencias(data, resultadoFinal, gravedadAnemia);
            
            // 1. Registrar en Firestore (Persistencia Real)
            const registroData = {
                DNI: formData.DNI, Nombre_Apellido: formData.Nombre_Apellido, Hemoglobina_g_dL: formData.Hemoglobina_g_dL, 
                Edad_meses: formData.Edad_meses, Region: formData.Region, riesgo: resultadoFinal, gravedad_anemia: gravedadAnemia,
                sugerencias: sugerenciasFinales
            };
            const docId = await registrarAlertaDB(registroData);

            // 2. Enviar Alerta SMS
            if (docId) {
                enviarAlertaSMS(formData.Celular, formData.Nombre_Apellido, formData.DNI, resultadoFinal, gravedadAnemia);
            }
            
            // 3. Mostrar Resultado
            setPredictionResult({ 
                data, resultadoFinal, probRiesgo, gravedadAnemia, sugerenciasFinales, hbCorregida, correccionAlt
            });
        };

        const renderForm = () => (
            <form onSubmit={handleSubmit} className="space-y-6 p-4 md:p-8 bg-white rounded-xl shadow-2xl">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <InputField label="DNI del Paciente (8 dígitos)" name="DNI" value={formData.DNI} onChange={handleInputChange} type="text" maxLength={8} required />
                    <InputField label="Nombre y Apellido" name="Nombre_Apellido" value={formData.Nombre_Apellido} onChange={handleInputChange} type="text" required />
                    <InputField label="Celular de Contacto (Ej: +519XXXXXXXX)" name="Celular" value={formData.Celular} onChange={handleInputChange} type="text" required />
                </div>
                
                <h3 className="text-xl font-semibold border-b pb-2 text-red-700">1. Factores Clínicos y Demográficos Clave</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <NumberField label="Hemoglobina (g/dL) - CRÍTICO" name="Hemoglobina_g_dL" value={formData.Hemoglobina_g_dL} onChange={handleInputChange} min={5.0} max={18.0} step={0.1} required />
                    <NumberField label="Edad (meses) [12-60]" name="Edad_meses" value={formData.Edad_meses} onChange={handleInputChange} min={12} max={60} step={1} required />
                    <SelectField label="Región (Define Altitud y Clima)" name="Region" value={formData.Region} onChange={handleInputChange} options={REGIONES_PERU} required />
                </div>
                <div className="text-sm text-blue-600 bg-blue-50 p-3 rounded-lg flex items-center">
                    <Info className="w-5 h-5 mr-2" />
                    Altitud asignada: **{getAltitudPorRegion(formData.Region)} msnm** | Clima: **{getClimaPorRegion(formData.Region)}**
                </div>

                <h3 className="text-xl font-semibold border-b pb-2 text-red-700">2. Factores Socioeconómicos</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <NumberField label="Ingreso Familiar (Soles/mes)" name="Ingreso_Familiar_Soles" value={formData.Ingreso_Familiar_Soles} onChange={handleInputChange} min={0} max={5000} step={100} required />
                    <NumberField label="Nro. de Hijos en el Hogar" name="Nro_Hijos" value={formData.Nro_Hijos} onChange={handleInputChange} min={1} max={15} step={1} required />
                    <SelectField label="Nivel Educ. Madre" name="Nivel_Educacion_Madre" value={formData.Nivel_Educacion_Madre} onChange={handleInputChange} options={["Secundaria", "Primaria", "Superior Técnica", "Universitaria", "Inicial", "Sin Nivel"]} required />
                    <SelectField label="Área de Residencia" name="Area" value={formData.Area} onChange={handleInputChange} options={['Urbana', 'Rural']} required />
                </div>
                <SelectField label="Sexo del Niño(a)" name="Sexo" value={formData.Sexo} onChange={handleInputChange} options={["Femenino", "Masculino"]} required />


                <h3 className="text-xl font-semibold border-b pb-2 text-red-700">3. Acceso a Programas y Servicios</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <RadioField label="Recibe Suplemento de Hierro" name="Suplemento_Hierro" value={formData.Suplemento_Hierro} onChange={handleInputChange} options={["No", "Sí"]} />
                    <RadioField label="Programa Qali Warma" name="Programa_QaliWarma" value={formData.Programa_QaliWarma} onChange={handleInputChange} options={["No", "Sí"]} />
                    <RadioField label="Programa Juntos" name="Programa_Juntos" value={formData.Programa_Juntos} onChange={handleInputChange} options={["No", "Sí"]} />
                    <RadioField label="Programa Vaso de Leche" name="Programa_VasoLeche" value={formData.Programa_VasoLeche} onChange={handleInputChange} options={["No", "Sí"]} />
                </div>

                <button type="submit" className="w-full py-3 px-4 bg-red-700 hover:bg-red-800 text-white font-bold rounded-xl transition duration-200 shadow-lg flex items-center justify-center">
                    <Save className="w-5 h-5 mr-2" /> GENERAR INFORME PERSONALIZADO Y REGISTRAR CASO
                </button>
            </form>
        );

        const renderResults = () => {
            if (!predictionResult) return null;
            const { resultadoFinal, probRiesgo, gravedadAnemia, sugerenciasFinales, hbCorregida, correccionAlt, data } = predictionResult;
            
            const isHighRisk = resultadoFinal.includes("ALTO RIESGO");
            const alertColor = isHighRisk ? 'bg-red-100 border-red-500 text-red-700' : resultadoFinal.includes("MEDIO RIESGO") ? 'bg-yellow-100 border-yellow-500 text-yellow-700' : 'bg-green-100 border-green-500 text-green-700';

            return (
                <div className="mt-8 p-6 bg-gray-50 rounded-xl shadow-inner">
                    <div className={`p-4 border-l-4 rounded-lg shadow-lg ${alertColor} mb-6 flex items-center`}>
                        {isHighRisk ? <AlertTriangle className="w-8 h-8 mr-3" /> : <CheckCircle className="w-8 h-8 mr-3" />}
                        <h2 className="text-2xl font-extrabold">{resultadoFinal}</h2>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 text-center">
                        <MetricCard label="Hemoglobina Medida (g/dL)" value={data.Hemoglobina_g_dL} />
                        <MetricCard label={`Corrección por Altitud (${data.Altitud_m}m)`} value={`+${correccionAlt.toFixed(1)} g/dL`} color="bg-blue-100" />
                        <MetricCard label="Hemoglobina Corregida (g/dL)" value={hbCorregida.toFixed(1)} delta={`Gravedad: ${gravedadAnemia}`} color={isHighRisk ? 'bg-red-100' : 'bg-green-100'} />
                    </div>
                    <MetricCard label="Prob. de Alto Riesgo por IA" value={`${(probRiesgo * 100).toFixed(2)}%`} color="bg-purple-100" />

                    <h3 className="text-xl font-semibold mt-8 mb-4 border-b pb-2 text-gray-700">📝 Plan de Intervención Oportuna:</h3>
                    <div className="space-y-3">
                        {sugerenciasFinales.map((sug, index) => (
                            <div key={index} className="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                                {sug.replace('|', ' **|** ').split(' | ').map((part, i) => (
                                    <span key={i} className={`mr-2 ${i === 0 ? 'font-medium' : 'text-sm text-gray-600'}`}>
                                        {part}
                                    </span>
                                ))}
                            </div>
                        ))}
                    </div>
                    
                    <div className="mt-8 text-center">
                         <button onClick={() => setPredictionResult(null)} className="py-2 px-4 bg-gray-600 hover:bg-gray-700 text-white font-bold rounded-xl transition duration-200 shadow-md">
                            <RefreshCw className="w-4 h-4 mr-2 inline-block" /> Realizar Nuevo Diagnóstico
                        </button>
                    </div>
                </div>
            );
        };

        return (
            <div className="p-4">
                <h1 className="text-3xl font-bold text-red-800 mb-6">📝 Informe y Diagnóstico de Riesgo de Anemia</h1>
                {predictionResult ? renderResults() : renderForm()}
            </div>
        );
    };

    // --- VISTA: Monitoreo de Alertas ---
    const MonitoringView = () => {
        const [filteredAlerts, setFilteredAlerts] = useState([]);
        const [filterStatus, setFilterStatus] = useState(['REGISTRADO', 'PENDIENTE (CLÍNICO URGENTE)', 'PENDIENTE (IA/VULNERABILIDAD)', 'EN SEGUIMIENTO']);
        const [isSaving, setIsSaving] = useState(false);
        const [changedStates, setChangedStates] = useState({});

        // Filtrar alertas basado en el estado (Alertas Activas)
        useEffect(() => {
            const activeAlerts = alertas.filter(alert => filterStatus.includes(alert.Estado));
            setFilteredAlerts(activeAlerts);
        }, [alertas, filterStatus]);
        
        const handleStateChange = (id, newState) => {
            // Guardar el cambio temporalmente
            setChangedStates(prev => ({ ...prev, [id]: newState }));
        };
        
        const handleSave = async () => {
            setIsSaving(true);
            setError(null);
            setMessage(null);
            let successCount = 0;
            let errorCount = 0;
            
            for (const id in changedStates) {
                const newState = changedStates[id];
                // Encontrar el estado original
                const originalAlert = alertas.find(a => a.id === id);
                if (originalAlert && originalAlert.Estado !== newState) {
                    const success = await actualizarEstadoAlerta(id, newState);
                    if (success) {
                        successCount++;
                    } else {
                        errorCount++;
                    }
                }
            }
            
            if (successCount > 0) setMessage({ type: 'success', text: `✅ Se actualizaron ${successCount} casos en Firestore.` });
            if (errorCount > 0) setError(`❌ Fallo al actualizar ${errorCount} casos. Consulte la consola.`);
            
            setChangedStates({}); // Limpiar cambios
            setIsSaving(false);
        };
        
        const renderActionableAlerts = () => {
            if (filteredAlerts.length === 0) {
                return <div className="p-4 bg-green-100 text-green-700 rounded-lg shadow-md">No hay casos activos pendientes de seguimiento. ✅</div>;
            }
            
            const opcionesEstado = ["REGISTRADO", "PENDIENTE (CLÍNICO URGENTE)", "PENDIENTE (IA/VULNERABILIDAD)", "EN SEGUIMIENTO", "RESUELTO", "CERRADO (NO APLICA)"];

            return (
                <div className="space-y-4">
                    <p className="text-lg font-medium text-blue-700">Casos activos encontrados: {filteredAlerts.length}</p>
                    <div className="overflow-x-auto bg-white rounded-xl shadow-lg">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">DNI / Nombre</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Hb / Riesgo</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Fecha Alerta</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Estado de Gestión</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {filteredAlerts.map((alert) => {
                                    const currentState = changedStates[alert.id] || alert.Estado;
                                    const isChanged = currentState !== alert.Estado;
                                    const riskColor = alert.Riesgo.includes("ALTO") ? 'text-red-600 font-bold' : alert.Riesgo.includes("MEDIO") ? 'text-yellow-600' : 'text-green-600';
                                    
                                    return (
                                        <tr key={alert.id} className={isChanged ? 'bg-yellow-50' : ''}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{alert.id.substring(0, 8)}...</td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="text-sm font-medium text-gray-900">{alert.Nombre}</div>
                                                <div className="text-xs text-gray-500">DNI: {alert.DNI}</div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="text-sm text-gray-900">Hb: {alert['Hb Inicial']} g/dL</div>
                                                <div className={`text-xs ${riskColor}`}>{alert.Riesgo}</div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(alert['Fecha Alerta']).toLocaleString()}</td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <select
                                                    value={currentState}
                                                    onChange={(e) => handleStateChange(alert.id, e.target.value)}
                                                    className={`p-2 border rounded-md text-sm ${isChanged ? 'border-yellow-500 bg-yellow-100' : 'border-gray-300'}`}
                                                >
                                                    {opcionesEstado.map(op => (
                                                        <option key={op} value={op}>{op}</option>
                                                    ))}
                                                </select>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                    <div className="flex justify-end">
                        <button 
                            onClick={handleSave} 
                            disabled={Object.keys(changedStates).length === 0 || isSaving}
                            className={`py-2 px-6 rounded-xl font-bold transition duration-200 shadow-md flex items-center ${Object.keys(changedStates).length === 0 ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700 text-white'}`}
                        >
                            {isSaving ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
                            Guardar {Object.keys(changedStates).length} Cambio(s)
                        </button>
                    </div>
                </div>
            );
        };
        
        const downloadCSV = () => {
            if (alertas.length === 0) return;
            const headers = Object.keys(alertas[0]);
            const csv = [
                headers.join(';'),
                ...alertas.map(row => headers.map(fieldName => {
                    let value = row[fieldName];
                    if (typeof value === 'string' && (value.includes(';') || value.includes('"'))) {
                        value = `"${value.replace(/"/g, '""')}"`;
                    }
                    return value;
                }).join(';'))
            ].join('\n');
            
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.setAttribute('href', url);
            link.setAttribute('download', `historial_alertas_anemia_${new Date().toISOString().slice(0, 10)}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };

        return (
            <div className="p-4">
                <h1 className="text-3xl font-bold text-red-800 mb-6">👁️ Monitoreo y Gestión de Alertas (Firestore)</h1>
                <p className="text-gray-600 mb-4">
                    Los datos se guardan permanentemente en la colección: <span className="font-mono bg-gray-200 p-1 rounded text-sm">artifacts/{appId}/public/data/alertas_anemia</span>
                </p>
                <div className="mb-6 p-4 bg-gray-50 rounded-xl shadow-inner">
                    <h2 className="text-xl font-semibold border-b pb-2 mb-4 text-red-700">Casos Activos de Seguimiento</h2>
                    <div className="flex items-center space-x-4 mb-4">
                         <label className="font-medium text-gray-700">Filtrar Estados Activos:</label>
                         <select 
                            multiple
                            value={filterStatus}
                            onChange={(e) => setFilterStatus(Array.from(e.target.selectedOptions, option => option.value))}
                            className="p-2 border rounded-md text-sm shadow-sm"
                         >
                            {opcionesEstado.map(op => (
                                <option key={op} value={op}>{op}</option>
                            ))}
                         </select>
                    </div>
                    {renderActionableAlerts()}
                </div>

                <div className="p-4 bg-gray-50 rounded-xl shadow-inner mt-8">
                    <h2 className="text-xl font-semibold border-b pb-2 mb-4 text-red-700">Historial Completo de Registros ({alertas.length} Casos)</h2>
                    <button onClick={downloadCSV} disabled={alertas.length === 0} className="py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl transition duration-200 shadow-md flex items-center">
                        <Download className="w-4 h-4 mr-2" /> Descargar Historial Completo (CSV)
                    </button>
                    <div className="mt-4 overflow-x-auto max-h-96">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50 sticky top-0">
                                <tr>
                                    {alertas.length > 0 && Object.keys(alertas[0]).map(key => (
                                        <th key={key} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{key.replace('_', ' ')}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {alertas.map((alert) => (
                                    <tr key={alert.id}>
                                        {Object.entries(alert).map(([key, value]) => (
                                            <td key={key} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 max-w-xs overflow-hidden text-ellipsis" title={value}>
                                                {typeof value === 'number' ? value.toFixed(2) : String(value)}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        );
    };
    
    // --- VISTA: Dashboard ---
    const DashboardView = () => {
        const [filterRegion, setFilterRegion] = useState([]);
        
        const dataForDashboard = useMemo(() => {
            if (alertas.length === 0) return { df_riesgo: [], df_estado: [], df_region_alto: [], df_tendencia: [] };

            const df = filterRegion.length === 0 
                ? alertas 
                : alertas.filter(a => filterRegion.includes(a.Region));
                
            if (df.length === 0) return { df_riesgo: [], df_estado: [], df_region_alto: [], df_tendencia: [] };

            // Agregación de datos
            const countBy = (arr, key) => arr.reduce((acc, curr) => {
                acc[curr[key]] = (acc[curr[key]] || 0) + 1;
                return acc;
            }, {});

            const dfRiesgo = Object.entries(countBy(df, 'Riesgo')).map(([name, value]) => ({ Riesgo: name, Conteo: value }));
            const dfEstado = Object.entries(countBy(df, 'Estado')).map(([name, value]) => ({ Estado: name, Conteo: value }));
            
            const dfRegionAlto = Object.entries(countBy(df.filter(a => a.Riesgo.includes('ALTO RIESGO')), 'Region'))
                .map(([name, value]) => ({ Region: name, 'Casos de Alto Riesgo': value }))
                .sort((a, b) => b['Casos de Alto Riesgo'] - a['Casos de Alto Riesgo'])
                .slice(0, 10); // Top 10

            // Tendencia mensual
            const dfTendenciaMap = df.reduce((acc, curr) => {
                const date = new Date(curr['Fecha Alerta']);
                const monthYear = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
                acc[monthYear] = (acc[monthYear] || 0) + 1;
                return acc;
            }, {});

            const dfTendencia = Object.entries(dfTendenciaMap)
                .map(([name, value]) => ({ 'Mes': name, 'Alertas Registradas': value }))
                .sort((a, b) => new Date(a.Mes) - new Date(b.Mes));


            return { dfRiesgo, dfEstado, dfRegionAlto, dfTendencia };

        }, [alertas, filterRegion]);
        
        const { dfRiesgo, dfEstado, dfRegionAlto, dfTendencia } = dataForDashboard;

        const RIESGO_COLORS = {
            'ALTO RIESGO (Alerta Clínica - SEVERA)': '#D32F2F', // Rojo oscuro
            'ALTO RIESGO (Alerta Clínica - MODERADA)': '#EF5350', // Rojo
            'ALTO RIESGO (Predicción ML - Anemia LEVE)': '#FF7043', // Naranja
            'MEDIO RIESGO (Predicción ML)': '#FFC107', // Amarillo
            'BAJO RIESGO (Predicción ML)': '#4CAF50', // Verde
        };
        
        const ESTADO_COLORS = {
            'PENDIENTE (CLÍNICO URGENTE)': '#D32F2F', 
            'PENDIENTE (IA/VULNERABILIDAD)': '#FFA000', 
            'EN SEGUIMIENTO': '#1976D2', 
            'RESUELTO': '#388E3C', 
            'REGISTRADO': '#757575',
            'CERRADO (NO APLICA)': '#9C27B0'
        };


        if (alertas.length === 0) {
            return <div className="p-4 text-center text-gray-500">No hay datos suficientes para el Dashboard.</div>;
        }

        return (
            <div className="p-4 space-y-8">
                <h1 className="text-3xl font-bold text-red-800 mb-4 flex items-center"><BarChartIcon className="w-6 h-6 mr-2" /> Panel Estadístico de Alertas</h1>

                <div className="p-4 bg-gray-50 rounded-xl shadow-inner">
                    <h2 className="text-xl font-semibold border-b pb-2 mb-4 text-red-700">Filtros</h2>
                    <SelectField 
                        label="Filtrar por Región (Multiselect)" 
                        name="filterRegion" 
                        value={filterRegion} 
                        onChange={(e) => {
                            const options = Array.from(e.target.selectedOptions, option => option.value);
                            setFilterRegion(options);
                        }} 
                        options={REGIONES_PERU} 
                        multiple={true}
                    />
                    <p className="text-sm text-gray-500 mt-2">Mostrando {dfRiesgo.length > 0 ? dfRiesgo.reduce((sum, item) => sum + item.Conteo, 0) : 0} registros filtrados.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Gráfico 1: Distribución de Riesgo (Pie Chart) */}
                    <div className="bg-white p-4 rounded-xl shadow-lg">
                        <h3 className="text-lg font-semibold mb-4 text-center">Distribución de Riesgo</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={dfRiesgo}
                                    dataKey="Conteo"
                                    nameKey="Riesgo"
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={100}
                                    fill="#8884d8"
                                    label
                                >
                                    {dfRiesgo.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={RIESGO_COLORS[entry.Riesgo] || '#A9A9A9'} />
                                    ))}
                                </Pie>
                                <Tooltip formatter={(value) => [`${value} casos`, 'Conteo']} />
                                <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ paddingLeft: '10px' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Gráfico 2: Estado de Gestión (Bar Chart) */}
                    <div className="bg-white p-4 rounded-xl shadow-lg">
                        <h3 className="text-lg font-semibold mb-4 text-center">Estado de Seguimiento de Casos</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={dfEstado} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="Estado" hide />
                                <YAxis />
                                <Tooltip formatter={(value) => [`${value} casos`, 'Conteo']} />
                                <Legend wrapperStyle={{ paddingTop: '10px' }} />
                                <Bar dataKey="Conteo" name="Alertas" fill="#8884d8">
                                    {dfEstado.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={ESTADO_COLORS[entry.Estado] || '#A9A9A9'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Gráfico 3: Tendencia Mensual (Line Chart) */}
                <div className="bg-white p-4 rounded-xl shadow-lg">
                    <h3 className="text-lg font-semibold mb-4 text-center">Tendencia Mensual de Alertas</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={dfTendencia} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="Mes" />
                            <YAxis allowDecimals={false} />
                            <Tooltip labelFormatter={(label) => `Mes: ${label}`} formatter={(value) => [`${value} casos`, 'Alertas']} />
                            <Legend />
                            <Line type="monotone" dataKey="Alertas Registradas" stroke="#8884d8" activeDot={{ r: 8 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                
                {/* Gráfico 4: Casos de Alto Riesgo por Región (Bar Chart Horizontal) */}
                <div className="bg-white p-4 rounded-xl shadow-lg">
                    <h3 className="text-lg font-semibold mb-4 text-center">Top 10 Regiones con Casos de Alto Riesgo</h3>
                    <ResponsiveContainer width="100%" height={Math.max(300, dfRegionAlto.length * 40)}>
                         <BarChart 
                            layout="vertical" 
                            data={dfRegionAlto} 
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" allowDecimals={false} />
                            <YAxis dataKey="Region" type="category" width={100} />
                            <Tooltip formatter={(value) => [`${value} casos`, 'Alto Riesgo']} />
                            <Bar dataKey="Casos de Alto Riesgo" fill="#D32F2F" name="Alto Riesgo" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    };

    // ==============================================================================
    // 5. RENDERIZADO PRINCIPAL
    // ==============================================================================
    
    const renderContent = () => {
        switch (view) {
            case 'Monitoreo de Alertas': return <MonitoringView />;
            case 'Panel de control estadístico': return <DashboardView />;
            default: return <PredictionView />;
        }
    };

    return (
        <div className="min-h-screen flex bg-gray-100 font-sans">
            <Sidebar userId={userId} numAlertas={alertas.length} currentView={view} setView={setView} />
            <main className="flex-1 p-6 overflow-y-auto">
                <header className="mb-6">
                    <h1 className="text-4xl font-extrabold text-red-900">Sistema de Gestión de Riesgo de Anemia (MIDIS)</h1>
                    <p className="text-lg text-gray-600">Integración de Diagnóstico Clínico, Modelo IA y Persistencia con Firestore.</p>
                </header>
                {renderMessage()}
                <div className="bg-white rounded-xl shadow-2xl">
                    {renderContent()}
                </div>
            </main>
        </div>
    );
};

// ==============================================================================
// 6. COMPONENTES DE UI REUTILIZABLES
// ==============================================================================

const Sidebar = ({ userId, numAlertas, currentView, setView }) => {
    const navItems = [
        { name: "Predicción y Reporte", icon: Info },
        { name: "Monitoreo de Alertas", icon: AlertTriangle },
        { name: "Panel de control estadístico", icon: BarChartIcon },
    ];
    return (
        <div className="w-64 bg-red-900 text-white p-6 flex flex-col shadow-xl">
            <h2 className="text-2xl font-bold mb-6 border-b border-red-700 pb-2">🩸 Alerta Anemia IA</h2>
            <nav className="flex-grow space-y-2">
                {navItems.map(item => (
                    <button
                        key={item.name}
                        onClick={() => setView(item.name)}
                        className={`w-full text-left py-3 px-4 rounded-xl transition duration-200 flex items-center ${currentView === item.name ? 'bg-red-700 font-bold shadow-lg' : 'hover:bg-red-800'}`}
                    >
                        <item.icon className="w-5 h-5 mr-3" />
                        {item.name}
                    </button>
                ))}
            </nav>
            <div className="mt-8 pt-4 border-t border-red-700 space-y-2 text-sm">
                <p className="font-semibold text-red-300">Estado del Sistema</p>
                <div className="flex items-center text-green-400">
                    <CheckCircle className="w-4 h-4 mr-2" /> Firestore Persistente
                </div>
                <div className="flex items-center text-blue-300">
                    <Smartphone className="w-4 h-4 mr-2" /> SMS Mock Activo
                </div>
                <p className="mt-2 text-yellow-300">
                    💾 Registros Permanentes: <strong>{numAlertas}</strong>
                </p>
                <p className="break-all text-xs opacity-70">
                    <span className="font-mono">ID Usuario:</span> {userId || 'Anonimo'}
                </p>
            </div>
        </div>
    );
};

const InputField = ({ label, name, value, onChange, type = 'text', maxLength, required = false, step, min, max }) => (
    <div>
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">{label}{required && <span className="text-red-500">*</span>}</label>
        <input
            type={type}
            id={name}
            name={name}
            value={value}
            onChange={onChange}
            maxLength={maxLength}
            step={step}
            min={min}
            max={max}
            required={required}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-red-500 focus:border-red-500 sm:text-sm"
        />
    </div>
);

const NumberField = (props) => (
    <InputField {...props} type="number" />
);

const SelectField = ({ label, name, value, onChange, options, required = false, multiple = false }) => (
    <div>
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">{label}{required && <span className="text-red-500">*</span>}</label>
        <select
            id={name}
            name={name}
            value={value}
            onChange={onChange}
            required={required}
            multiple={multiple}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-red-500 focus:border-red-500 sm:text-sm"
            style={multiple ? { height: '100px' } : {}}
        >
            {!multiple && <option value="">Seleccionar...</option>}
            {options.map(option => (
                <option key={option} value={option}>{option}</option>
            ))}
        </select>
    </div>
);

const RadioField = ({ label, name, value, onChange, options }) => (
    <div className="p-3 bg-gray-50 rounded-md shadow-sm">
        <p className="text-sm font-medium text-gray-700 mb-2">{label}</p>
        <div className="flex space-x-4">
            {options.map(option => (
                <label key={option} className="flex items-center">
                    <input
                        type="radio"
                        name={name}
                        value={option}
                        checked={value === option}
                        onChange={onChange}
                        className="focus:ring-red-500 h-4 w-4 text-red-600 border-gray-300"
                    />
                    <span className="ml-2 text-sm text-gray-700">{option}</span>
                </label>
            ))}
        </div>
    </div>
);

const MetricCard = ({ label, value, delta, color = 'bg-white' }) => (
    <div className={`p-4 rounded-xl shadow-md ${color}`}>
        <p className="text-sm font-medium text-gray-500">{label}</p>
        <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
        {delta && <p className="text-xs text-red-600 mt-1">{delta}</p>}
    </div>
);

export default App;
