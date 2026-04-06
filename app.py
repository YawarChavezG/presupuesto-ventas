import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import google.generativeai as genai
import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente de Presupuestos COFAR", layout="wide", page_icon="💊")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración del Sistema")
api_key = st.sidebar.text_input("Google API Key (Para habilitar Chat)", type="password")
uploaded_file = st.sidebar.file_uploader("Subir Matriz de Ventas (Excel)", type=["xlsx"])

# Inicializar Gemini si hay API Key
model_gemini = None
if api_key:
    genai.configure(api_key=api_key)
    system_instruction = (
        "Eres el Asistente de Planificación de COFAR. Ayudas a los Jefes de Línea a interpretar "
        "las proyecciones de ventas de su portafolio OTC. Tienes acceso al contexto del modelo predictivo. "
        "Responde dudas sobre estacionalidades, justifica por qué un descuento sube o baja la venta "
        "según el historial, y sugiere escenarios estratégicos de forma corporativa."
    )
    model_gemini = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
else:
    st.sidebar.warning("Ingresa tu API Key de Google para activar el Asistente Conversacional.")

# --- FUNCIONES DE INGESTA Y LIMPIEZA ---
def clean_data(df):
    # 1. Limpieza rigurosa de encabezados ocultos (\n y espacios)
    df.columns = df.columns.str.replace('\n', ' ').str.strip()
    
    # 2. Renombrar columnas al estándar del modelo
    rename_map = {
        'Mes (YYYY-MM)': 'Fecha',
        'Regional (Oficina Ventas)': 'Oficina_Ventas',
        'Venta Cantidad (unidades)': 'Venta_Cantidad',
        'Quiebre Stock (días)': 'Quiebre_Stock',
        'Descuento (%)': 'Descuento',
        'Incremento Precio (0/1)': 'Incremento_Precio',
        'Estacionalidad (0/1)': 'Estacionalidad'
    }
    df = df.rename(columns=rename_map)
    
    # 3. Eliminar MS% Retail si existe
    if 'MS% Retail (promedio dpto.)' in df.columns:
        df = df.drop(columns=['MS% Retail (promedio dpto.)'])
    
    # 4. Limpieza de formatos numéricos latinos
    def clean_numeric(val):
        if pd.isna(val): return 0.0
        if isinstance(val, str):
            val = val.replace('.', '').replace(',', '.')
            if '%' in val:
                return float(val.replace('%', '')) / 100.0
            try:
                return float(val)
            except ValueError:
                return 0.0
        return float(val)

    cols_to_clean = ['Venta_Cantidad', 'Quiebre_Stock', 'Descuento', 'Incremento_Precio', 'Estacionalidad']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
            
    # Manejar formatos de porcentaje que vienen como entero (ej. 15 en vez de 0.15)
    df['Descuento'] = df['Descuento'].apply(lambda x: x / 100 if x > 1 else x)
    
    # 5. Formato de Tiempo
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.month
    df['Anio'] = df['Fecha'].dt.year
    
    return df

def create_lags(df):
    # Ordenamiento vital para series de tiempo
    df = df.sort_values(['Producto', 'Oficina_Ventas', 'Fecha']).reset_index(drop=True)
    df['Lag_1'] = df.groupby(['Producto', 'Oficina_Ventas'])['Venta_Cantidad'].shift(1)
    df['Lag_2'] = df.groupby(['Producto', 'Oficina_Ventas'])['Venta_Cantidad'].shift(2)
    df['Lag_3'] = df.groupby(['Producto', 'Oficina_Ventas'])['Venta_Cantidad'].shift(3)
    return df.dropna().reset_index(drop=True)

# --- ENTRENAMIENTO DINÁMICO (CACHED) ---
@st.cache_resource(show_spinner=False)
def train_models(data):
    models = {}
    combinations = data.groupby(['Producto', 'Oficina_Ventas'])
    features = ['Mes', 'Anio', 'Quiebre_Stock', 'Descuento', 'Incremento_Precio', 'Estacionalidad', 'Lag_1', 'Lag_2', 'Lag_3']
    
    for (prod, office), group in combinations:
        if len(group) > 5: # Validar que haya historial suficiente
            X = group[features]
            y = group['Venta_Cantidad']
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X, y)
            models[(prod, office)] = model
    return models

# --- LÓGICA DE LA APLICACIÓN ---
if uploaded_file:
    try:
        with st.spinner("Procesando datos y entrenando inteligencia artificial..."):
            df_raw = pd.read_excel(uploaded_file, sheet_name='Data_Transformada')
            df = clean_data(df_raw)
            df_lags = create_lags(df)
            all_models = train_models(df_lags)
            
        st.success("✅ Modelo entrenado exitosamente. Listo para simular.")
        
        # --- INTERFAZ DE SIMULACIÓN ---
        st.title("📊 Simulador de Presupuestos IA (Bottom-Up)")
        st.markdown("Ajusta los parámetros operativos para ver cómo reacciona el modelo predictivo.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configurar Escenario")
            producto = st.selectbox("Línea de Producto", df['Producto'].unique())
            oficina = st.selectbox("Oficina de Ventas", df['Oficina_Ventas'].unique())
            
            # Autocompletar la siguiente fecha lógica
            next_date = df['Fecha'].max() + pd.DateOffset(months=1)
            
            c_mes, c_anio = st.columns(2)
            with c_mes: target_month = st.number_input("Mes", 1, 12, next_date.month)
            with c_anio: target_year = st.number_input("Año", 2024, 2030, next_date.year)
            
            descuento = st.slider("Descuento Comercial (%)", 0.0, 50.0, 0.0, step=1.0) / 100
            
            c_check1, c_check2 = st.columns(2)
            with c_check1: quiebre = st.checkbox("Simular Quiebre de Stock")
            with c_check2: incremento = st.checkbox("Aplicar Incremento Precio")
            estacionalidad = st.checkbox("Marcar como Temporada Alta (Estacionalidad)")
            
            btn_calc = st.button("🚀 Calcular Proyección", use_container_width=True)

        contexto_proyeccion = "Aún no se ha calculado ningún escenario."
        
        if btn_calc:
            if (producto, oficina) in all_models:
                # Recuperar historial exacto de esa regional
                last_data = df_lags[(df_lags['Producto'] == producto) & (df_lags['Oficina_Ventas'] == oficina)].iloc[-1]
                
                input_data = pd.DataFrame([{
                    'Mes': target_month,
                    'Anio': target_year,
                    'Quiebre_Stock': 1 if quiebre else 0,
                    'Descuento': descuento,
                    'Incremento_Precio': 1 if incremento else 0,
                    'Estacionalidad': 1 if estacionalidad else 0,
                    'Lag_1': last_data['Venta_Cantidad'],
                    'Lag_2': last_data['Lag_1'],
                    'Lag_3': last_data['Lag_2']
                }])
                
                pred = all_models[(producto, oficina)].predict(input_data)[0]
                pred = max(0, round(pred)) # Evitar negativos y fracciones
                
                with col2:
                    st.subheader("Resultado del Modelo XGBoost")
                    st.metric("Volumen Sugerido a Presupuestar", f"{pred:,} Cajas")
                    st.info(f"📍 Segmento: {producto} | {oficina}")
                    
                    # Guardar string de contexto para el LLM
                    contexto_proyeccion = (
                        f"Acabamos de simular un escenario para el producto '{producto}' en la regional '{oficina}'. "
                        f"Para la fecha {target_month}/{target_year}, aplicando un descuento de {descuento*100}%, "
                        f"quiebre de stock={quiebre}, incremento de precio={incremento} y estacionalidad={estacionalidad}. "
                        f"El modelo XGBoost proyectó una demanda exacta de {pred} unidades."
                    )
            else:
                st.error("No hay suficiente historial para proyectar este producto en esta regional.")

        # --- CHATBOT GEMINI ---
        st.divider()
        st.subheader("💬 Consultor Estratégico (Gemini)")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Pregunta sobre tu presupuesto, el efecto de los descuentos o estrategias..."):
            if not api_key:
                st.error("⚠️ Para usar el chat, ingresa tu API Key en la barra lateral.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # Empaquetamos el contexto dinámico y la pregunta
                    full_prompt = f"CONTEXTO DEL ESCENARIO ACTUAL: {contexto_proyeccion}\n\nPREGUNTA DEL USUARIO: {prompt}"
                    
                    try:
                        response = model_gemini.generate_content(full_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Ocurrió un error con la API de Gemini: {e}")

    except Exception as e:
        st.error(f"Error procesando el archivo Excel. Revisa el formato. Detalle técnico: {e}")
else:
    # Pantalla de bienvenida
    st.title("Bienvenido al Asistente de Presupuestos con IA")
    st.markdown("""
    Esta herramienta reemplaza el proceso de presupuestación manual (Top-Down) por un modelo **XGBoost (Bottom-Up)**.
    1. Despliega la barra lateral izquierda.
    2. Sube el archivo base (`data_ventas_transformada.xlsx`).
    3. Ingresa tu Google API Key para activar las recomendaciones estratégicas.
    """)
