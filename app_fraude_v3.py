import streamlit as st
import json
import os
import sys
from datetime import datetime
import pandas as pd
import re

# Importar las funciones del módulo de fraude
try:
    from herramientas_fraude_avanzado_v3 import (
        detectar_fraude_heuristico,
        entrenar_modelo_sintetico,
        predecir_modelo,
        guardar_reclamo,
        consultar_reclamos,
        extraer_codigos_cie,
        extraer_cpt,
        normalizar_texto,
        PALABRAS_SOSPECHOSAS,
        DIAGNOSTICOS_CRITICOS_PREFIJOS,
        PROCEDIMIENTOS_CAROS,
        SKLEARN_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        MODELO_FRAUDE_PATH,
        DB_PATH
    )
    MODULO_DISPONIBLE = True
except ImportError as e:
    MODULO_DISPONIBLE = False
    st.error(f"Error importando módulo: {e}")
    st.info("Asegúrate de que 'herramientas_fraude_avanzado_v3.py' esté en el mismo directorio")

# Intentar importar librerías opcionales para visualización
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# CONFIGURACIÓN DE LA APP
# ============================================================================

st.set_page_config(
    page_title="Sistema de Detección de Fraude Médico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_risk_level(probabilidad):
    """Determina el nivel de riesgo basado en la probabilidad"""
    if probabilidad >= 0.7:
        return "🔴 CRÍTICO", "alert-high"
    elif probabilidad >= 0.5:
        return "🟠 ALTO", "alert-high"
    elif probabilidad >= 0.3:
        return "🟡 MEDIO", "alert-medium"
    else:
        return "🟢 BAJO", "alert-low"


def format_probability(prob):
    """Formatea la probabilidad como porcentaje"""
    return f"{prob * 100:.1f}%"


def format_clp(monto):
    """Formatea un número con separador de miles (punto) para CLP"""
    return f"${monto:,.0f}".replace(",", ".")


def parse_clp_input(texto):
    """Convierte texto con formato CLP a número"""
    if not texto:
        return 0
    # Remover todo excepto dígitos
    numeros = re.sub(r'[^\d]', '', texto)
    return int(numeros) if numeros else 0


def create_gauge_chart(probabilidad, title="Probabilidad de Fraude"):
    """Crea un gráfico de gauge para mostrar la probabilidad"""
    if not PLOTLY_AVAILABLE:
        return None

    # Determinar color
    if probabilidad >= 0.7:
        color = "red"
    elif probabilidad >= 0.5:
        color = "orange"
    elif probabilidad >= 0.3:
        color = "yellow"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probabilidad * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 50], 'color': '#fff3e0'},
                {'range': [50, 70], 'color': '#ffebee'},
                {'range': [70, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def create_factors_chart(scores_dict):
    """Crea gráfico de barras con los factores de riesgo"""
    if not PLOTLY_AVAILABLE or not scores_dict:
        return None

    factors = list(scores_dict.keys())
    values = list(scores_dict.values())

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=factors,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='RdYlGn_r',
                showscale=False
            ),
            text=[f"{v:.1%}" for v in values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Factores de Riesgo Detectados",
        xaxis_title="Score",
        yaxis_title="Factor",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 1])
    )

    return fig


# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
    st.title("⚙️ Configuración")

    # Verificar estado del sistema
    st.subheader("Estado del Sistema")

    if MODULO_DISPONIBLE:
        st.success("✅ Módulo cargado")
    else:
        st.error("❌ Módulo no disponible")

    col1, col2 = st.columns(2)
    with col1:
        if SKLEARN_AVAILABLE:
            st.success("✅ scikit-learn")
        else:
            st.warning("⚠️ scikit-learn")

    with col2:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            st.success("✅ NLP")
        else:
            st.warning("⚠️ NLP")

    # Verificar modelo entrenado
    modelo_existe = os.path.exists(MODELO_FRAUDE_PATH) if MODULO_DISPONIBLE else False

    if modelo_existe:
        st.success(f"✅ Modelo ML entrenado")
    else:
        st.warning("⚠️ Modelo no entrenado")

    st.divider()

    # Sección de entrenamiento
    st.subheader("🤖 Entrenamiento ML")

    n_samples = st.number_input(
        "Muestras sintéticas",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        if not MODULO_DISPONIBLE or not SKLEARN_AVAILABLE:
            st.error("scikit-learn no disponible")
        else:
            with st.spinner("Entrenando modelo..."):
                try:
                    resultado = entrenar_modelo_sintetico(n_samples=n_samples)
                    st.success("✅ Modelo entrenado exitosamente")
                    st.json(resultado)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Información
    st.subheader("ℹ️ Información")
    st.info("""
    **Sistema de Detección de Fraude Médico**

    Analiza reclamos usando:
    - Análisis heurístico
    - Códigos CIE-10
    - Códigos CPT
    - NLP semántico
    - Modelo ML
    """)

    st.caption("Desarrollado por Juan Carlos Cruces / Neosoltec - v1 2025")


# ============================================================================
# PÁGINA PRINCIPAL
# ============================================================================

st.markdown('<h1 class="main-header">🏥 Sistema de Detección de Fraude Médico</h1>',
            unsafe_allow_html=True)

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Análisis de Reclamo",
    "📊 Historial",
    "📈 Estadísticas",
    "ℹ️ Ayuda"
])


# ============================================================================
# TAB 1: ANÁLISIS DE RECLAMO
# ============================================================================

with tab1:
    st.header("Analizar Nuevo Reclamo")

    if not MODULO_DISPONIBLE:
        st.error("⚠️ Módulo no disponible. Verifica la instalación.")
        st.stop()

    # Formulario de entrada
    # Formulario de entrada
    with st.form("form_analisis"):
        col1, col2 = st.columns([2, 1])

        with col1:
            texto_reclamo = st.text_area(
                "Descripción del Reclamo",
                height=200,
                placeholder="Ingrese el texto completo del reclamo médico...\n\n"
                           "Ejemplo: Paciente de 55 años con dolor crónico severo en hombro derecho. "
                           "Diagnóstico: Hombro congelado (M75.0). Procedimiento: Artroscopia "
                           "(CPT 29826). Monto reclamado: $8.500.000 CLP. Incapacidad: 180 días. "
                           "Múltiples sesiones de rehabilitación prolongada.",
                help="Incluya toda la información disponible: diagnósticos, procedimientos, síntomas, etc."
            )

        with col2:
            st.subheader("Datos Adicionales ")

            # Input de monto simple
            monto_texto = st.text_input(
                "Monto Reclamado (CLP)",
                value="",
                placeholder="Ej: 5000000",
                help="Ingrese el monto en pesos chilenos (solo números o con puntos)"
            )

            # Convertir y mostrar
            monto = parse_clp_input(monto_texto)
            if monto > 0:
                st.caption(f"💰 **{format_clp(monto)} CLP**")

            edad = st.number_input(
                "Edad del Paciente",
                min_value=0,
                max_value=120,
                value=45,
                help="Edad en años"
            )

            dias_incapacidad = st.number_input(
                "Días de Incapacidad",
                min_value=0,
                max_value=365,
                value=0,
                help="Días de licencia médica"
            )

        submitted = st.form_submit_button(
            "🔎 Analizar Reclamo",
            type="primary",
            use_container_width=True
        )
    # Procesar análisis
    if submitted and texto_reclamo:
        with st.spinner("🔍 Analizando reclamo..."):
            try:
                # Análisis heurístico
                resultado = detectar_fraude_heuristico(texto_reclamo)

                # Predicción con modelo ML (si existe)
                prob_modelo = None
                alertas_modelo = []  # Inicializar por defecto
                nivel_riesgo_modelo = ""
                accion_sugerida = ""

                if os.path.exists(MODELO_FRAUDE_PATH) and SKLEARN_AVAILABLE:
                    try:
                        cnt_kw = len(resultado.get("keywords_detectadas", []))
                        cpt_codes = resultado.get("procedimientos_detectados", [])
                        has_cpt5 = int(any(len(x) >= 5 for x in cpt_codes))

                        pred = predecir_modelo(
                            monto=monto if monto > 0 else 5_000_000,
                            edad=edad,
                            dias_incapacidad=dias_incapacidad,
                            cnt_keywords=cnt_kw,
                            has_cpt5=has_cpt5
                        )
                        prob_modelo = pred.get("probabilidad_modelo")
                        # Extraer información adicional del modelo
                        alertas_modelo = pred.get("alertas", [])
                        nivel_riesgo_modelo = pred.get("nivel_riesgo", "")
                        accion_sugerida = pred.get("accion_sugerida", "")
                    except Exception as e:
                        st.warning(f"No se pudo ejecutar modelo ML: {e}")

                # Calcular probabilidad combinada
                prob_heuristica = resultado["probabilidad_heuristica"]

                if prob_modelo is not None:
                    # Combinar heurística y modelo (50% heurística, 50% modelo)
                    prob_final = 0.4 * prob_heuristica + 0.6 * prob_modelo
                else:
                    prob_final = prob_heuristica

                # Guardar en base de datos
                try:
                    guardar_reclamo(texto_reclamo, prob_heuristica, prob_modelo)
                except Exception as e:
                    st.warning(f"No se pudo guardar en BD: {e}")

                # ============================================================
                # MOSTRAR RESULTADOS
                # ============================================================

                st.success("✅ Análisis completado")

                # Nivel de riesgo
                nivel, clase_css = get_risk_level(prob_final)

                st.markdown(f"""
                <div class="{clase_css}">
                    <h2 style="margin:0;">{nivel}</h2>
                    <h3 style="margin:0.5rem 0;">Probabilidad de Fraude: {format_probability(prob_final)}</h3>
                </div>
                """, unsafe_allow_html=True)

                st.divider()

                # Gráficos
                col1, col2 = st.columns(2)

                with col1:
                    if PLOTLY_AVAILABLE:
                        fig_gauge = create_gauge_chart(prob_final)
                        if fig_gauge:
                            st.plotly_chart(fig_gauge, use_container_width=True)
                    else:
                        st.metric(
                            "Probabilidad de Fraude",
                            format_probability(prob_final)
                        )

                with col2:
                    if PLOTLY_AVAILABLE:
                        scores = {
                            "Keywords": resultado["score_kw"],
                            "Diagnósticos": resultado["score_dx"],
                            "Procedimientos": resultado["score_cpt"],
                            "Semántico": resultado["score_sem"]
                        }
                        fig_factors = create_factors_chart(scores)
                        if fig_factors:
                            st.plotly_chart(fig_factors, use_container_width=True)
                    else:
                        st.metric("Score Keywords", format_probability(resultado["score_kw"]))
                        st.metric("Score Diagnósticos", format_probability(resultado["score_dx"]))
                        st.metric("Score Procedimientos", format_probability(resultado["score_cpt"]))

                st.divider()

                # Detalles del análisis
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("🔑 Keywords Detectadas")
                    kw = resultado.get("keywords_detectadas", [])
                    if kw:
                        for word in kw:
                            st.markdown(f"- `{word}`")
                    else:
                        st.info("No se detectaron keywords sospechosas")

                with col2:
                    st.subheader("🏥 Códigos CIE-10")
                    codigos = resultado.get("codigos_cie", [])
                    if codigos:
                        for codigo in codigos:
                            es_critico = any(codigo.startswith(p) for p in DIAGNOSTICOS_CRITICOS_PREFIJOS)
                            emoji = "🔴" if es_critico else "⚪"
                            st.markdown(f"{emoji} `{codigo}`")
                    else:
                        st.info("No se detectaron códigos CIE-10")

                with col3:
                    st.subheader("💉 Procedimientos CPT")
                    procs = resultado.get("procedimientos_detectados", [])
                    if procs:
                        for proc in procs:
                            es_caro = proc in PROCEDIMIENTOS_CAROS
                            emoji = "💰" if es_caro else "⚪"
                            st.markdown(f"{emoji} `{proc}`")
                    else:
                        st.info("No se detectaron códigos CPT")

                # Mostrar alertas del modelo ML (si existen)
                if prob_modelo is not None and alertas_modelo:
                    st.divider()
                    st.subheader("🚨 Alertas del Modelo ML")
                    for alerta in alertas_modelo:
                        st.warning(f"⚠️ {alerta}")

                # Recomendaciones
                st.divider()
                st.subheader("💡 Recomendaciones")

                if prob_final >= 0.7:
                    st.error("""
                    **🚨 ACCIÓN INMEDIATA REQUERIDA:**
                    - Solicitar auditoría médica completa
                    - Verificar toda la documentación de respaldo
                    - Contactar al prestador para aclaración
                    - Revisar historial del paciente y prestador
                    - Considerar inspección in situ
                    """)
                elif prob_final >= 0.5:
                    st.warning("""
                    **⚠️ REVISIÓN DETALLADA:**
                    - Validar coherencia clínica del diagnóstico
                    - Verificar procedimientos realizados
                    - Revisar montos vs promedios del mercado
                    - Consultar con médico auditor
                    """)
                elif prob_final >= 0.3:
                    st.info("""
                    **ℹ️ REVISIÓN ESTÁNDAR:**
                    - Verificar documentación básica
                    - Validar códigos médicos
                    - Procesamiento normal con validación
                    """)
                else:
                    st.success("""
                    **✅ RIESGO BAJO:**
                    - Procesamiento estándar
                    - No se requieren acciones adicionales
                    """)

                # JSON detallado (expandible)
                with st.expander("📄 Ver Análisis Completo (JSON)"):
                    resultado_completo = {
                        "timestamp": datetime.now().isoformat(),
                        "probabilidad_final": prob_final,
                        "probabilidad_heuristica": prob_heuristica,
                        "probabilidad_modelo": prob_modelo,
                        "nivel_riesgo": nivel,
                        "nivel_riesgo_modelo": nivel_riesgo_modelo,
                        "accion_sugerida": accion_sugerida,
                        "alertas_modelo": alertas_modelo,
                        "monto_analizado_clp": monto,
                        "detalles": resultado
                    }
                    st.json(resultado_completo)

            except Exception as e:
                st.error(f"❌ Error en el análisis: {e}")
                st.exception(e)

    elif submitted:
        st.warning("⚠️ Por favor ingrese un texto de reclamo para analizar")


# ============================================================================
# TAB 2: HISTORIAL
# ============================================================================

with tab2:
    st.header("📊 Historial de Reclamos Analizados")

    if not MODULO_DISPONIBLE:
        st.error("Módulo no disponible")
    else:
        try:
            # Cargar historial
            reclamos = consultar_reclamos(limit=100)

            if not reclamos:
                st.info("No hay reclamos analizados aún")
            else:
                # Convertir a DataFrame
                df = pd.DataFrame(reclamos)

                # Estadísticas rápidas
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Reclamos", len(df))

                with col2:
                    alto_riesgo = len(df[df["prob_heuristica"] >= 0.5])
                    st.metric("Alto Riesgo", alto_riesgo)

                with col3:
                    if "prob_heuristica" in df.columns:
                        promedio = df["prob_heuristica"].mean()
                        st.metric("Probabilidad Promedio", f"{promedio:.1%}")

                with col4:
                    if "prob_modelo" in df.columns and df["prob_modelo"].notna().any():
                        con_modelo = df["prob_modelo"].notna().sum()
                        st.metric("Con Modelo ML", con_modelo)

                st.divider()

                # Filtros
                col1, col2 = st.columns(2)

                with col1:
                    filtro_riesgo = st.selectbox(
                        "Filtrar por Riesgo",
                        ["Todos", "Alto (>50%)", "Medio (30-50%)", "Bajo (<30%)"]
                    )

                with col2:
                    num_mostrar = st.slider("Número de registros", 10, 100, 20)

                # Aplicar filtros
                df_filtrado = df.copy()

                if filtro_riesgo == "Alto (>50%)":
                    df_filtrado = df_filtrado[df_filtrado["prob_heuristica"] >= 0.5]
                elif filtro_riesgo == "Medio (30-50%)":
                    df_filtrado = df_filtrado[
                        (df_filtrado["prob_heuristica"] >= 0.3) &
                        (df_filtrado["prob_heuristica"] < 0.5)
                    ]
                elif filtro_riesgo == "Bajo (<30%)":
                    df_filtrado = df_filtrado[df_filtrado["prob_heuristica"] < 0.3]

                df_filtrado = df_filtrado.head(num_mostrar)

                # Mostrar tabla
                st.dataframe(
                    df_filtrado,
                    use_container_width=True,
                    column_config={
                        "id": "ID",
                        "reclamo": st.column_config.TextColumn("Reclamo", width="large"),
                        "prob_heuristica": st.column_config.ProgressColumn(
                            "Prob. Heurística",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                        "prob_modelo": st.column_config.ProgressColumn(
                            "Prob. Modelo",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                        "created": "Fecha"
                    },
                    hide_index=True
                )

                # Visualización de distribución
                if PLOTLY_AVAILABLE and len(df) > 0:
                    st.subheader("Distribución de Probabilidades")

                    fig = px.histogram(
                        df,
                        x="prob_heuristica",
                        nbins=20,
                        title="Distribución de Probabilidad de Fraude",
                        labels={"prob_heuristica": "Probabilidad"},
                        color_discrete_sequence=["#1f77b4"]
                    )

                    fig.add_vline(x=0.3, line_dash="dash", line_color="yellow",
                                  annotation_text="Riesgo Medio")
                    fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                                  annotation_text="Riesgo Alto")
                    fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                                  annotation_text="Riesgo Crítico")

                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error cargando historial: {e}")
            st.exception(e)


# ============================================================================
# TAB 3: ESTADÍSTICAS
# ============================================================================

with tab3:
    st.header("📈 Estadísticas y Análisis")

    if not MODULO_DISPONIBLE:
        st.error("Módulo no disponible")
    else:
        try:
            reclamos = consultar_reclamos(limit=1000)

            if not reclamos or len(reclamos) < 5:
                st.info("Se necesitan al menos 5 reclamos para mostrar estadísticas")
            else:
                df = pd.DataFrame(reclamos)

                # Métricas generales
                st.subheader("📊 Métricas Generales")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Analizados", len(df))

                with col2:
                    criticos = len(df[df["prob_heuristica"] >= 0.7])
                    st.metric("Riesgo Crítico", criticos,
                             delta=f"{criticos/len(df)*100:.1f}%")

                with col3:
                    promedio = df["prob_heuristica"].mean()
                    st.metric("Probabilidad Promedio", f"{promedio:.2%}")

                with col4:
                    mediana = df["prob_heuristica"].median()
                    st.metric("Mediana", f"{mediana:.2%}")

                if PLOTLY_AVAILABLE:
                    # Gráfico de tendencia temporal
                    st.subheader("📈 Tendencia Temporal")

                    df['created'] = pd.to_datetime(df['created'])
                    df_temporal = df.set_index('created').resample('D')['prob_heuristica'].mean().reset_index()

                    fig = px.line(
                        df_temporal,
                        x='created',
                        y='prob_heuristica',
                        title="Promedio Diario de Probabilidad de Fraude",
                        labels={'created': 'Fecha', 'prob_heuristica': 'Probabilidad'}
                    )

                    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                                  annotation_text="Umbral Alto Riesgo")

                    st.plotly_chart(fig, use_container_width=True)

                    # Distribución por categoría
                    st.subheader("🎯 Distribución por Nivel de Riesgo")

                    df['categoria'] = pd.cut(
                        df['prob_heuristica'],
                        bins=[0, 0.3, 0.5, 0.7, 1.0],
                        labels=['Bajo', 'Medio', 'Alto', 'Crítico']
                    )

                    conteo = df['categoria'].value_counts().reset_index()
                    conteo.columns = ['Nivel', 'Cantidad']

                    fig = px.pie(
                        conteo,
                        values='Cantidad',
                        names='Nivel',
                        title="Distribución de Reclamos por Nivel de Riesgo",
                        color='Nivel',
                        color_discrete_map={
                            'Bajo': '#4caf50',
                            'Medio': '#ff9800',
                            'Alto': '#ff5722',
                            'Crítico': '#f44336'
                        }
                    )

                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generando estadísticas: {e}")
            st.exception(e)


# ============================================================================
# TAB 4: AYUDA
# ============================================================================

with tab4:
    st.header("ℹ️ Ayuda y Documentación")

    st.markdown("""
    ## 🎯 Cómo usar el sistema

    ### 1️⃣ Analizar un Reclamo
    1. Ve a la pestaña **"Análisis de Reclamo"**
    2. Ingresa el texto completo del reclamo médico
    3. Opcionalmente, agrega datos adicionales:
       - **Monto**: Usa punto como separador de miles (ej: 5.000.000)
       - **Edad**: Edad del paciente
       - **Días de incapacidad**: Días de licencia médica
    4. Haz clic en **"Analizar Reclamo"**
    5. Revisa los resultados y recomendaciones

    ### 2️⃣ Entrenar el Modelo ML
    1. En la barra lateral, ve a **"Entrenamiento ML"**
    2. Selecciona el número de muestras sintéticas (recomendado: 1000-2000)
    3. Haz clic en **"Entrenar Modelo"**
    4. Espera a que termine el entrenamiento

    ### 3️⃣ Revisar Historial
    1. Ve a la pestaña **"Historial"**
    2. Usa los filtros para buscar reclamos específicos
    3. Revisa las estadísticas generales

    ---

    ## 📋 Interpretación de Resultados

    ### Niveles de Riesgo:
    - 🟢 **BAJO** (<30%): Procesamiento estándar
    - 🟡 **MEDIO** (30-50%): Requiere revisión adicional
    - 🟠 **ALTO** (50-70%): Requiere auditoría detallada
    - 🔴 **CRÍTICO** (>70%): Acción inmediata requerida

    ### Factores Analizados:
    - **Keywords**: Palabras clave sospechosas en el texto
    - **Diagnósticos**: Códigos CIE-10 de riesgo
    - **Procedimientos**: Códigos CPT de alto valor
    - **Semántico**: Análisis de similitud con patrones de fraude

    ---

    ## 💡 Consejos de Uso

    - Incluye toda la información posible en la descripción del reclamo
    - Los montos elevados (>9M CLP) se marcan automáticamente como sospechosos
    - Las incapacidades largas (>200 días) requieren revisión especial
    - Entrena el modelo periódicamente con más muestras para mejor precisión
    """)
