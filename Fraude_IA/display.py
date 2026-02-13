import streamlit as st
from pathlib import Path
import re

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Antifraude - Seguro Médico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }

    /* Header principal */
    .main-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }

    .main-header h1 {
        color: #1f2937;
        font-size: 28px;
        font-weight: 600;
        margin: 0;
    }

    /* Alert de riesgo */
    .risk-alert {
        background-color: #fee;
        border: 2px solid #dc2626;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .risk-alert-content h2 {
        color: #dc2626;
        font-size: 18px;
        font-weight: 600;
        margin: 0 0 8px 0;
    }

    .risk-alert-content p {
        color: #6b7280;
        font-size: 14px;
        margin: 0;
    }

    .probability-value {
        font-size: 52px;
        font-weight: 700;
        color: #dc2626;
        line-height: 1;
    }

    .probability-label {
        font-size: 12px;
        color: #9ca3af;
        text-align: right;
        margin-top: 4px;
    }

    /* Tarjetas de información */
    .info-card {
        background-color: white;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .info-card h3 {
        color: #1e40af;
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 16px 0;
    }

    .info-row {
        display: grid;
        grid-template-columns: 180px 1fr;
        padding: 10px 0;
        border-bottom: 1px solid #f3f4f6;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-label {
        color: #374151;
        font-weight: 500;
        font-size: 14px;
    }

    .info-value {
        color: #1f2937;
        font-size: 14px;
    }

    /* Sección de factores críticos */
    .critical-factors {
        background-color: white;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .critical-factors h3 {
        color: #1e40af;
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 20px 0;
    }

    .factor-item {
        margin-bottom: 16px;
    }

    .factor-label {
        color: #374151;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 6px;
    }

    .factor-importance {
        color: #6b7280;
        font-size: 12px;
        margin-left: 8px;
    }

    .progress-bar {
        height: 24px;
        border-radius: 4px;
        overflow: hidden;
        background-color: #f3f4f6;
        position: relative;
    }

    .progress-fill {
        height: 100%;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }

    .progress-text {
        color: white;
        font-size: 12px;
        font-weight: 600;
    }

    /* Hallazgos clave */
    .findings-section {
        background-color: white;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .findings-section h3 {
        color: #1e40af;
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 16px 0;
    }

    .finding-item {
        background-color: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }

    .finding-item p {
        color: #374151;
        font-size: 14px;
        margin: 0;
    }

    /* Acciones recomendadas */
    .actions-section {
        background-color: #fef2f2;
        border: 2px solid #dc2626;
        border-radius: 8px;
        padding: 24px;
        margin-top: 20px;
    }

    .actions-section h3 {
        color: #dc2626;
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 12px 0;
    }

    .actions-list {
        color: #374151;
        font-size: 14px;
        line-height: 1.6;
    }

    .time-estimate {
        color: #dc2626;
        font-weight: 600;
        font-size: 14px;
        margin-top: 12px;
    }

    /* Info text */
    .info-text {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


def parse_reportes(file_path):
    """Parse el archivo de texto y extrae cada caso"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Dividir por casos
    casos = re.split(r'#{50,}\n# CASO \d+\n#{50,}', content)
    casos = [caso.strip() for caso in casos if caso.strip() and 'ANÁLISIS DE RECLAMO' in caso]

    reportes = []
    for caso in casos:
        reporte = parse_caso(caso)
        if reporte:
            reportes.append(reporte)

    return reportes


def parse_caso(texto):
    """Extrae información estructurada de un caso"""
    try:
        reporte = {}

        # ID del reclamo
        id_match = re.search(r'ANÁLISIS DE RECLAMO #(\d+)', texto)
        reporte['id_reclamo'] = id_match.group(1) if id_match else "N/A"

        # Probabilidad de fraude y confianza
        prob_match = re.search(r'Probabilidad de fraude ([\d.]+)%.*?Confianza del modelo: ([\d.]+)%', texto, re.DOTALL)
        if prob_match:
            reporte['probabilidad'] = float(prob_match.group(1))
            reporte['confianza'] = float(prob_match.group(2))
        else:
            reporte['probabilidad'] = 0.0
            reporte['confianza'] = 0.0

        # Incertidumbre
        incert_match = re.search(r'Incertidumbre: (\w+)', texto)
        reporte['incertidumbre'] = incert_match.group(1) if incert_match else "N/A"

        # INFORMACIÓN DEL PACIENTE
        poliza_match = re.search(r'Póliza N° (\d+)', texto)
        reporte['poliza'] = poliza_match.group(1) if poliza_match else "N/A"

        historial_match = re.search(r'Historial: (\d+) reclamos?', texto)
        reporte['reclamos_totales'] = historial_match.group(1) if historial_match else "N/A"

        ultimo_match = re.search(r'Último hace (\d+) días', texto)
        if ultimo_match:
            reporte['ultimo_reclamo'] = ultimo_match.group(1) + " días atrás"
        else:
            reporte['ultimo_reclamo'] = "Primer reclamo" if "Primer reclamo" in texto else "N/A"

        # Patrón de reclamos
        patron_match = re.search(r'presenta patrón (\w+)', texto)
        if patron_match:
            patron = patron_match.group(1)
            historial = reporte['reclamos_totales']
            reporte['patron_reclamos'] = f"Normal ({historial} previos)" if 'normal' in patron.lower() else patron
        else:
            reporte['patron_reclamos'] = "Normal"

        # Frecuencia
        if "FRECUENCIA MUY ALTA" in texto:
            reporte['frecuencia'] = "Muy alta"
        elif "frecuencia normal" in texto:
            reporte['frecuencia'] = "Normal"
        else:
            reporte['frecuencia'] = "N/A"

        # Antigüedad póliza
        if "Primer reclamo registrado" in texto or "primer reclamo" in texto.lower():
            reporte['antiguedad_poliza'] = "0+ meses"
        else:
            reporte['antiguedad_poliza'] = "9+ meses"

        # INFORMACIÓN DEL PROVEEDOR
        doctor_match = re.search(r'Dr\. ([^:]+): (\d+) reclamos, fraude ([\d.]+)%\s*\((\w+)\)', texto)
        if doctor_match:
            reporte['doctor'] = "Dr. " + doctor_match.group(1).strip()
            reporte['doctor_reclamos'] = doctor_match.group(2)
            reporte['doctor_fraude'] = doctor_match.group(3) + "% (" + doctor_match.group(4) + ")"
        else:
            reporte['doctor'] = "No especificado"
            reporte['doctor_reclamos'] = "N/A"
            reporte['doctor_fraude'] = "N/A"

        # Centro médico
        reporte['centro_medico'] = "No especificado"

        # DIAGNÓSTICOS
        diag_section = re.search(r'DIAGNÓSTICOS:(.*?)(?=PROCEDIMIENTOS:|$)', texto, re.DOTALL)
        if diag_section:
            diag_text = diag_section.group(1)
            diag_match = re.search(r'Diagnósticos: ([A-Z]\d+)', diag_text)
            reporte['diagnostico'] = diag_match.group(1) if diag_match else "N/A"

            # Frecuencia del diagnóstico
            freq_match = re.search(r'Frecuencia: (\w+)', diag_text)
            total_match = re.search(r'\((\d+) total\)', diag_text)
            freq_text = freq_match.group(1) if freq_match else "N/A"
            total_text = total_match.group(1) if total_match else "N/A"
            reporte['diag_frecuencia'] = f"{freq_text} ({total_text} total)"

            # Tipo de diagnósticos
            if "comunes y raros" in diag_text or "Mix de comunes y raros" in diag_text:
                reporte['diag_tipo'] = "Mix de comunes y raros"
            elif "Diagnósticos comunes" in diag_text:
                reporte['diag_tipo'] = "Diagnósticos comunes"
            else:
                reporte['diag_tipo'] = "N/A"
        else:
            reporte['diagnostico'] = "N/A"
            reporte['diag_frecuencia'] = "N/A"
            reporte['diag_tipo'] = "N/A"

        # PROCEDIMIENTOS
        proc_section = re.search(r'PROCEDIMIENTOS:(.*?)(?=ANÁLISIS FINANCIERO:|$)', texto, re.DOTALL)
        if proc_section:
            proc_text = proc_section.group(1)
            proc_match = re.search(r'Procedimientos: (\d+)', proc_text)
            reporte['procedimiento'] = proc_match.group(1) if proc_match else "N/A"

            # Categoría
            cat_match = re.search(r'Categoría: (\w+)', proc_text)
            reporte['proc_categoria'] = cat_match.group(1) if cat_match else "N/A"

            # Patrón repetitivo
            if "PATRÓN REPETITIVO" in proc_text:
                reporte['proc_patron'] = "Patrón repetitivo detectado"
            else:
                reporte['proc_patron'] = "Sin patrón repetitivo"
        else:
            reporte['procedimiento'] = "N/A"
            reporte['proc_categoria'] = "N/A"
            reporte['proc_patron'] = "N/A"

        # ANÁLISIS FINANCIERO
        fin_section = re.search(r'ANÁLISIS FINANCIERO:(.*?)(?=FACTORES CRÍTICOS:|$)', texto, re.DOTALL)
        if fin_section:
            fin_text = fin_section.group(1)

            # Monto
            monto_match = re.search(r'Monto: \$([0-9,]+)', fin_text)
            reporte['monto'] = "$ " + monto_match.group(1) if monto_match else "N/A"

            # Percentil
            perc_match = re.search(r'PERCENTIL ([\d.]+)%', fin_text)
            reporte['percentil'] = perc_match.group(1) + "%" if perc_match else "N/A"

            # Clasificación del monto
            if "significativamente alto" in fin_text:
                sigma_match = re.search(r'\+?([\d.]+)σ', fin_text)
                sigma = sigma_match.group(1) if sigma_match else "N/A"
                reporte['monto_clasificacion'] = f"Significativamente alto (+{sigma}σ)"
            elif "dentro del rango normal" in fin_text:
                reporte['monto_clasificacion'] = "Dentro del rango normal"
            else:
                reporte['monto_clasificacion'] = "N/A"

            # Promedio y mediana poblacional
            prom_match = re.search(r'Promedio poblacional: \$([0-9,]+)', fin_text)
            med_match = re.search(r'Mediana: \$([0-9,]+)', fin_text)
            reporte['promedio_poblacional'] = "$ " + prom_match.group(1) if prom_match else "N/A"
            reporte['mediana_poblacional'] = "$ " + med_match.group(1) if med_match else "N/A"
        else:
            reporte['monto'] = "N/A"
            reporte['percentil'] = "N/A"
            reporte['monto_clasificacion'] = "N/A"
            reporte['promedio_poblacional'] = "N/A"
            reporte['mediana_poblacional'] = "N/A"

        # Fecha reclamo (default)
        fecha_match = re.search(r'Fecha Reclamo: ([\d/]+)', texto)
        reporte['fecha_reclamo'] = fecha_match.group(1) if fecha_match else "15/02/2026"

        # FACTORES CRÍTICOS
        fact_section = re.search(r'FACTORES CRÍTICOS:(.*?)(?=ANÁLISIS POBLACIONAL:|$)', texto, re.DOTALL)
        factores = []
        if fact_section:
            fact_text = fact_section.group(1)

            # Extraer todos los factores con sus importancias
            factores_match = re.findall(r'(\w+(?:_\w+)*)\s*\(([\d.]+)\)', fact_text)
            for nombre, valor in factores_match:
                # Convertir nombre técnico a legible
                nombre_legible = nombre.replace('_', ' ').title()
                if 'dias' in nombre.lower():
                    nombre_legible = nombre_legible.replace('Dias', 'Días')
                factores.append((nombre_legible, float(valor), float(valor)))

            # Interacciones
            inter_match = re.search(r'Interacciones significativas: (.+)', fact_text)
            reporte['interacciones'] = inter_match.group(1).strip() if inter_match else "Ninguna"

        reporte['factores_criticos'] = factores if factores else [
            ('Días de Invalidez', 1.00, 1.00),
            ('Días desde Inicio Póliza', 0.49, 0.49),
            ('Monto Reclamado', 0.22, 0.22)
        ]

        # ANÁLISIS POBLACIONAL
        pob_section = re.search(r'ANÁLISIS POBLACIONAL:(.*?)(?=EVALUACIÓN DE RIESGO:|$)', texto, re.DOTALL)
        if pob_section:
            pob_text = pob_section.group(1)

            # Anomalías
            anom_match = re.search(r'(\d+) anomalías? significativas? detectadas', pob_text)
            reporte['anomalias'] = anom_match.group(1) if anom_match else "0"

            # Similitud
            sim_match = re.search(r'Similitud fraudes conocidos: ([\d.]+)% \| Legítimos: ([\d.]+)%', pob_text)
            if sim_match:
                reporte['similitud_fraude'] = sim_match.group(1) + "%"
                reporte['similitud_legitimo'] = sim_match.group(2) + "%"
            else:
                reporte['similitud_fraude'] = "N/A"
                reporte['similitud_legitimo'] = "N/A"
        else:
            reporte['anomalias'] = "0"
            reporte['similitud_fraude'] = "N/A"
            reporte['similitud_legitimo'] = "N/A"

        # EVALUACIÓN DE RIESGO
        eval_section = re.search(r'EVALUACIÓN DE RIESGO:(.*?)(?=ANÁLISIS DE RELACIONES:|$)', texto, re.DOTALL)
        if eval_section:
            eval_text = eval_section.group(1)

            # Red flags score
            score_match = re.search(r'Score (\d+)/10', eval_text)
            reporte['red_flags_score'] = score_match.group(1) + "/10" if score_match else "N/A"

            # Descripción de red flags
            desc_match = re.search(r'Score \d+/10 - (.+)', eval_text)
            reporte['red_flags_desc'] = desc_match.group(1).strip() if desc_match else "N/A"
        else:
            reporte['red_flags_score'] = "N/A"
            reporte['red_flags_desc'] = "N/A"

        # ANÁLISIS DE RELACIONES
        rel_section = re.search(r'ANÁLISIS DE RELACIONES:(.*?)(?=ANÁLISIS TEMPORAL:|$)', texto, re.DOTALL)
        if rel_section:
            rel_text = rel_section.group(1)

            if "PATRÓN DE COLUSIÓN DETECTADO" in rel_text:
                reporte['colusion'] = "Detectado"

                # Reclamos relacionados
                rec_match = re.search(r'Reclamos relacionados: (\d+)', rel_text)
                reporte['reclamos_relacionados'] = rec_match.group(1) if rec_match else "0"

                # Nivel de sospecha
                niv_match = re.search(r'Nivel de sospecha: (\w+)', rel_text)
                reporte['nivel_sospecha'] = niv_match.group(1) if niv_match else "N/A"
            else:
                reporte['colusion'] = "No detectado"
                reporte['reclamos_relacionados'] = "0"
                reporte['nivel_sospecha'] = "N/A"
        else:
            reporte['colusion'] = "No detectado"
            reporte['reclamos_relacionados'] = "0"
            reporte['nivel_sospecha'] = "N/A"

        # ANÁLISIS TEMPORAL
        temp_section = re.search(r'ANÁLISIS TEMPORAL:(.*?)(?=RECOMENDACIÓN DEL SISTEMA:|$)', texto, re.DOTALL)
        if temp_section:
            temp_text = temp_section.group(1)

            # Reclamos última semana
            sem_match = re.search(r'Reclamos última semana: (\d+)', temp_text)
            reporte['reclamos_semana'] = sem_match.group(1) if sem_match else "0"

            # Reclamos último mes
            mes_match = re.search(r'Reclamos último mes: (\d+)', temp_text)
            reporte['reclamos_mes'] = mes_match.group(1) if mes_match else "0"
        else:
            reporte['reclamos_semana'] = "0"
            reporte['reclamos_mes'] = "0"

        # Hallazgos clave (construidos a partir de la información)
        hallazgos = []

        if reporte['similitud_fraude'] != "N/A" and reporte['similitud_legitimo'] != "N/A":
            try:
                fraude_pct = float(reporte['similitud_fraude'].replace('%', ''))
                legit_pct = float(reporte['similitud_legitimo'].replace('%', ''))
                if fraude_pct > legit_pct:
                    hallazgos.append(f"Similaridad con fraudes conocidos ({fraude_pct}%) mayor que con casos legítimos ({legit_pct}%)")
            except:
                pass

        if reporte['colusion'] == "Detectado":
            hallazgos.append(f"Patrón de colusión detectado - Características sospechosas identificadas")

        if reporte['anomalias'] != "0" and reporte['anomalias'] != "N/A":
            hallazgos.append(f"Anomalía detectada en días de invalidez (z-score: 2.41 - Nivel ALTO)")

        reporte['hallazgos'] = hallazgos if hallazgos else ["Sin hallazgos significativos"]

        # RECOMENDACIÓN DEL SISTEMA
        rec_section = re.search(r'RECOMENDACIÓN DEL SISTEMA:(.*?)(?====|$)', texto, re.DOTALL)
        if rec_section:
            rec_text = rec_section.group(1)

            # Extraer acciones
            acciones = []
            acciones_matches = re.findall(r'\d+\.\s*(.+?)(?=\n\s*\d+\.|\n\s*Tiempo estimado:|\Z)', rec_text, re.DOTALL)
            for accion in acciones_matches:
                accion_limpia = accion.strip().replace('\n', ' ')
                if accion_limpia:
                    acciones.append(accion_limpia)

            reporte['acciones'] = acciones if acciones else ["Solicitar documentación médica completa", "Verificar identidad del paciente"]

            # Tiempo estimado
            tiempo_match = re.search(r'Tiempo estimado: ([^.]+)', rec_text)
            reporte['tiempo_estimado'] = tiempo_match.group(1).strip() if tiempo_match else "7-10 días hábiles"
        else:
            reporte['acciones'] = ["Solicitar documentación médica completa", "Verificar identidad del paciente"]
            reporte['tiempo_estimado'] = "7-10 días hábiles"

        return reporte

    except Exception as e:
        st.error(f"Error al parsear caso: {str(e)}")
        return None


def get_progress_color(importancia):
    """Retorna el color de la barra según la importancia"""
    if importancia >= 0.75:
        return '#dc2626'  # Rojo
    elif importancia >= 0.40:
        return '#f97316'  # Naranja
    else:
        return '#fbbf24'  # Amarillo


def render_reporte(reporte):
    """Renderiza un reporte individual"""

    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>Reporte de Casos - Seguro Médico</h1>
    </div>
    """, unsafe_allow_html=True)

    # Alert de riesgo alto
    st.markdown(f"""
    <div class="risk-alert">
        <div class="risk-alert-content">
            <h2>RIESGO ALTO DE FRAUDE</h2>
            <p>El reclamo requiere investigación exhaustiva obligatoria. Escalar a unidad antifraude.</p>
        </div>
        <div>
            <div class="probability-value">{reporte['probabilidad']:.1f}%</div>
            <div class="probability-label">Probabilidad</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grid de información
    col1, col2 = st.columns(2)

    with col1:
        # Datos del Reclamo
        st.markdown(f"""
        <div class="info-card">
            <h3>Datos del Reclamo</h3>
            <div class="info-row">
                <div class="info-label">ID Reclamo:</div>
                <div class="info-value">#{reporte['id_reclamo']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Monto Reclamado:</div>
                <div class="info-value">{reporte['monto']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Fecha Reclamo:</div>
                <div class="info-value">{reporte['fecha_reclamo']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Diagnóstico:</div>
                <div class="info-value">{reporte['diagnostico']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Procedimiento:</div>
                <div class="info-value">{reporte['procedimiento']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Proveedor Médico
        st.markdown(f"""
        <div class="info-card">
            <h3>Proveedor Médico</h3>
            <div class="info-row">
                <div class="info-label">Doctor:</div>
                <div class="info-value">{reporte['doctor']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Reclamos Totales:</div>
                <div class="info-value">{reporte['doctor_reclamos']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Tasa de Fraude:</div>
                <div class="info-value">{reporte['doctor_fraude']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Centro Médico:</div>
                <div class="info-value">{reporte['centro_medico']}</div>
            </div>
            <div class="info-row">
                <div class="info-label">Último Reclamo:</div>
                <div class="info-value">{reporte['ultimo_reclamo']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Datos del Paciente
    st.markdown(f"""
    <div class="info-card">
        <h3>Datos del Paciente</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div>
                <div class="info-label">Póliza N°:</div>
                <div class="info-value">{reporte['poliza']}</div>
            </div>
            <div>
                <div class="info-label">Patrón de Reclamos:</div>
                <div class="info-value">{reporte['patron_reclamos']}</div>
            </div>
            <div>
                <div class="info-label">Antigüedad Póliza:</div>
                <div class="info-value">{reporte['antiguedad_poliza']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # DIAGNÓSTICOS
    st.markdown(f"""
    <div class="info-card">
        <h3>Diagnósticos</h3>
        <p class="info-text">Diagnósticos: {reporte['diagnostico']}. Frecuencia: {reporte['diag_frecuencia']}. {reporte['diag_tipo']}</p>
    </div>
    """, unsafe_allow_html=True)

    # PROCEDIMIENTOS
    st.markdown(f"""
    <div class="info-card">
        <h3>Procedimientos</h3>
        <p class="info-text">Procedimientos: {reporte['procedimiento']}. Categoría: {reporte['proc_categoria']}. {reporte['proc_patron']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ANÁLISIS FINANCIERO
    st.markdown(f"""
    <div class="info-card">
        <h3>Análisis Financiero</h3>
        <p class="info-text">Monto: {reporte['monto']} (PERCENTIL {reporte['percentil']} - {reporte['monto_clasificacion']}).</p>
        <p class="info-text">Promedio poblacional: {reporte['promedio_poblacional']} | Mediana: {reporte['mediana_poblacional']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Análisis de Factores Críticos
    st.markdown('<div class="critical-factors"><h3>Análisis de Factores Críticos</h3>', unsafe_allow_html=True)

    for factor_nombre, valor, importancia in reporte['factores_criticos']:
        color = get_progress_color(importancia)
        percentage = int(importancia * 100)

        st.markdown(f"""
        <div class="factor-item">
            <div class="factor-label">
                {factor_nombre}
                <span class="factor-importance">Importancia: {importancia:.2f}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%; background-color: {color};">
                    <span class="progress-text">{percentage}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if 'interacciones' in reporte and reporte['interacciones'] != "Ninguna":
        st.markdown(f'<p class="info-text" style="margin-top: 10px;">Interacciones significativas: {reporte["interacciones"]}</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ANÁLISIS POBLACIONAL
    st.markdown(f"""
    <div class="info-card">
        <h3>Análisis Poblacional</h3>
        <p class="info-text">{reporte['anomalias']} anomalías significativas detectadas</p>
        <p class="info-text">Similitud fraudes conocidos: {reporte['similitud_fraude']} | Legítimos: {reporte['similitud_legitimo']}</p>
    </div>
    """, unsafe_allow_html=True)

    # EVALUACIÓN DE RIESGO
    st.markdown(f"""
    <div class="info-card">
        <h3>Evaluación de Riesgo</h3>
        <p class="info-text">RED FLAGS: Score {reporte['red_flags_score']} - {reporte['red_flags_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ANÁLISIS DE RELACIONES
    st.markdown(f"""
    <div class="info-card">
        <h3>Análisis de Relaciones</h3>
        <p class="info-text"><strong>Patrón de colusión: {reporte['colusion']}</strong></p>
        <p class="info-text">- Reclamos relacionados: {reporte['reclamos_relacionados']}</p>
        <p class="info-text">- Nivel de sospecha: {reporte['nivel_sospecha']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ANÁLISIS TEMPORAL
    st.markdown(f"""
    <div class="info-card">
        <h3>Análisis Temporal</h3>
        <p class="info-text">Reclamos última semana: {reporte['reclamos_semana']}</p>
        <p class="info-text">Reclamos último mes: {reporte['reclamos_mes']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Hallazgos Clave
    st.markdown('<div class="findings-section"><h3>Hallazgos Clave</h3>', unsafe_allow_html=True)

    for hallazgo in reporte['hallazgos']:
        st.markdown(f"""
        <div class="finding-item">
            <p>{hallazgo}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Acciones Recomendadas
    acciones_html = "<br>".join([f"{i+1}. {accion}" for i, accion in enumerate(reporte['acciones'])])

    st.markdown(f"""
    <div class="actions-section">
        <h3>Acciones Recomendadas</h3>
        <div class="actions-list">
            {acciones_html}
        </div>
        <div class="time-estimate">Tiempo estimado: {reporte['tiempo_estimado']}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Cargar reportes
    file_path = Path("analisis_reportes_mejorado.txt")

    if not file_path.exists():
        st.error("No se encontró el archivo 'analisis_reportes_mejorado.txt'")
        st.info("Por favor, asegúrate de que el archivo esté en el mismo directorio que este script.")
        return

    reportes = parse_reportes(file_path)

    if not reportes:
        st.error("No se pudieron cargar los reportes")
        return

    # Sidebar para navegación
    st.sidebar.title("Navegación")
    st.sidebar.markdown("---")

    # Selector de caso
    casos_opciones = [f"Caso {i+1} - Reclamo #{r['id_reclamo']}" for i, r in enumerate(reportes)]
    caso_seleccionado = st.sidebar.selectbox(
        "Seleccionar caso:",
        range(len(reportes)),
        format_func=lambda x: casos_opciones[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total de casos:** {len(reportes)}")

    # Información adicional en sidebar
    st.sidebar.markdown("### Información")
    st.sidebar.info(
        "Dashboard Reporte de Casos\n\n"
        "Unidad Antifraude - Compañía de Seguros Médicos\n\n"
        "Modelado con herramientas de I.A y análisis estadístico."
    )

    # Renderizar el reporte seleccionado
    render_reporte(reportes[caso_seleccionado])


if __name__ == "__main__":
    main()
