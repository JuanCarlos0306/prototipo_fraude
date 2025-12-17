import pandas as pd
from faker import Faker
import random
import numpy as np
import sqlite3

num_registros = 1000


fake = Faker('es_ES')

# Definir cuántos doctores únicos queremos simular y proporcio de frude en data set
NUM_DOCTORES_UNICOS = 50
FRAUDE_PROPORCION = 0.1  # 5% ó 10%
# Generar la lista de doctores únicos
# Usamos un conjunto (set) para asegurar unicidad y luego lo convertimos a lista
#catalogo_doctores = list(
 #   {fake.name_male() + " " + fake.last_name() for _ in range(NUM_DOCTORES_UNICOS // 2)} | # 50% hombres
  #  {fake.name_female() + " " + fake.last_name() for _ in range(NUM_DOCTORES_UNICOS // 2)} # 50% mujeres
#)

catalogo_doctores = [fake.name() for _ in range(NUM_DOCTORES_UNICOS)]

fake = Faker('es_ES')

# Definir plantillas de texto (las mantenemos igual)
plantillas_descripcion = [
    "Paciente {sexo}, {edad} años. Diagnóstico: {diagnostico} ({codigo_cie}). Procedimiento: {procedimiento} (CPT {codigo_cpt}). Monto reclamado: ${monto_reclamado}. Incapacidad: {dias_invalidez} días.",
    "Se atendió a paciente {sexo} de {edad} años con diagnóstico de {diagnostico} ({codigo_cie}). Se realizó {procedimiento} con código CPT {codigo_cpt}. Monto reclamado: ${monto_reclamado}. Incapacidad: {dias_invalidez} días.",
    "Paciente {sexo}, {edad} años, presentó {diagnostico} ({codigo_cie}). Se le realizó {procedimiento} (CPT {codigo_cpt}). Monto reclamado: ${monto_reclamado}. Incapacidad: {dias_invalidez} días."
]

# Definir datos médicos organizados por CATEGORÍAS
# Cada categoría tiene diagnósticos relacionados con sus procedimientos apropiados
categorias_medicas = {
    "respiratorio_leve": {
        "diagnosticos": [
            ("Resfriado común", "J00"),
            ("Gripe", "J11"),
            ("Bronquitis aguda", "J20"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Radiografía de tórax, 2 vistas", "71046"),
            ("Hemograma completo con recuento diferencial", "85025"),
        ],
        "monto_rango": (50000, 400000),
        "dias_invalidez_rango": (0, 7)
    },
    "respiratorio_grave": {
        "diagnosticos": [
            ("Neumonía", "J18"),
            ("Asma", "J45"),
            ("Enfermedad pulmonar obstructiva crónica (EPOC)", "J44"),
            ("Embolia pulmonar", "I26"),
            ("Pleuresía", "J90"),
            ("Neumotórax", "J93"),
        ],
        "procedimientos": [
            ("Visita hospitalaria de emergencia con pruebas de laboratorio complejas", "99285"),
            ("Radiografía de tórax, 2 vistas", "71046"),
            ("Tomografía computarizada de abdomen con contraste", "74160"),
            ("Hemograma completo con recuento diferencial", "85025"),
            ("Electrocardiograma completo", "93000"),
            ("Panel metabólico completo", "80053"),
        ],
        "monto_rango": (800000, 3500000),
        "dias_invalidez_rango": (3, 15)
    },
    "cardiovascular": {
        "diagnosticos": [
            ("Hipertensión", "I10"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Electrocardiograma completo", "93000"),
            ("Prueba de esfuerzo cardiovascular", "93015"),
            ("Prueba de glucosa en sangre", "82947"),
            ("Panel metabólico completo", "80053"),
        ],
        "monto_rango": (100000, 800000),
        "dias_invalidez_rango": (0, 3)
    },
    "endocrino": {
        "diagnosticos": [
            ("Diabetes tipo 2", "E11"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Prueba de glucosa en sangre", "82947"),
            ("Panel metabólico completo", "80053"),
            ("Hemograma completo con recuento diferencial", "85025"),
        ],
        "monto_rango": (80000, 600000),
        "dias_invalidez_rango": (0, 5)
    },
    "traumatologia_leve": {
        "diagnosticos": [
            ("Esguinces y distensiones", "S93.4"),
            ("Dolor de espalda", "M54"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Radiografía de columna lumbar", "72110"),
            ("Fisioterapia, evaluación inicial", "97161"),
            ("Sutura simple de herida superficial", "12001"),
        ],
        "monto_rango": (100000, 600000),
        "dias_invalidez_rango": (0, 7)
    },
    "traumatologia_grave": {
        "diagnosticos": [
            ("Fractura de tobillo", "S82.5"),
            ("Fractura de cadera", "S72.0"),
            ("Fractura de muñeca", "S62.0"),
            ("Lesión de ligamento cruzado anterior", "S83.5"),
            ("Lesión de menisco", "S83.2"),
        ],
        "procedimientos": [
            ("Visita hospitalaria de emergencia con pruebas de laboratorio complejas", "99285"),
            ("Radiografía de columna lumbar", "72110"),
            ("Resonancia magnética de cerebro sin contraste", "70551"),
            ("Tomografía computarizada de abdomen con contraste", "74160"),
            ("Consulta de especialista, problema complejo", "99244"),
            ("Sutura simple de herida superficial", "12001"),
        ],
        "monto_rango": (1500000, 8000000),
        "dias_invalidez_rango": (7, 15)
    },
    "articular": {
        "diagnosticos": [
            ("Artritis reumatoide", "M06"),
            ("Osteoartritis de rodilla", "M17"),
            ("Osteoartritis de cadera", "M16"),
            ("Osteoporosis", "M81"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Inyección intraarticular de rodilla", "20610"),
            ("Fisioterapia, evaluación inicial", "97161"),
            ("Radiografía de columna lumbar", "72110"),
            ("Hemograma completo con recuento diferencial", "85025"),
        ],
        "monto_rango": (150000, 1200000),
        "dias_invalidez_rango": (0, 10)
    },
    "oftalmologico": {
        "diagnosticos": [
            ("Catarata", "H25"),
            ("Glaucoma", "H40"),
            ("Hipertensión ocular", "H40.0"),
        ],
        "procedimientos": [
            ("Consulta de especialista, problema complejo", "99244"),
            ("Cirugía de cataratas con implante de lente", "66984"),
        ],
        "monto_rango": (500000, 4000000),
        "dias_invalidez_rango": (0, 7)
    },
    "digestivo_leve": {
        "diagnosticos": [
            ("Gastroenteritis", "A09"),
            ("Caries dental", "K02"),
            ("Gingivitis", "K05"),
            ("Periodontitis", "K05.3"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Panel metabólico completo", "80053"),
        ],
        "monto_rango": (50000, 400000),
        "dias_invalidez_rango": (0, 5)
    },
    "digestivo_grave": {
        "diagnosticos": [
            ("Úlcera gástrica", "K25"),
            ("Colitis ulcerosa", "K51"),
            ("Enfermedad de Crohn", "K50"),
        ],
        "procedimientos": [
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Endoscopia digestiva alta con biopsia", "43239"),
            ("Colonoscopia con biopsia", "45380"),
            ("Tomografía computarizada de abdomen con contraste", "74160"),
            ("Hemograma completo con recuento diferencial", "85025"),
        ],
        "monto_rango": (800000, 3500000),
        "dias_invalidez_rango": (3, 12)
    },
    "quirurgico": {
        "diagnosticos": [
            ("Úlcera gástrica", "K25"),
        ],
        "procedimientos": [
            ("Cirugía de apendicitis", "44970"),
            ("Cirugía de vesícula biliar laparoscópica", "47562"),
            ("Visita hospitalaria de emergencia con pruebas de laboratorio complejas", "99285"),
            ("Tomografía computarizada de abdomen con contraste", "74160"),
        ],
        "monto_rango": (2500000, 8000000),
        "dias_invalidez_rango": (7, 15)
    },
    "urologico": {
        "diagnosticos": [
            ("Insuficiencia renal crónica", "N18"),
            ("Cistitis", "N30"),
            ("Prostatitis", "N41"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Panel metabólico completo", "80053"),
            ("Hemograma completo con recuento diferencial", "85025"),
        ],
        "monto_rango": (150000, 1500000),
        "dias_invalidez_rango": (0, 7)
    },
    "oncologico": {
        "diagnosticos": [
            ("Cáncer de mama", "C50"),
            ("Cáncer de próstata", "C61"),
            ("Cáncer de colon", "C18"),
            ("Cáncer de pulmón", "C34"),
            ("Cáncer de pulmón de células no pequeñas", "C34.9"),
            ("Cáncer de pulmón de células pequeñas", "C34.1"),
        ],
        "procedimientos": [
            ("Consulta de especialista, problema complejo", "99244"),
            ("Mamografía bilateral de screening", "77067"),
            ("Colonoscopia con biopsia", "45380"),
            ("Biopsia de piel con cierre simple", "11102"),
            ("Tomografía computarizada de abdomen con contraste", "74160"),
            ("Radiografía de tórax, 2 vistas", "71046"),
        ],
        "monto_rango": (1500000, 10000000),
        "dias_invalidez_rango": (5, 15)
    },
    "psiquiatrico": {
        "diagnosticos": [
            ("Depresión", "F32"),
            ("Ansiedad", "F41"),
            ("Trastorno bipolar", "F31"),
            ("Esquizofrenia", "F20"),
        ],
        "procedimientos": [
            ("Consulta médica general", "99213"),
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Consulta de especialista, problema complejo", "99244"),
        ],
        "monto_rango": (80000, 800000),
        "dias_invalidez_rango": (0, 10)
    },
    "ocupacional": {
        "diagnosticos": [
            ("Asbestosis", "J61"),
            ("Silicosis", "J62"),
        ],
        "procedimientos": [
            ("Consulta de especialista, problema complejo", "99244"),
            ("Radiografía de tórax, 2 vistas", "71046"),
            ("Tomografía computarizada de abdomen con contraste", "74160"),
        ],
        "monto_rango": (600000, 3000000),
        "dias_invalidez_rango": (5, 15)
    },
    "infeccioso": {
        "diagnosticos": [
            ("Tuberculosis pulmonar", "A15"),
        ],
        "procedimientos": [
            ("Consulta de seguimiento establecida, nivel moderado", "99214"),
            ("Radiografía de tórax, 2 vistas", "71046"),
            ("Hemograma completo con recuento diferencial", "85025"),
        ],
        "monto_rango": (400000, 2000000),
        "dias_invalidez_rango": (7, 15)
    },
    "prenatal": {
        "diagnosticos": [
            ("Resfriado común", "J00"),
        ],
        "procedimientos": [
            ("Ultrasonido obstétrico completo", "76805"),
            ("Consulta médica general", "99213"),
        ],
        "monto_rango": (100000, 600000),
        "dias_invalidez_rango": (0, 3)
    },
    "preventivo": {
        "diagnosticos": [
            ("Hipertensión", "I10"),
            ("Diabetes tipo 2", "E11"),
        ],
        "procedimientos": [
            ("Vacuna contra influenza", "90686"),
            ("Consulta médica general", "99213"),
        ],
        "monto_rango": (30000, 250000),
        "dias_invalidez_rango": (0, 0)
    }
}


# Función para generar un registro coherente
def generar_registro_coherente():
    # Seleccionar una categoría médica aleatoria
    categoria = random.choice(list(categorias_medicas.keys()))
    categoria_data = categorias_medicas[categoria]

    # Seleccionar diagnóstico y procedimiento de la misma categoría
    diagnostico, codigo_cie = random.choice(categoria_data["diagnosticos"])
    procedimiento, codigo_cpt = random.choice(categoria_data["procedimientos"])

    # Generar datos del paciente
    sexo = random.choice(["masculino", "femenino"])
    edad = fake.random_int(min=18, max=80)

    # Generar monto y días de invalidez según la categoría
    monto_min, monto_max = categoria_data["monto_rango"]
    dias_min, dias_max = categoria_data["dias_invalidez_rango"]

    monto_reclamado = round(random.uniform(monto_min, monto_max),0)
    dias_invalidez = random.randint(dias_min, dias_max)

    # Generar fechas
    fecha_inicio = fake.date_between(start_date='-5y', end_date='-6m')
    fecha_reclamo = fake.date_between(start_date=fecha_inicio, end_date='today')

    # Remplazar valores en la plantilla de texto
    descripcion = random.choice(plantillas_descripcion).format(
        sexo=sexo,
        edad=edad,
        diagnostico=diagnostico,
        codigo_cie=codigo_cie,
        procedimiento=procedimiento,
        codigo_cpt=codigo_cpt,
        monto_reclamado=monto_reclamado,
        dias_invalidez=dias_invalidez
    )

    return {
        'numero_poliza': fake.random_int(min=10000, max=99999),
        'nombre': fake.name_male() if sexo == "masculino" else fake.name_female(),
        'codigos_cie': codigo_cie,
        'codigos_cpt': codigo_cpt,
        'fecha_inicio_poliza': fecha_inicio,
        'fecha_reclamo': fecha_reclamo,
        'descripcion': descripcion,
        'dias_invalidez': dias_invalidez,
        'monto_reclamado': monto_reclamado,
        'doctor_tratante': random.choice(catalogo_doctores),
        'fraude_label': 0
    }

# Generar registros

data_legitima = []

for _ in range(num_registros):
    data_legitima.append(generar_registro_coherente())

df_base_1 = pd.DataFrame(data_legitima)

# --- 1. Configuración ---
N_FILAS = len(df_base_1)
N_FRAUDE = int(N_FILAS * FRAUDE_PROPORCION)

# Asegurarse de que las columnas de fecha son datetime para operar con ellas
df_base_1['fecha_inicio_poliza'] = pd.to_datetime(df_base_1['fecha_inicio_poliza'])
df_base_1['fecha_reclamo'] = pd.to_datetime(df_base_1['fecha_reclamo'])

# --- 2. Seleccionar los Índices de Fraude de forma Vectorial ---

# Crea un array booleano. False para la mayoría, True para N_FRAUDE filas.
es_fraude_mask = np.full(N_FILAS, False)
es_fraude_mask[:N_FRAUDE] = True

# Mezcla el array booleano para seleccionar filas aleatoriamente
np.random.shuffle(es_fraude_mask)

# Asignar la etiqueta de fraude usando la máscara
df_base_1['fraude_label'] = np.where(es_fraude_mask, 1, 0)

# Obtener la máscara booleana final para usar en las modificaciones
mask_fraude = df_base_1['fraude_label'] == 1


# --- 3. Modificación Vectorial de Valores (Reducción de Líneas) ---

# A. Modificación de 'monto_reclamado' (Aumento)
# Aplicamos un aumento aleatorio del 50% al 150% solo a las filas fraudulentas.
aumento_monto = np.random.uniform(1.5, 2.5, size=N_FILAS)
df_base_1.loc[mask_fraude, 'monto_reclamado'] = round(
    df_base_1.loc[mask_fraude, 'monto_reclamado'] * aumento_monto[mask_fraude]
)

# B. Modificación de 'dias_invalidez' (Aumento)
# Aplicamos una suma aleatoria de 5 a 30 días adicionales.
aumento_dias = np.random.randint(5, 31, size=N_FILAS)
df_base_1.loc[mask_fraude, 'dias_invalidez'] = (
    df_base_1.loc[mask_fraude, 'dias_invalidez'] + aumento_dias[mask_fraude]
)

# C. Modificación de 'fecha_reclamo' (Adelanto - Reclamo Rápido)
# Simulamos un reclamo en los primeros 7 días de la póliza.
dias_adelanto = pd.to_timedelta(np.random.randint(1, 8, size=N_FILAS), unit='D')

df_base_1.loc[mask_fraude, 'fecha_reclamo'] = (
    df_base_1.loc[mask_fraude, 'fecha_inicio_poliza'] + dias_adelanto[mask_fraude]
)
# --- 1. Configuración ---

N_FILAS = len(df_base_1)
N_FRAUDE = int(N_FILAS * FRAUDE_PROPORCION)

# Hacer una copia para no modificar el original
df_fraude = df_base_1.copy()

# Asegurarse de que las fechas son datetime
df_fraude['fecha_inicio_poliza'] = pd.to_datetime(df_fraude['fecha_inicio_poliza'])
df_fraude['fecha_reclamo'] = pd.to_datetime(df_fraude['fecha_reclamo'])

# --- 2. IMPORTANTE: Inicializar fraude_label en 0 PRIMERO ---
df_fraude['fraude_label'] = 0

# --- 3. Definir pools de datos sospechosos ---
doctores_sospechosos = [
    "Loreto Vidal Murillo",
    "Esteban Garcia Lopez",
    "Francisco Rojas Herrera",
    "Constanza Smith Cepa",
    "Ana Manriquez Trujillo"
    "Benito Morales Rossi"
]

# Códigos CIE/CPT raros o costosos (poco frecuentes)
codigos_cie_raros = ["J61", "J62", "C34.9", "C34.1", "S83.5", "S72.0"]

procedimientos_costosos = [
    ("Cirugía de vesícula biliar laparoscópica", "47562"),
    ("Cirugía de cataratas con implante de lente", "66984"),
    ("Resonancia magnética de cerebro sin contraste", "70551"),
    ("Tomografía computarizada de abdomen con contraste", "74160")
]

# Procedimientos inconsistentes (no relacionados con diagnósticos comunes)
procedimientos_inconsistentes = [
    ("Mamografía bilateral de screening", "77067"),
    ("Ultrasonido obstétrico completo", "76805"),
    ("Cirugía de cataratas con implante de lente", "66984"),
    ("Colonoscopia con biopsia", "45380")
]

# --- 4. Seleccionar Índices de Fraude ---
indices_fraude = np.random.choice(df_fraude.index, size=N_FRAUDE, replace=False)

# --- 5. Aplicar Patrones de Fraude SOLO EN COLUMNAS EXISTENTES ---
for idx in indices_fraude:

    # A. Monto inflado (70% de probabilidad en casos de fraude)
    if random.random() < 0.70:
        factor_inflacion = np.random.uniform(1.3, 3.0)  # 30% a 200% de aumento
        df_fraude.loc[idx, 'monto_reclamado'] = round(
            df_fraude.loc[idx, 'monto_reclamado'] * factor_inflacion
        )

    # B. Días de invalidez exagerados (60% de probabilidad)
    if random.random() < 0.60:
        dias_extra = np.random.randint(7, 45)
        df_fraude.loc[idx, 'dias_invalidez'] = (
            df_fraude.loc[idx, 'dias_invalidez'] + dias_extra
        )

    # C. Reclamo sospechosamente rápido (50% de probabilidad)
    if random.random() < 0.50:
        dias_rapido = np.random.randint(1, 15)
        df_fraude.loc[idx, 'fecha_reclamo'] = (
            df_fraude.loc[idx, 'fecha_inicio_poliza'] + pd.Timedelta(days=dias_rapido)
        )

    # D. Doctor tratante sospechoso (40% probabilidad)
    if random.random() < 0.40:
        df_fraude.loc[idx, 'doctor_tratante'] = random.choice(doctores_sospechosos)

    # E. Códigos CIE-CPT inconsistentes (30% probabilidad)
    if random.random() < 0.30:
        proc_nombre, proc_codigo = random.choice(procedimientos_inconsistentes)
        df_fraude.loc[idx, 'codigos_cpt'] = proc_codigo

        # Actualizar la descripción para mantener consistencia
        descripcion_actual = df_fraude.loc[idx, 'descripcion']
        descripcion_nueva = re.sub(r'CPT \d+', f'CPT {proc_codigo}', descripcion_actual)
        descripcion_nueva = re.sub(
            r'Procedimiento: .+ \(CPT',
            f'Procedimiento: {proc_nombre} (CPT',
            descripcion_nueva
        )
        df_fraude.loc[idx, 'descripcion'] = descripcion_nueva

    # F. Códigos CIE raros (25% probabilidad)
    if random.random() < 0.25:
        codigo_cie_raro = random.choice(codigos_cie_raros)
        df_fraude.loc[idx, 'codigos_cie'] = codigo_cie_raro

        # Actualizar descripción
        descripcion_actual = df_fraude.loc[idx, 'descripcion']
        descripcion_nueva = re.sub(r'\([A-Z]\d+\.?\d*\)', f'({codigo_cie_raro})', descripcion_actual)
        df_fraude.loc[idx, 'descripcion'] = descripcion_nueva

    # G. Montos justo debajo del límite de auditoría (20% probabilidad)
    if random.random() < 0.20:
        limites_auditoria = [2000000, 3000000, 5000000]  # CLP
        limite = random.choice(limites_auditoria)
        monto_justo_debajo = limite - np.random.randint(10000, 100000)
        df_fraude.loc[idx, 'monto_reclamado'] = monto_justo_debajo

        # Actualizar descripción con nuevo monto
        descripcion_actual = df_fraude.loc[idx, 'descripcion']
        descripcion_nueva = re.sub(
            r'\$\d+',
            f'${monto_justo_debajo}',
            descripcion_actual
        )
        df_fraude.loc[idx, 'descripcion'] = descripcion_nueva

    # H. Póliza nueva con reclamo inmediato (45% probabilidad)
    if random.random() < 0.45:
        fecha_poliza_reciente = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 30))
        df_fraude.loc[idx, 'fecha_inicio_poliza'] = fecha_poliza_reciente

        dias_inmediato = np.random.randint(1, 8)
        df_fraude.loc[idx, 'fecha_reclamo'] = fecha_poliza_reciente + pd.Timedelta(days=dias_inmediato)

    # I. Actualizar descripción con días de invalidez modificados
    descripcion_actual = df_fraude.loc[idx, 'descripcion']
    descripcion_nueva = re.sub(
        r'Incapacidad: \d+ días',
        f'Incapacidad: {df_fraude.loc[idx, "dias_invalidez"]} días',
        descripcion_actual
    )
    df_fraude.loc[idx, 'descripcion'] = descripcion_nueva

# --- 6. IMPORTANTE: Agregar ruido a los casos legítimos ---
# Esto evita que el modelo aprenda que "cualquier variación = fraude"

indices_legitimos = df_fraude[df_fraude['fraude_label'] == 0].index
n_legitimos_variables = int(len(indices_legitimos) * 0.15)  # 15% con variaciones

indices_legitimos_variables = np.random.choice(
    indices_legitimos,
    size=n_legitimos_variables,
    replace=False
)

for idx in indices_legitimos_variables:
    # Variaciones legítimas (pero menos extremas que fraude)
    if random.random() < 0.3:
        # Aumento moderado de monto (casos complicados legítimos)
        factor = np.random.uniform(1.1, 1.4)  # Solo 10-40% más
        df_fraude.loc[idx, 'monto_reclamado'] = round(
            df_fraude.loc[idx, 'monto_reclamado'] * factor
        )

        # Actualizar descripción
        descripcion_actual = df_fraude.loc[idx, 'descripcion']
        descripcion_nueva = re.sub(
            r'\$\d+',
            f'${int(df_fraude.loc[idx, "monto_reclamado"])}',
            descripcion_actual
        )
        df_fraude.loc[idx, 'descripcion'] = descripcion_nueva

    if random.random() < 0.2:
        # Algunos reclamos legítimos también pueden ser rápidos (emergencias reales)
        dias = np.random.randint(1, 30)
        df_fraude.loc[idx, 'fecha_reclamo'] = (
            df_fraude.loc[idx, 'fecha_inicio_poliza'] + pd.Timedelta(days=dias)
        )

# --- 7. AHORA SÍ asignar la etiqueta de fraude ---
df_fraude.loc[indices_fraude, 'fraude_label'] = 1

#Crar archivo excel
df_fraude.to_excel("BD_Modelo.xlsx",index=False)

# Crear BD en SQL
conn = sqlite3.connect('BD_Modelo_SQL')
df_fraude.to_sql('Tabla_Fraude', conn, if_exists='replace', index=False)
conn.close()
