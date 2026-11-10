"""
Motor de Recomendaciones Pedagógicas Basado en Reglas Explícitas y Fundamentadas.
Proyecto: Asistente de Inteligencia Artificial para la Planificación Docente (OULAD).

Nota Metodológica Importante:
Este módulo implementa un sistema experto de reglas pedagógicas deterministas que
complementa (pero se distingue estrictamente de) la estimación probabilística del
modelo de Machine Learning. Las recomendaciones aquí generadas son orientativas para
la planificación docente y nunca sustituyen el juicio profesional del educador.
"""

from typing import Dict, List, Any


# Escenarios pedagógicos preconfigurados (sin datos personales reales) para exploración en el simulador
PREDEFINED_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Personalizado (Ingreso manual)": {},
    "Escenario 1: Reingreso con sobrecarga académica (Alerta Alta)": {
        "code_module": "CCC",
        "code_presentation": "2014J",
        "gender": "M",
        "region": "North Western Region",
        "highest_education": "Lower Than A Level",
        "imd_band": "10-20",
        "age_band": "0-35",
        "num_of_prev_attempts": 2,
        "studied_credits": 120,
        "disability": "N",
    },
    "Escenario 2: Primer ingreso con carga estándar y formación previa sólida (Pronóstico Favorable)": {
        "code_module": "AAA",
        "code_presentation": "2014J",
        "gender": "F",
        "region": "South East Region",
        "highest_education": "HE Qualification",
        "imd_band": "80-90%",
        "age_band": "35-55",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
    },
    "Escenario 3: Estudiante con condición de discapacidad y alta privación socioeconómica (Acompañamiento Inclusivo)": {
        "code_module": "BBB",
        "code_presentation": "2014B",
        "gender": "F",
        "region": "Scotland",
        "highest_education": "No Formal quals",
        "imd_band": "0-10%",
        "age_band": "0-35",
        "num_of_prev_attempts": 1,
        "studied_credits": 90,
        "disability": "Y",
    },
    "Escenario 4: Adulto mayor en módulo STEM con carga parcial (Seguimiento Formativo)": {
        "code_module": "FFF",
        "code_presentation": "2013J",
        "gender": "M",
        "region": "London Region",
        "highest_education": "A Level or Equivalent",
        "imd_band": "40-50%",
        "age_band": "55<=",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
    },
}


def clasificar_nivel_alerta(prob_fracaso: float) -> Dict[str, str]:
    """
    Traduce la probabilidad estimada de fracaso/retiro (clase 0) en un nivel orientativo
    de prioridad pedagógica con explicación comprensible para el docente.
    """
    if prob_fracaso >= 0.60:
        return {
            "nivel": "Prioridad Pedagógica Alta",
            "color": "#DC2626",
            "badge": "ALTA PRIORIDAD DE ACOMPAÑAMIENTO",
            "resumen": (
                "El modelo estadístico estima una probabilidad elevada de dificultad académica o retiro "
                f"({prob_fracaso * 100:.1f}%). Se sugiere activar tempranamente estrategias de tutoría "
                "individualizada, dosificación de carga y evaluación formativa continua."
            ),
        }
    elif prob_fracaso >= 0.45:
        return {
            "nivel": "Prioridad Pedagógica Moderada (Zona de Incertidumbre)",
            "color": "#D97706",
            "badge": "SEGUIMIENTO PREVENTIVO",
            "resumen": (
                "Las puntuaciones estimadas se encuentran en una franja intermedia de incertidumbre "
                f"({prob_fracaso * 100:.1f}% de riesgo estimado). Pequeñas intervenciones oportunas en la "
                "planificación docente y en la retroalimentación pueden inclinar favorablemente el resultado."
            ),
        }
    else:
        return {
            "nivel": "Pronóstico Favorable (Seguimiento Estándar)",
            "color": "#059669",
            "badge": "PRONÓSTICO FAVORABLE",
            "resumen": (
                "El perfil académico presenta condiciones estadísticas asociadas al éxito en el módulo "
                f"({(1.0 - prob_fracaso) * 100:.1f}% de probabilidad estimada de aprobación). Se recomienda "
                "mantener el estímulo académico y ofrecer retos de profundización."
            ),
        }


def generar_recomendaciones_pedagogicas(
    student_profile: Dict[str, Any],
    predicted_class: int,
    prob_fracaso: float,
    prob_exito: float,
) -> Dict[str, Any]:
    """
    Ejecuta el motor explícito de reglas pedagógicas sobre las características del estudiante
    y la estimación probabilística del modelo.

    Distingue explícitamente:
      1. El resultado estadístico del modelo (predicción + probabilidades).
      2. Las reglas pedagógicas activadas por condiciones transparentes de entrada.
    """
    alerta = clasificar_nivel_alerta(prob_fracaso)
    reglas_activadas: List[Dict[str, str]] = []

    num_attempts = int(student_profile.get("num_of_prev_attempts", 0))
    credits = int(student_profile.get("studied_credits", 60))
    edu = str(student_profile.get("highest_education", "A Level or Equivalent"))
    imd = str(student_profile.get("imd_band", "30-40%"))
    disability = str(student_profile.get("disability", "N"))
    age = str(student_profile.get("age_band", "0-35"))
    module = str(student_profile.get("code_module", "BBB"))

    # 1. Tutorías y acompañamiento individual
    if num_attempts >= 1 or prob_fracaso >= 0.55:
        motivo = (
            f"Regla R1 activada: Registra {num_attempts} intento(s) previo(s) en el módulo y/o "
            f"probabilidad estimada de dificultad del {prob_fracaso * 100:.1f}%."
        )
        accion = (
            "Programar sesiones quincenales de tutoría individualizada desde la semana 1 para "
            "diagnosticar los obstáculos específicos de intentos anteriores y establecer compromisos graduales."
        )
    else:
        motivo = "Regla R1 (Preventiva): Sin intentos fallidos previos registrados."
        accion = (
            "Mantener canales de tutoría grupal y consulta asincrónica abiertos para resolver dudas puntuales."
        )
    reglas_activadas.append(
        {
            "dimension": "1. Tutorías y acompañamiento individual",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": "Alta" if (num_attempts >= 1 or prob_fracaso >= 0.55) else "Estándar",
        }
    )

    # 2. Actividades de refuerzo
    if edu in ["No Formal quals", "Lower Than A Level"]:
        motivo = f"Regla R2 activada: Nivel educativo previo reportado como '{edu}'."
        accion = (
            "Incorporar módulos de nivelación propedéutica, guías paso a paso y talleres de refuerzo "
            "en competencias base (comprensión lectora académica, razonamiento cuantitativo y hábitos de estudio)."
        )
        prio_r2 = "Alta"
    else:
        motivo = f"Regla R2 (Consolidación): Nivel educativo previo '{edu}'."
        accion = (
            "Integrar ejercicios de aplicación práctica y casos de estudio integradores que conecten "
            "los conocimientos previos con los objetivos del módulo."
        )
        prio_r2 = "Media"
    reglas_activadas.append(
        {
            "dimension": "2. Actividades de refuerzo",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": prio_r2,
        }
    )

    # 3. Recursos educativos complementarios
    low_imd_bands = ["0-10%", "10-20", "20-30%"]
    if disability == "Y" or imd in low_imd_bands:
        factores = []
        if disability == "Y":
            factores.append("condición de discapacidad declarada (Y)")
        if imd in low_imd_bands:
            factores.append(f"banda socioeconómica de mayor vulnerabilidad (IMD {imd})")
        motivo = "Regla R3 activada: Presencia de " + " y ".join(factores) + "."
        accion = (
            "Garantizar recursos bajo Diseño Universal para el Aprendizaje (DUA): materiales descargables "
            "de bajo consumo de datos (offline), transcripciones completas, audiolibros y cápsulas breves."
        )
        prio_r3 = "Alta"
    else:
        motivo = f"Regla R3 (Enriquecimiento): IMD '{imd}' y discapacidad '{disability}'."
        accion = (
            "Proporcionar repositorios digitales complementarios, lecturas avanzadas y simuladores interactivos en el VLE."
        )
        prio_r3 = "Estándar"
    reglas_activadas.append(
        {
            "dimension": "3. Recursos educativos complementarios",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": prio_r3,
        }
    )

    # 4. Evaluaciones formativas
    if predicted_class == 0 or num_attempts >= 1:
        motivo = (
            "Regla R4 activada: Predicción en zona de alerta o historial de repitencia en el módulo."
        )
        accion = (
            "Implementar micro-evaluaciones formativas semanales sin peso punitivo (quizzes de autoevaluación "
            "y entregas preliminares antes del primer TMA/examen) para verificar comprensión temprana."
        )
        prio_r4 = "Alta"
    else:
        motivo = "Regla R4 (Monitoreo continuo): Perfil con pronóstico de aprobación."
        accion = (
            "Utilizar rúbricas de autoevaluación y coevaluación entre pares antes de cada entrega sumativa oficial."
        )
        prio_r4 = "Estándar"
    reglas_activadas.append(
        {
            "dimension": "4. Evaluaciones formativas",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": prio_r4,
        }
    )

    # 5. Retroalimentación personalizada
    if prob_fracaso >= 0.50 or age == "55<=":
        motivo = (
            f"Regla R5 activada: Probabilidad de riesgo ({prob_fracaso * 100:.1f}%) o grupo etario '{age}'."
        )
        accion = (
            "Aplicar retroalimentación dialógica y prospectiva (feed-forward) en un plazo máximo de 72 horas "
            "tras cada actividad, destacando logros concretos y detallando pasos accionables de mejora."
        )
        prio_r5 = "Alta"
    else:
        motivo = "Regla R5 (Estímulo académico): Desempeño proyectado favorable."
        accion = (
            "Brindar retroalimentación orientada al pensamiento crítico y a la transferencia del aprendizaje a contextos reales."
        )
        prio_r5 = "Estándar"
    reglas_activadas.append(
        {
            "dimension": "5. Retroalimentación personalizada",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": prio_r5,
        }
    )

    # 6. Seguimiento de la participación académica
    if credits >= 90 or prob_fracaso >= 0.50:
        motivo = (
            f"Regla R6 activada: Carga académica de {credits} créditos (variable más influyente del modelo) "
            f"y/o alerta predictiva."
        )
        accion = (
            "Activar monitoreo semanal de interacción en la plataforma virtual (VLE): verificar acceso a foros, "
            "lectura de recursos y entrega puntual en las primeras 3 semanas críticas de retención."
        )
        prio_r6 = "Alta"
    else:
        motivo = f"Regla R6 (Seguimiento regular): Carga manejable de {credits} créditos."
        accion = (
            "Realizar seguimiento quincenal de hitos de participación en actividades colaborativas del entorno virtual."
        )
        prio_r6 = "Media"
    reglas_activadas.append(
        {
            "dimension": "6. Seguimiento de la participación académica",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": prio_r6,
        }
    )

    # 7. Ajustes en la planificación docente
    if credits >= 120:
        motivo = (
            f"Regla R7 activada: Sobrecarga académica crítica ({credits} créditos matriculados simultáneamente)."
        )
        accion = (
            f"Ajustar la planificación didáctica del módulo {module}: fragmentar entregas extensas en hitos "
            "modulares, flexibilizar cronogramas de entrega y orientar al estudiante sobre gestión del tiempo o "
            "ajuste de carga crediticia."
        )
        prio_r7 = "Alta"
    elif module in ["CCC", "DDD", "FFF"]:
        motivo = f"Regla R7 activada: Módulo técnico/STEM ({module}) con alta exigencia analítica."
        accion = (
            f"Integrar sesiones prácticas guiadas, laboratorios resueltos paso a paso y aprendizaje colaborativo "
            f"en la secuencia didáctica del módulo {module}."
        )
        prio_r7 = "Media"
    else:
        motivo = f"Regla R7 (Optimización de secuencia): Módulo {module} con carga de {credits} créditos."
        accion = (
            "Equilibrar actividades sincrónicas y autónomas en el plan de clase, incorporando pausas activas "
            "de síntesis conceptual al cierre de cada unidad temática."
        )
        prio_r7 = "Estándar"
    reglas_activadas.append(
        {
            "dimension": "7. Ajustes en la planificación docente",
            "regla": motivo,
            "recomendacion": accion,
            "prioridad": prio_r7,
        }
    )

    return {
        "alerta": alerta,
        "prediccion_modelo": {
            "clase_predicha": int(predicted_class),
            "etiqueta": (
                "Éxito Académico (Pass / Distinction)"
                if int(predicted_class) == 1
                else "Riesgo de Fracaso o Retiro (Fail / Withdrawn)"
            ),
            "probabilidad_fracaso": round(float(prob_fracaso), 4),
            "probabilidad_exito": round(float(prob_exito), 4),
        },
        "reglas_pedagogicas": reglas_activadas,
        "nota_etica": (
            "ADVERTENCIA ÉTICA Y PEDAGÓGICA: Esta predicción constituye una estimación estadística sujeta a "
            "incertidumbre (Accuracy del modelo: ~61%, ROC-AUC: ~0.65 sobre variables iniciales de matrícula) "
            "y NO representa un diagnóstico definitivo ni una determinación sobre la capacidad del estudiante. "
            "Las recomendaciones generadas provienen de un motor de reglas pedagógicas explícitas diseñado "
            "exclusivamente para apoyar la planificación docente y la intervención formativa oportuna."
        ),
    }

