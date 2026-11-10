"""
Aplicación Web Interactiva: Asistente de Inteligencia Artificial para la Planificación Docente.

Asignatura: Aprendizaje Automático (IAE-2501)
Programa: Maestría en Inteligencia Artificial Aplicada a la Educación e Investigación
Institución: Universidad Federico Henríquez y Carvajal (UFHEC)
Docente: Feibert Guzmán
Estudiante: Jhessika Lidabed Espaillat Espaillat

Arquitectura modular:
- Presentación e interfaz de usuario: app.py
- Preprocesamiento y validación de dominio: src/preprocessing.py
- Inferencia y carga de modelos/métricas: src/inference.py
- Motor híbrido de reglas pedagógicas: src/recommendations.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.preprocessing import FEATURE_DOMAINS
from src.inference import (
    load_trained_pipeline,
    load_evaluation_metrics,
    predict_student_outcome,
)
from src.recommendations import PREDEFINED_SCENARIOS

# Configuración general de la página
st.set_page_config(
    page_title="Asistente IA para la Planificación Docente | UFHEC",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_IMG_DIR = ROOT_DIR / "assets" / "img"
REPORTS_DIR = ROOT_DIR / "reports"


def load_custom_css() -> None:
    """Inyecta estilos CSS institucionales para mejorar la legibilidad académica."""
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #0f2942 0%, #1e4e79 100%);
            color: white;
            padding: 1.6rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 1.85rem;
            margin-bottom: 0.3rem;
        }
        .main-header p {
            color: #dbeafe;
            margin: 0.15rem 0;
            font-size: 0.96rem;
        }
        .badge-ufhec {
            display: inline-block;
            background-color: #f59e0b;
            color: #0f172a;
            font-weight: 700;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }
        .metric-box {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #1e4e79;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.8rem;
        }
        .ethical-box {
            background-color: #fffbeb;
            border: 1px solid #f59e0b;
            border-left: 6px solid #d97706;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            color: #78350f;
            font-size: 0.92rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Cargando pipeline de Machine Learning...")
def get_cached_pipeline():
    """Carga en caché el pipeline entrenado para optimizar tiempos de respuesta."""
    return load_trained_pipeline()


@st.cache_data(show_spinner="Cargando métricas de evaluación verificadas...")
def get_cached_metrics():
    """Carga en caché las métricas reales generadas por el cuaderno."""
    return load_evaluation_metrics()


def render_sidebar() -> None:
    """Renderiza la barra lateral con ficha institucional y resumen metodológico."""
    with st.sidebar:
        st.markdown("### 🏛️ Ficha Institucional")
        st.markdown(
            """
            **Institución:** Universidad Federico Henríquez y Carvajal (UFHEC)  
            **Programa:** Maestría en Inteligencia Artificial Aplicada a la Educación e Investigación  
            **Asignatura:** Aprendizaje Automático (`IAE-2501`)  
            **Docente:** Feibert Guzmán  
            **Autora:** Jhessika Lidabed Espaillat Espaillat
            """
        )
        st.divider()
        st.markdown("### 🧠 Arquitectura del Sistema")
        st.markdown(
            """
            1. **Ingesta y ETL sin Fuga (*Leak-Free*):** Imputación y codificación ajustadas exclusivamente en entrenamiento (`n = 26,074`).
            2. **Modelo Principal:** `RandomForestClassifier` comparado contra `LogisticRegression` y `DecisionTreeClassifier`.
            3. **Motor Pedagógico Basado en Reglas:** 7 dimensiones de intervención docente explícitas e independientes de la inferencia estadística.
            """
        )
        st.divider()
        st.info(
            "💡 **Nota Ética:** Las predicciones son estimaciones probabilísticas sujetas a incertidumbre y nunca sustituyen el criterio pedagógico docente."
        )


def render_tab_analisis(metrics_data: dict) -> None:
    """Renderiza la Pestaña 1: Análisis Exploratorio y Evaluación del Modelo."""
    st.subheader("📌 1. Descripción del Proyecto y Contextualización")
    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.markdown(
            """
            El **Asistente de Inteligencia Artificial para la Planificación Docente** es un sistema de apoyo a la decisión pedagógica fundamentado en analítica del aprendizaje (*Learning Analytics*) y aprendizaje automático supervisado.

            Su propósito es asistir al cuerpo docente y a los equipos de coordinación académica en la **identificación temprana de patrones asociados al fracaso o retiro estudiantil**, facilitando el diseño anticipado de estrategias de acompañamiento antes de que el resultado académico sea irreversible.

            El problema se formula como una tarea de **clasificación binaria supervisada**:
            - **Clase `0` — Fracaso académico o retiro (`Fail` / `Withdrawn`):** Estudiantes que reprueban o se retiran formalmente del módulo.
            - **Clase `1` — Éxito académico (`Pass` / `Distinction`):** Estudiantes que aprueban satisfactoriamente o con distinción.
            """
        )
    with col_intro2:
        ds_info = metrics_data.get("dataset_summary", {})
        tgt_dist = ds_info.get("target_distribution", {})
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("#### 📊 Resumen del Dataset OULAD (`studentInfo.csv`)")
        st.write(f"- **Total de registros analizados:** `{ds_info.get('total_rows', 32593):,}` estudiantes")
        st.write(f"- **Conjunto de entrenamiento (80%):** `{ds_info.get('train_rows', 26074):,}` registros")
        st.write(f"- **Conjunto de prueba (20%):** `{ds_info.get('test_rows', 6519):,}` registros")
        st.write(
            f"- **Tasa de Fracaso / Retiro (Clase 0):** `{tgt_dist.get('pct_0_fracaso_retiro', 52.8)}%` (`{tgt_dist.get('0_fracaso_retiro', 17208):,}` casos)"
        )
        st.write(
            f"- **Tasa de Éxito Académico (Clase 1):** `{tgt_dist.get('pct_1_exito_academico', 47.2)}%` (`{tgt_dist.get('1_exito_academico', 15385):,}` casos)"
        )
        st.write("- **Variables predictoras originales:** `10` (`2` numéricas, `8` categóricas)")
        st.write(f"- **Atributos tras One-Hot Encoding:** `{ds_info.get('encoded_features_count', 40)}` columnas")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("🔎 2. Principales Hallazgos del Análisis Exploratorio de Datos (EDA)")
    st.markdown(
        """
        A continuación se presentan las visualizaciones generadas directamente desde el dataset **Open University Learning Analytics Dataset (OULAD)** junto con su interpretación pedagógica para la planificación docente:
        """
    )

    eda_row1_col1, eda_row1_col2 = st.columns(2)
    with eda_row1_col1:
        fig1_path = ASSETS_IMG_DIR / "fig1_distribucion_resultados.png"
        if fig1_path.exists():
            st.image(str(fig1_path), use_container_width=True)
        st.caption(
            "**Interpretación pedagógica:** Más de la mitad de las matrículas (52.8%) culminan en reprobación (21.6%) o retiro (31.2%), lo que confirma un problema estructural de retención que requiere intervención temprana desde el diseño instruccional."
        )

    with eda_row1_col2:
        fig2_path = ASSETS_IMG_DIR / "fig2_intentos_previos.png"
        if fig2_path.exists():
            st.image(str(fig2_path), use_container_width=True)
        st.caption(
            "**Interpretación pedagógica:** La tasa de éxito cae progresivamente desde un 49.4% en estudiantes sin intentos previos hasta menos del 34% a partir del segundo intento fallido. Los estudiantes recursantes requieren metodologías diferenciadas y tutoría focalizada."
        )

    eda_row2_col1, eda_row2_col2 = st.columns(2)
    with eda_row2_col1:
        fig3_path = ASSETS_IMG_DIR / "fig3_creditos_educacion.png"
        if fig3_path.exists():
            st.image(str(fig3_path), use_container_width=True)
        st.caption(
            "**Interpretación pedagógica:** Los estudiantes con titulaciones inferiores a *A Level* o sin cualificación formal presentan tasas de fracaso superiores al 63%, mientras que una sobrecarga de créditos inscritos (>120 créditos simultáneos) incrementa el riesgo de abandono."
        )

    with eda_row2_col2:
        fig4_path = ASSETS_IMG_DIR / "fig4_correlacion_imd.png"
        if fig4_path.exists():
            st.image(str(fig4_path), use_container_width=True)
        st.caption(
            "**Interpretación pedagógica:** Se observa un gradiente claro según el índice de privación socioeconómica (`imd_band`): los estudiantes en entornos de mayor vulnerabilidad (0-20%) aprueban en menor proporción (~35-40%) que aquellos en los deciles de menor privación (80-100%, >57%)."
        )

    st.divider()
    st.subheader("📈 3. Evaluación y Comparación de Modelos Supervisados")
    st.markdown(
        """
        Siguiendo las mejores prácticas de ingeniería de Machine Learning, todas las transformaciones que aprenden parámetros (`SimpleImputer` por moda/mediana, `StandardScaler` y `OneHotEncoder`) se ajustaron **exclusivamente sobre el subconjunto de entrenamiento (`26,074` estudiantes)** mediante un `Pipeline` unificado de Scikit-learn, previniendo cualquier fuga de información (*Data Leakage*) hacia el conjunto de prueba (`6,519` estudiantes).
        """
    )

    models_dict = metrics_data.get("models_evaluation", {})
    if models_dict:
        rf_eval = models_dict.get("Random Forest", {})
        rf_test = rf_eval.get("test_metrics", {})
        rf_cv = rf_eval.get("cv_5fold", {})

        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("Accuracy (RF)", f"{rf_test.get('accuracy', 0)*100:.2f}%")
        m_col2.metric("Balanced Acc.", f"{rf_test.get('balanced_accuracy', 0)*100:.2f}%")
        m_col3.metric("ROC-AUC (RF)", f"{rf_test.get('roc_auc', 0):.4f}")
        m_col4.metric("F1 Clase 0 (Riesgo)", f"{rf_test.get('f1_class_0_fracaso', 0):.4f}")
        m_col5.metric("Recall Clase 0 (Riesgo)", f"{rf_test.get('recall_class_0_fracaso', 0)*100:.2f}%")
        m_col6.metric("CV 5-Fold ROC-AUC", f"{rf_cv.get('roc_auc', {}).get('mean', 0):.4f}")

        st.markdown("#### Tabla Comparativa de Modelos en el Conjunto de Prueba (`n = 6,519`)")
        comp_rows = []
        for model_name, m_obj in models_dict.items():
            tm = m_obj.get("test_metrics", {})
            cvm = m_obj.get("cv_5fold", {}).get("roc_auc", {})
            comp_rows.append(
                {
                    "Modelo Supervisado": model_name,
                    "Accuracy": tm.get("accuracy"),
                    "Balanced Accuracy": tm.get("balanced_accuracy"),
                    "Precision Clase 0 (Riesgo)": tm.get("precision_class_0_fracaso"),
                    "Recall Clase 0 (Riesgo)": tm.get("recall_class_0_fracaso"),
                    "F1 Clase 0 (Riesgo)": tm.get("f1_class_0_fracaso"),
                    "Precision Clase 1 (Éxito)": tm.get("precision_class_1_exito"),
                    "Recall Clase 1 (Éxito)": tm.get("recall_class_1_exito"),
                    "F1 Clase 1 (Éxito)": tm.get("f1_class_1_exito"),
                    "ROC-AUC": tm.get("roc_auc"),
                    "CV 5-Fold ROC-AUC (Media ± DE)": f"{cvm.get('mean', 0):.4f} ± {cvm.get('std', 0):.4f}",
                }
            )
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        fig5_path = ASSETS_IMG_DIR / "fig5_matrices_confusion.png"
        if fig5_path.exists():
            st.image(str(fig5_path), use_container_width=True)
        st.markdown(
            """
            **Análisis de errores críticos en planificación docente:**  
            En un sistema de alerta temprana educativa, el error más costoso pedagógicamente es clasificar como fuera de peligro a un estudiante que finalmente reprueba o se retira (`1,373` casos en el conjunto de prueba de `6,519`), ya que podría privarlo de acompañamiento oportuno. El modelo **Random Forest Classifier** logra identificar correctamente **`2,069` de los `3,442` casos de fracaso o retiro (`60.11%` de Recall en Clase 0)** y **`1,969` de los `3,077` casos de éxito (`63.99%` de Recall en Clase 1)** empleando exclusivamente variables de matrícula inicial.
            """
        )

    with eval_col2:
        fig6_path = ASSETS_IMG_DIR / "fig6_curvas_roc.png"
        if fig6_path.exists():
            st.image(str(fig6_path), use_container_width=True)
        temp_eval = metrics_data.get("temporal_cohort_evaluation", {})
        if temp_eval:
            st.info(
                f"**⏱️ Validación por Cohorte Temporal ({temp_eval.get('test_cohort', '2014J')}):** "
                f"Al entrenar con cohortes históricas (`2013B`, `2013J`, `2014B`; `n = {temp_eval.get('train_size', 21333):,}`) "
                f"y evaluar en la cohorte futura independiente `2014J` (`n = {temp_eval.get('test_size', 11260):,}`), "
                f"el modelo mantiene un **Accuracy de `{temp_eval.get('accuracy', 0)*100:.2f}%`**, un **ROC-AUC de `{temp_eval.get('roc_auc', 0):.4f}`** "
                f"y un **F1 en Clase 0 (Riesgo) de `{temp_eval.get('f1_class_0_fracaso', 0):.4f}`**, "
                f"demostrando estabilidad inter-semestral sin degradación por deriva temporal."
            )

    st.divider()
    st.subheader("🔑 4. Importancia de Variables Predictoras")
    imp_col1, imp_col2 = st.columns([3, 2])
    with imp_col1:
        fig7_path = ASSETS_IMG_DIR / "fig7_importancia_variables.png"
        if fig7_path.exists():
            st.image(str(fig7_path), use_container_width=True)
    with imp_col2:
        st.markdown("#### Interpretación de las Variables más Influyentes")
        st.markdown(
            """
            1. **Créditos matriculados (`studied_credits` — `16.78%`):** Es el predictor individual con mayor peso relativo. Cargas académicas elevadas incrementan la probabilidad de retiro o fracaso.
            2. **Nivel educativo previo (`highest_education_Lower Than A Level` — `14.01%`):** Ingresar con escolaridad inferior a *A Level* constituye el segundo factor de mayor peso en la estructura de los árboles.
            3. **Intentos fallidos previos (`num_of_prev_attempts` — `8.03%`):** Refleja rezago acumulado en el módulo y es un indicador directo para activar tutorías de nivelación.
            4. **Módulo académico (`code_module_CCC`, `GGG`, `DDD`):** Captura la complejidad diferencial de asignaturas STEM frente a ciencias sociales.
            5. **Edad (`age_band_35-55`), Discapacidad (`disability_Y`) e Índice Socioeconómico (`imd_band`):** Evidencian el rol de las condiciones de vida y accesibilidad en la persistencia universitaria.
            """
        )

    st.divider()
    st.subheader("📋 5. Perfilado Automático (YData Profiling) y Limitaciones del Sistema")
    prof_col1, prof_col2 = st.columns(2)
    with prof_col1:
        st.markdown("#### Informe de Perfilado Automático (`YData Profiling`)")
        st.markdown(
            """
            Se ejecutó el perfilado estadístico automatizado sobre los `32,593` registros de `studentInfo.csv`, auditando:
            - **Completitud:** `99.7%` de celdas completas; valores faltantes concentrados únicamente en `imd_band` (`1,111` registros, `3.41%`).
            - **Unicidad:** `0` filas duplicadas exactas.
            - **Alertas estadísticas:** Asimetría positiva en `num_of_prev_attempts` (`87.2%` de ceros) y alta cardinalidad en `id_student` (excluido del modelado).
            """
        )
        report_html_path = REPORTS_DIR / "Reporte_Perfilado_OULAD.html"
        if report_html_path.exists():
            with open(report_html_path, "rb") as f_html:
                st.download_button(
                    label="📥 Descargar Informe Completo YData Profiling (HTML)",
                    data=f_html,
                    file_name="Reporte_Perfilado_OULAD.html",
                    mime="text/html",
                )
            with st.expander("👁️ Previsualizar Informe YData Profiling integrado"):
                html_text = report_html_path.read_text(encoding="utf-8", errors="ignore")
                components.html(html_text, height=550, scrolling=True)

    with prof_col2:
        st.markdown("#### ⚠️ Limitaciones Metodológicas y del Sistema")
        st.markdown(
            """
            - **Alcance de las variables de entrada:** El modelo utiliza variables sociodemográficas y de matrícula inicial (`studentInfo.csv`), lo que permite generar alertas **desde el día 1 de clases**, pero acota el desempeño máximo alcanzable (`Accuracy ~61.9%–62.4%`, `ROC-AUC ~0.665–0.670`) al no incluir todavía calificaciones intermedias ni clics diarios en el entorno virtual (VLE).
            - **Contexto institucional de origen:** El dataset OULAD proviene de *The Open University* (educación superior británica a distancia, 2013–2014). Su aplicación en universidades presenciales o semipresenciales de República Dominicana (como UFHEC) requiere recalibración con datos locales.
            - **Ausencia de causalidad:** Las asociaciones identificadas son correlacionales y predictivas, no causales.
            """
        )


def render_tab_simulador(pipeline) -> None:
    """Renderiza la Pestaña 2: Simulador Inteligente de Planificación Docente."""
    st.subheader("🎯 Simulador Inteligente de Planificación Docente")
    st.markdown(
        """
        Seleccione un **escenario pedagógico preconfigurado** para explorar casos representativos o personalice manualmente las características académicas y sociodemográficas del estudiante en el formulario inferior.
        """
    )

    scenario_names = list(PREDEFINED_SCENARIOS.keys())
    selected_scenario = st.selectbox(
        "📂 Cargar perfil de referencia rápido:",
        options=scenario_names,
        index=1,
        help="Permite cargar perfiles representativos de riesgo alto, moderado o bajo para probar el simulador.",
    )

    if selected_scenario != "Personalizado (Ingreso manual)" and PREDEFINED_SCENARIOS[selected_scenario]:
        defaults = PREDEFINED_SCENARIOS[selected_scenario]
    else:
        defaults = PREDEFINED_SCENARIOS["Escenario 1: Reingreso con sobrecarga académica (Alerta Alta)"]

    with st.form("student_simulation_form"):
        st.markdown("#### 📝 Formulario de Características Académicas y Sociodemográficas")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            mod_opts = FEATURE_DOMAINS["code_module"]["options"]
            code_module = st.selectbox(
                FEATURE_DOMAINS["code_module"]["label"],
                options=mod_opts,
                index=mod_opts.index(defaults["code_module"]),
                help=FEATURE_DOMAINS["code_module"]["description"],
            )
            pres_opts = FEATURE_DOMAINS["code_presentation"]["options"]
            code_presentation = st.selectbox(
                FEATURE_DOMAINS["code_presentation"]["label"],
                options=pres_opts,
                index=pres_opts.index(defaults["code_presentation"]),
                help=FEATURE_DOMAINS["code_presentation"]["description"],
            )
            studied_credits = st.slider(
                FEATURE_DOMAINS["studied_credits"]["label"],
                min_value=int(FEATURE_DOMAINS["studied_credits"]["min"]),
                max_value=int(FEATURE_DOMAINS["studied_credits"]["max"]),
                value=int(defaults["studied_credits"]),
                step=5,
                help=FEATURE_DOMAINS["studied_credits"]["description"],
            )
            num_of_prev_attempts = st.number_input(
                FEATURE_DOMAINS["num_of_prev_attempts"]["label"],
                min_value=int(FEATURE_DOMAINS["num_of_prev_attempts"]["min"]),
                max_value=int(FEATURE_DOMAINS["num_of_prev_attempts"]["max"]),
                value=int(defaults["num_of_prev_attempts"]),
                step=1,
                help=FEATURE_DOMAINS["num_of_prev_attempts"]["description"],
            )

        with f_col2:
            edu_opts = FEATURE_DOMAINS["highest_education"]["options"]
            highest_education = st.selectbox(
                FEATURE_DOMAINS["highest_education"]["label"],
                options=edu_opts,
                index=edu_opts.index(defaults["highest_education"]),
                help=FEATURE_DOMAINS["highest_education"]["description"],
            )
            imd_opts = FEATURE_DOMAINS["imd_band"]["options"]
            imd_band = st.selectbox(
                FEATURE_DOMAINS["imd_band"]["label"],
                options=imd_opts,
                index=imd_opts.index(defaults["imd_band"]),
                help=FEATURE_DOMAINS["imd_band"]["description"],
            )
            age_opts = FEATURE_DOMAINS["age_band"]["options"]
            age_band = st.selectbox(
                FEATURE_DOMAINS["age_band"]["label"],
                options=age_opts,
                index=age_opts.index(defaults["age_band"]),
                help=FEATURE_DOMAINS["age_band"]["description"],
            )

        with f_col3:
            gen_opts = FEATURE_DOMAINS["gender"]["options"]
            gender = st.selectbox(
                FEATURE_DOMAINS["gender"]["label"],
                options=gen_opts,
                index=gen_opts.index(defaults["gender"]),
                format_func=lambda x: "Masculino (M)" if x == "M" else "Femenino (F)",
            )
            dis_opts = FEATURE_DOMAINS["disability"]["options"]
            disability = st.selectbox(
                FEATURE_DOMAINS["disability"]["label"],
                options=dis_opts,
                index=dis_opts.index(defaults["disability"]),
                format_func=lambda x: "Sí declara discapacidad (Y)" if x == "Y" else "No declara discapacidad (N)",
            )
            reg_opts = FEATURE_DOMAINS["region"]["options"]
            region = st.selectbox(
                FEATURE_DOMAINS["region"]["label"],
                options=reg_opts,
                index=reg_opts.index(defaults["region"]),
                help=FEATURE_DOMAINS["region"]["description"],
            )

        st.form_submit_button(
            "🔍 Ejecutar Predicción y Generar Plan de Intervención Docente",
            use_container_width=True,
        )

    student_input = {
        "code_module": code_module,
        "code_presentation": code_presentation,
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "disability": disability,
    }

    resultado = predict_student_outcome(student_input, pipeline=pipeline)
    if resultado.get("errores_validacion"):
        for err_msg in resultado["errores_validacion"]:
            st.warning(f"⚠️ Validación de entrada: {err_msg}")

    pred_mod = resultado["prediccion_modelo"]
    alerta = resultado["alerta"]
    prob_fail = pred_mod["probabilidad_fracaso"]
    prob_pass = pred_mod["probabilidad_exito"]

    st.divider()
    st.markdown("### 📊 1. Estimación Estadística del Modelo (`RandomForestClassifier`)")
    st.caption(
        "Esta sección muestra exclusivamente la salida probabilística del modelo supervisado entrenado con datos históricos de OULAD."
    )

    res_col1, res_col2, res_col3 = st.columns([2, 1.5, 1.5])
    with res_col1:
        if pred_mod["clase_predicha"] == 0:
            st.error(f"**Categoría Estimada (Clase 0):** {pred_mod['etiqueta']}")
        else:
            st.success(f"**Categoría Estimada (Clase 1):** {pred_mod['etiqueta']}")
        st.markdown(f"**Nivel de Alerta Temprana:** `{alerta['nivel']}` (`{alerta['badge']}`)")

    with res_col2:
        st.metric(
            label="Probabilidad Estimada de Fracaso / Retiro (Clase 0)",
            value=f"{prob_fail * 100:.1f}%",
        )
        st.progress(min(max(prob_fail, 0.0), 1.0))

    with res_col3:
        st.metric(
            label="Probabilidad Estimada de Éxito Académico (Clase 1)",
            value=f"{prob_pass * 100:.1f}%",
        )
        st.progress(min(max(prob_pass, 0.0), 1.0))

    st.info(f"🗣️ **Explicación en lenguaje claro:** {alerta['resumen']}")

    st.divider()
    st.markdown("### 🧭 2. Recomendaciones Pedagógicas Basadas en Reglas Explícitas")
    st.caption(
        "Nota de arquitectura: A diferencia de la probabilidad superior (generada por el modelo estadístico Random Forest), las siguientes recomendaciones provienen de un **motor determinista de reglas pedagógicas** que audita las condiciones académicas del estudiante en 7 dimensiones de planificación docente."
    )

    for idx, rec in enumerate(resultado["reglas_pedagogicas"], start=1):
        prioridad_badge = (
            "🔴 Prioridad Alta"
            if rec["prioridad"] == "Alta"
            else ("🟡 Prioridad Media" if rec["prioridad"] == "Media" else "🟢 Prioridad Estándar")
        )
        with st.expander(
            f"{rec['dimension']} — [{prioridad_badge}]",
            expanded=(rec["prioridad"] == "Alta" or idx <= 3),
        ):
            st.markdown(f"**Regla explícita evaluada:** `{rec['regla']}`")
            st.markdown(f"**Acción docente recomendada:** {rec['recomendacion']}")

    st.markdown(
        f'<div class="ethical-box">⚖️ <strong>{resultado["nota_etica"]}</strong></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Punto de entrada principal de la aplicación Streamlit."""
    load_custom_css()
    render_sidebar()

    st.markdown(
        """
        <div class="main-header">
            <span class="badge-ufhec">UFHEC · Maestría en IA Aplicada a la Educación e Investigación · IAE-2501</span>
            <h1>🎓 Asistente de Inteligencia Artificial para la Planificación Docente</h1>
            <p>Sistema de Predicción Temprana de Rendimiento Académico y Recomendaciones Pedagógicas Basado en el Dataset OULAD</p>
            <p><strong>Autora:</strong> Jhessika Lidabed Espaillat Espaillat &nbsp;|&nbsp; <strong>Docente:</strong> Feibert Guzmán</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pipeline = get_cached_pipeline()
        metrics_data = get_cached_metrics()
    except Exception as exc:
        st.error(
            f"No fue posible cargar los artefactos del modelo entrenado: {exc}. "
            "Verifique que la carpeta `models/` contenga `random_forest_oulad.joblib` y `comparative_metrics.json`."
        )
        return

    tab1, tab2 = st.tabs(
        [
            "📊 Pestaña 1: Análisis y Evaluación del Modelo",
            "🎯 Pestaña 2: Simulador Inteligente de Planificación Docente",
        ]
    )

    with tab1:
        render_tab_analisis(metrics_data)

    with tab2:
        render_tab_simulador(pipeline)


if __name__ == "__main__":
    main()
