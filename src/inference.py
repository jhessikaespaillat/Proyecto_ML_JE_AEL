"""
Módulo de Inferencia y Carga de Artefactos de Machine Learning.
Proyecto: Asistente de Inteligencia Artificial para la Planificación Docente (OULAD).
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional
import joblib
import pandas as pd

from src.preprocessing import validate_and_build_input_df
from src.recommendations import generar_recomendaciones_pedagogicas


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_trained_pipeline(model_path: Optional[Path] = None):
    """
    Carga el Pipeline serializado (Preprocesador ColumnTransformer + Clasificador Random Forest)
    desde models/random_forest_oulad.joblib.
    """
    if model_path is None:
        model_path = get_project_root() / "models" / "random_forest_oulad.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {model_path}. Ejecute primero el script de entrenamiento."
        )
    return joblib.load(model_path)


def load_evaluation_metrics(metrics_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Carga las métricas reales de evaluación y comparación de modelos generadas desde el cuaderno/entrenamiento.
    """
    if metrics_path is None:
        metrics_path = get_project_root() / "models" / "comparative_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de métricas reales en {metrics_path}."
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_student_outcome(
    user_input: Dict[str, Any],
    pipeline=None,
) -> Dict[str, Any]:
    """
    Valida las características introducidas, aplica el pipeline de preprocesamiento e inferencia
    sin asignar ceros arbitrarios, calcula probabilidades para ambas clases y ejecuta el motor
    de recomendaciones pedagógicas basado en reglas explícitas.
    """
    df_input, validation_errors = validate_and_build_input_df(user_input)

    if pipeline is None:
        pipeline = load_trained_pipeline()

    pred_class = int(pipeline.predict(df_input)[0])
    pred_proba = pipeline.predict_proba(df_input)[0]
    prob_fracaso = float(pred_proba[0])
    prob_exito = float(pred_proba[1])

    resultado_integral = generar_recomendaciones_pedagogicas(
        student_profile=df_input.iloc[0].to_dict(),
        predicted_class=pred_class,
        prob_fracaso=prob_fracaso,
        prob_exito=prob_exito,
    )
    resultado_integral["errores_validacion"] = validation_errors
    resultado_integral["datos_validados"] = df_input.iloc[0].to_dict()
    return resultado_integral

