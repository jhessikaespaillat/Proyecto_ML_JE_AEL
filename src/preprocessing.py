"""
Módulo de Ingesta, Limpieza y Preprocesamiento Libre de Fuga de Información (Data Leakage).
Proyecto: Asistente de Inteligencia Artificial para la Planificación Docente (OULAD).
"""

from pathlib import Path
import zipfile
from typing import Tuple, Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Definición canónica de columnas del dataset OULAD (studentInfo.csv)
NUMERIC_FEATURES: List[str] = [
    "num_of_prev_attempts",
    "studied_credits",
]

CATEGORICAL_FEATURES: List[str] = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

ALL_INPUT_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_MAPPING: Dict[str, int] = {
    "Pass": 1,
    "Distinction": 1,
    "Fail": 0,
    "Withdrawn": 0,
}

# Diccionario descriptivo y dominios válidos para validación de entradas en el simulador
FEATURE_DOMAINS: Dict[str, Dict[str, Any]] = {
    "code_module": {
        "label": "Módulo Académico (Asignatura)",
        "options": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        "default": "BBB",
        "description": "Código identificador del módulo o asignatura cursada en The Open University.",
    },
    "code_presentation": {
        "label": "Cohorte / Presentación Académica",
        "options": ["2013B", "2013J", "2014B", "2014J"],
        "default": "2014J",
        "description": "Periodo de inicio del curso (B = Febrero, J = Octubre).",
    },
    "gender": {
        "label": "Género declarado",
        "options": ["F", "M"],
        "default": "F",
        "description": "Género registrado del estudiante (F = Femenino, M = Masculino).",
    },
    "region": {
        "label": "Región Geográfica",
        "options": [
            "East Anglian Region",
            "Scotland",
            "North Western Region",
            "South East Region",
            "West Midlands Region",
            "Wales",
            "North Region",
            "South Region",
            "Ireland",
            "South West Region",
            "East Midlands Region",
            "Yorkshire Region",
            "London Region",
        ],
        "default": "Scotland",
        "description": "Región geográfica de residencia del estudiante en el Reino Unido.",
    },
    "highest_education": {
        "label": "Nivel Educativo Previo",
        "options": [
            "No Formal quals",
            "Lower Than A Level",
            "A Level or Equivalent",
            "HE Qualification",
            "Post Graduate Qualification",
        ],
        "default": "A Level or Equivalent",
        "description": "Máximo nivel de escolaridad completado al momento de la matrícula.",
    },
    "imd_band": {
        "label": "Banda Socioeconómica (Índice IMD)",
        "options": [
            "0-10%",
            "10-20",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
        ],
        "default": "30-40%",
        "description": "Índice de Privación Múltiple (Index of Multiple Deprivation) de la zona de residencia.",
    },
    "age_band": {
        "label": "Rango de Edad",
        "options": ["0-35", "35-55", "55<="],
        "default": "0-35",
        "description": "Grupo etario del estudiante al iniciar el módulo.",
    },
    "disability": {
        "label": "Condición de Discapacidad Declarada",
        "options": ["N", "Y"],
        "default": "N",
        "description": "Indica si el estudiante ha reportado alguna condición de discapacidad (N = No, Y = Sí).",
    },
    "num_of_prev_attempts": {
        "label": "Número de Intentos Previos en el Módulo",
        "min": 0,
        "max": 6,
        "default": 0,
        "description": "Cantidad de veces anteriores que el estudiante ha intentado cursar este módulo.",
    },
    "studied_credits": {
        "label": "Créditos Académicos Matriculados",
        "min": 30,
        "max": 655,
        "default": 60,
        "description": "Carga académica total (en créditos) que el estudiante cursa simultáneamente.",
    },
}


def load_oulad_student_info(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga el archivo studentInfo.csv desde data/raw/studentInfo.csv o directamente
    desde el archivo local archive.zip sin depender de rutas absolutas externas.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    csv_candidates = [
        base_dir / "data" / "raw" / "studentInfo.csv",
        base_dir / "studentInfo.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            return pd.read_csv(csv_path)

    zip_candidates = [
        base_dir / "archive.zip",
        base_dir / "archive - copia.zip",
    ]
    for zip_path in zip_candidates:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open("studentInfo.csv") as f:
                    return pd.read_csv(f)

    raise FileNotFoundError(
        "No se encontró 'studentInfo.csv' ni 'archive.zip' en el directorio del proyecto."
    )


def clean_and_prepare_target(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza estructural (sin ajustar parámetros estadísticos globales para
    prevenir fuga de información) y construye la variable objetivo binaria 'target':
      - 1: Éxito académico (Pass / Distinction)
      - 0: Fracaso académico o retiro (Fail / Withdrawn)
    """
    df = df_raw.copy()

    # Normalizar cadenas vacías o '?' como NaN reales
    df = df.replace(["?", "", "None", "nan"], np.nan)

    # Verificar que final_result exista y mapear a target binario
    if "final_result" not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'final_result'.")

    df = df[df["final_result"].isin(TARGET_MAPPING.keys())].copy()
    df["target"] = df["final_result"].map(TARGET_MAPPING).astype(int)

    return df


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Construye un ColumnTransformer de scikit-learn que encapsula:
      1. Imputación de mediana y escalamiento estándar opcional para variables numéricas.
      2. Imputación de la moda (most_frequent) y codificación One-Hot para variables categóricas.
    Al integrarse en un Pipeline y ajustarse solo sobre X_train, previene al 100% el data leakage.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_train_test_oulad(
    df_clean: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa las variables predictoras (X) y la variable objetivo (y) realizando una
    partición estratificada ANTES de cualquier imputación o codificación.
    """
    X = df_clean[ALL_INPUT_FEATURES].copy()
    y = df_clean["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def validate_and_build_input_df(user_input: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Valida un diccionario con las características de un estudiante y devuelve un DataFrame
    de 1 fila con las columnas, nombres, tipos y orden exactos empleados en el entrenamiento.
    No asigna ceros arbitrarios a columnas desconocidas.
    """
    errors: List[str] = []
    record: Dict[str, Any] = {}

    # Validar variables numéricas
    for col in NUMERIC_FEATURES:
        domain = FEATURE_DOMAINS[col]
        val = user_input.get(col, domain["default"])
        try:
            num_val = int(float(val))
            if num_val < domain["min"] or num_val > domain["max"]:
                errors.append(
                    f"La variable '{domain['label']}' ({num_val}) está fuera del rango permitido [{domain['min']}, {domain['max']}]."
                )
            record[col] = num_val
        except (ValueError, TypeError):
            errors.append(f"Valor numérico inválido para '{domain['label']}': {val}")
            record[col] = domain["default"]

    # Validar variables categóricas
    for col in CATEGORICAL_FEATURES:
        domain = FEATURE_DOMAINS[col]
        val = user_input.get(col, domain["default"])
        if val not in domain["options"]:
            errors.append(
                f"Valor '{val}' no reconocido para '{domain['label']}'. Opciones válidas: {domain['options']}"
            )
            record[col] = domain["default"]
        else:
            record[col] = str(val)

    df_single = pd.DataFrame([record], columns=ALL_INPUT_FEATURES)
    return df_single, errors

