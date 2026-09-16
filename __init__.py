"""
Paquete de predicción de burnout académico

Módulos:
- data_processing: Carga y preprocesamiento de OULAD
- feature_engineering: Extracción de características
- model_training: Entrenamiento de modelos
- evaluation: Evaluación de rendimiento
- explainability: Análisis SHAP

Uso:
    from src.data_processing import load_and_preprocess_oulad
    from src.feature_engineering import extract_burnout_features
    from src.model_training import train_models
    from src.evaluation import evaluate_models
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__all__ = [
    'data_processing',
    'feature_engineering',
    'model_training',
    'evaluation',
    'explainability'
]
