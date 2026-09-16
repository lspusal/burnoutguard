#!/usr/bin/env python3
"""
Pipeline con datos OULAD reales para predicción de burnout académico

Este script:
1. Carga los datos OULAD reales
2. Extrae características de engagement, temporales, académicas
3. Entrena modelos ML (XGBoost, LightGBM, etc.)
4. Evalúa con métricas estándar (AUROC, AUPRC, F1)
5. Genera análisis SHAP

Uso:
    python oulad_data.py

Autor: Research Project
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('real_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def load_oulad_data(data_dir: str = 'data/') -> dict:
    """Cargar todos los archivos CSV de OULAD"""
    data_path = Path(data_dir)
    
    logger.info("="*70)
    logger.info("CARGANDO DATOS OULAD REALES")
    logger.info("="*70)
    
    # Cargar archivos
    files = {
        'studentVle': 'studentVle.csv',
        'studentInfo': 'studentInfo.csv',
        'studentAssessment': 'studentAssessment.csv',
        'assessments': 'assessments.csv',
        'courses': 'courses.csv',
        'vle': 'vle.csv',
        'studentRegistration': 'studentRegistration.csv'
    }
    
    data = {}
    for name, filename in files.items():
        filepath = data_path / filename
        if filepath.exists():
            data[name] = pd.read_csv(filepath)
            logger.info(f"  ✓ {name}: {len(data[name]):,} filas")
        else:
            logger.warning(f"  ✗ {filename} no encontrado")
    
    return data


def extract_features(data: dict) -> tuple:
    """
    Extraer características de OULAD para predicción de burnout
    
    Features:
    - Engagement: clicks totales, frecuencia, tendencia
    - Temporal: días sin actividad, volatilidad
    - Académico: scores, submissions tardías
    - Demográfico: edad, género, educación previa
    """
    logger.info("\n" + "="*70)
    logger.info("EXTRACCIÓN DE CARACTERÍSTICAS")
    logger.info("="*70)
    
    df_vle = data['studentVle']
    df_info = data['studentInfo']
    df_assess = data.get('studentAssessment', pd.DataFrame())
    df_assess_info = data.get('assessments', pd.DataFrame())
    
    # ========== FEATURES DE ENGAGEMENT ==========
    logger.info("\n1. Extrayendo features de engagement...")
    
    # Agregar por estudiante-curso
    engagement = df_vle.groupby(['id_student', 'code_module', 'code_presentation']).agg({
        'sum_click': ['sum', 'mean', 'std', 'max'],
        'date': ['min', 'max', 'count', 'nunique']
    }).reset_index()
    
    engagement.columns = [
        'id_student', 'code_module', 'code_presentation',
        'total_clicks', 'avg_clicks', 'std_clicks', 'max_clicks',
        'first_access_day', 'last_access_day', 'n_access_events', 'n_unique_days'
    ]
    
    # Calcular tendencia de engagement (slope)
    def calc_engagement_trend(group):
        if len(group) < 3:
            return 0
        days = group['date'].values
        clicks = group['sum_click'].values
        if np.std(days) == 0:
            return 0
        # Linear regression slope
        slope = np.polyfit(days, clicks, 1)[0]
        return slope
    
    trend = df_vle.groupby(['id_student', 'code_module', 'code_presentation']).apply(
        calc_engagement_trend
    ).reset_index(name='engagement_trend_slope')
    
    engagement = engagement.merge(trend, on=['id_student', 'code_module', 'code_presentation'])
    
    # ========== FEATURES TEMPORALES ==========
    logger.info("2. Extrayendo features temporales...")
    
    def calc_temporal_features(group):
        days = sorted(group['date'].unique())
        if len(days) < 2:
            return pd.Series({
                'max_gap_days': 0,
                'mean_gap_days': 0,
                'activity_volatility': 0,
                'negative_changes_pct': 0
            })
        
        gaps = np.diff(days)
        daily_clicks = group.groupby('date')['sum_click'].sum().values
        
        # Volatilidad y cambios negativos
        if len(daily_clicks) > 1:
            changes = np.diff(daily_clicks)
            neg_pct = (changes < 0).sum() / len(changes) if len(changes) > 0 else 0
            volatility = np.std(daily_clicks)
        else:
            neg_pct = 0
            volatility = 0
        
        return pd.Series({
            'max_gap_days': gaps.max() if len(gaps) > 0 else 0,
            'mean_gap_days': gaps.mean() if len(gaps) > 0 else 0,
            'activity_volatility': volatility,
            'negative_changes_pct': neg_pct
        })
    
    temporal = df_vle.groupby(['id_student', 'code_module', 'code_presentation']).apply(
        calc_temporal_features
    ).reset_index()
    
    engagement = engagement.merge(temporal, on=['id_student', 'code_module', 'code_presentation'])
    
    # ========== FEATURES ACADÉMICAS ==========
    logger.info("3. Extrayendo features académicas...")
    
    if len(df_assess) > 0 and len(df_assess_info) > 0:
        # Merge assessment info
        assess = df_assess.merge(df_assess_info, on='id_assessment')
        
        # Calcular features por estudiante-curso
        academic = assess.groupby(['id_student', 'code_module', 'code_presentation']).agg({
            'score': ['mean', 'std', 'min', 'max', 'count'],
            'is_banked': 'sum'
        }).reset_index()
        
        academic.columns = [
            'id_student', 'code_module', 'code_presentation',
            'mean_score', 'std_score', 'min_score', 'max_score', 
            'n_assessments', 'n_banked'
        ]
        
        # Late submissions (date > due_date)
        assess['is_late'] = assess['date_submitted'] > assess['date']
        late = assess.groupby(['id_student', 'code_module', 'code_presentation'])['is_late'].mean().reset_index()
        late.columns = ['id_student', 'code_module', 'code_presentation', 'late_submission_rate']
        
        academic = academic.merge(late, on=['id_student', 'code_module', 'code_presentation'], how='left')
        academic['late_submission_rate'] = academic['late_submission_rate'].fillna(0)
        
        # Grade trend
        def calc_grade_trend(group):
            if len(group) < 2:
                return 0
            dates = group['date_submitted'].values
            scores = group['score'].values
            valid = ~(np.isnan(dates) | np.isnan(scores))
            if valid.sum() < 2:
                return 0
            dates = dates[valid]
            scores = scores[valid]
            if np.std(dates) == 0:
                return 0
            return np.polyfit(dates, scores, 1)[0]
        
        grade_trend = assess.groupby(['id_student', 'code_module', 'code_presentation']).apply(
            calc_grade_trend
        ).reset_index(name='grade_trend_slope')
        
        academic = academic.merge(grade_trend, on=['id_student', 'code_module', 'code_presentation'], how='left')
        academic['grade_trend_slope'] = academic['grade_trend_slope'].fillna(0)
        
        engagement = engagement.merge(academic, on=['id_student', 'code_module', 'code_presentation'], how='left')
    
    # ========== FEATURES DEMOGRÁFICAS ==========
    logger.info("4. Extrayendo features demográficas...")
    
    # One-hot encode categorical variables
    df_info_encoded = df_info.copy()
    
    # Gender
    df_info_encoded['gender_M'] = (df_info_encoded['gender'] == 'M').astype(int)
    
    # Age band
    df_info_encoded['age_band_35_55'] = (df_info_encoded['age_band'] == '35-55').astype(int)
    df_info_encoded['age_band_55_plus'] = (df_info_encoded['age_band'] == '55<=').astype(int)
    
    # Disability
    df_info_encoded['has_disability'] = (df_info_encoded['disability'] == 'Y').astype(int)
    
    # Highest education
    edu_map = {
        'No Formal quals': 0,
        'Lower Than A Level': 1,
        'A Level or Equivalent': 2,
        'HE Qualification': 3,
        'Post Graduate Qualification': 4
    }
    df_info_encoded['education_level'] = df_info_encoded['highest_education'].map(edu_map).fillna(1)
    
    # IMD band (deprivation index)
    df_info_encoded['imd_band_num'] = df_info_encoded['imd_band'].str.extract(r'(\d+)').astype(float).fillna(50)
    
    # Studied credits
    df_info_encoded['studied_credits'] = df_info_encoded['studied_credits'].fillna(60)
    
    # Previous attempts
    df_info_encoded['num_of_prev_attempts'] = df_info_encoded['num_of_prev_attempts'].fillna(0)
    
    # ========== TARGET: BURNOUT RISK ==========
    # Using final_result as proxy: Withdrawn or Fail = high risk
    df_info_encoded['burnout_risk'] = df_info_encoded['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
    
    demo_cols = [
        'id_student', 'code_module', 'code_presentation',
        'gender_M', 'age_band_35_55', 'age_band_55_plus', 'has_disability',
        'education_level', 'imd_band_num', 'studied_credits', 'num_of_prev_attempts',
        'burnout_risk'
    ]
    
    demographics = df_info_encoded[demo_cols]
    
    # ========== MERGE TODO ==========
    logger.info("\n5. Combinando todas las características...")
    
    features = engagement.merge(
        demographics, 
        on=['id_student', 'code_module', 'code_presentation'],
        how='inner'
    )
    
    # Fill NaN
    features = features.fillna(0)
    
    # Separate X and y
    feature_cols = [c for c in features.columns if c not in 
                   ['id_student', 'code_module', 'code_presentation', 'burnout_risk']]
    
    X = features[feature_cols]
    y = features['burnout_risk']
    
    logger.info(f"\n✓ Features extraídas:")
    logger.info(f"  - Muestras: {len(X):,}")
    logger.info(f"  - Features: {len(feature_cols)}")
    logger.info(f"  - Distribución target: Low risk={sum(y==0):,}, High risk={sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def run_ml_pipeline(X: pd.DataFrame, y: pd.Series, feature_names: list, output_dir: str = 'results/'):
    """Ejecutar pipeline ML completo"""
    from sklearn.model_selection import train_test_split
    from src.model_training import BurnoutModelTrainer
    from src.evaluation import ModelEvaluator
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'metrics').mkdir(exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)
    
    # ========== SPLIT ==========
    logger.info("\n" + "="*70)
    logger.info("DIVISIÓN DE DATOS")
    logger.info("="*70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, stratify=y_temp, random_state=42
    )
    
    logger.info(f"  Train: {len(X_train):,} muestras")
    logger.info(f"  Val: {len(X_val):,} muestras")
    logger.info(f"  Test: {len(X_test):,} muestras")
    
    # ========== ENTRENAR ==========
    logger.info("\n" + "="*70)
    logger.info("ENTRENAMIENTO DE MODELOS")
    logger.info("="*70)
    
    trainer = BurnoutModelTrainer(output_dir=str(output_path / 'models'))
    
    # Baseline
    trainer.train_baseline_models(X_train, y_train, X_val, y_val)
    
    # Advanced
    trainer.train_advanced_models(X_train, y_train, X_val, y_val)
    
    # Ensemble
    try:
        trainer.create_ensemble()
    except Exception as e:
        logger.warning(f"No se pudo crear ensemble: {e}")
    
    # ========== EVALUAR ==========
    logger.info("\n" + "="*70)
    logger.info("EVALUACIÓN DE MODELOS")
    logger.info("="*70)
    
    evaluator = ModelEvaluator(output_dir=str(output_path / 'metrics'))
    results_df = evaluator.evaluate_multiple_models(trainer.trained_models, X_test, y_test)
    
    # Generar figuras
    evaluator.plot_roc_curves(
        trainer.trained_models, X_test, y_test,
        save_path=str(output_path / 'figures' / 'roc_curves.png')
    )
    
    evaluator.plot_pr_curves(
        trainer.trained_models, X_test, y_test,
        save_path=str(output_path / 'figures' / 'pr_curves.png')
    )
    
    # Mejor modelo
    best_model_name = results_df['auroc'].idxmax()
    best_model = trainer.trained_models[best_model_name]
    
    y_pred = best_model.predict(X_test)
    evaluator.plot_confusion_matrix(
        y_test, y_pred, best_model_name,
        save_path=str(output_path / 'figures' / 'confusion_matrix.png')
    )
    
    evaluator.plot_calibration_curve(
        best_model, X_test, y_test, best_model_name,
        save_path=str(output_path / 'figures' / 'calibration.png')
    )
    
    # ========== SHAP ==========
    logger.info("\n" + "="*70)
    logger.info("ANÁLISIS DE INTERPRETABILIDAD (SHAP)")
    logger.info("="*70)
    
    try:
        from src.explainability import SHAPAnalyzer
        
        analyzer = SHAPAnalyzer(output_dir=str(output_path / 'figures'))
        
        # Use XGBoost or LightGBM for SHAP
        shap_model = trainer.trained_models.get('xgboost', trainer.trained_models.get('lightgbm', best_model))
        
        analyzer.compute_shap_values(shap_model, X_test, sample_size=min(1000, len(X_test)))
        analyzer.summary_plot(save_path=str(output_path / 'figures' / 'shap_summary.png'))
        analyzer.bar_plot(save_path=str(output_path / 'figures' / 'shap_bar.png'))
        
        importance_df = analyzer.get_feature_importance(top_n=20)
        importance_df.to_csv(output_path / 'metrics' / 'feature_importance.csv', index=False)
        
    except Exception as e:
        logger.warning(f"Error en SHAP: {e}")
    
    # ========== RESUMEN ==========
    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE RESULTADOS")
    logger.info("="*70)
    
    print("\n" + results_df.to_string())
    
    logger.info(f"\n✓ Mejor modelo: {best_model_name}")
    logger.info(f"  - AUROC: {results_df.loc[best_model_name, 'auroc']:.4f}")
    logger.info(f"  - AUPRC: {results_df.loc[best_model_name, 'auprc']:.4f}")
    logger.info(f"  - F1: {results_df.loc[best_model_name, 'f1_score']:.4f}")
    
    results_df.to_csv(output_path / 'metrics' / 'model_results.csv')
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("="*70)
    
    return results_df, trainer.trained_models


def main():
    # Cargar datos
    data = load_oulad_data('data/')
    
    if 'studentVle' not in data or 'studentInfo' not in data:
        logger.error("Datos OULAD no encontrados. Ejecutar primero la descarga.")
        sys.exit(1)
    
    # Extraer features
    X, y, feature_names = extract_features(data)
    
    # Ejecutar pipeline ML
    results, models = run_ml_pipeline(X, y, feature_names)
    
    return results


if __name__ == '__main__':
    main()
