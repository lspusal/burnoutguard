#!/usr/bin/env python3
"""
EWS-Enhanced: Sistema de Alerta Temprana Mejorado

Contribuciones Técnicas:
1. Features de "decay pattern" - patrones de desvinculación en primeros días
2. Comparación con comportamiento de pares exitosos
3. Early engagement score normalizado
4. Modelo calibrado para intervención temprana

Target: Superar AUROC 0.77 en semana 4

Autor: Research Project
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from risk_set import apply_risk_set
from horizon_features import add_horizon_features
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_enhanced_features(df_vle, df_info, week: int):
    """
    Extraer features mejoradas para predicción temprana
    
    Nuevas features:
    - Engagement decay (cambio día a día)
    - Missing days pattern
    - Week-over-week ratio
    - Percentile vs peers
    """
    max_day = week * 7
    df_vle_filtered = df_vle[df_vle['date'] <= max_day].copy()
    
    if len(df_vle_filtered) == 0:
        return None, None
    
    # === FEATURES BÁSICAS ===
    basic = df_vle_filtered.groupby(['id_student', 'code_module', 'code_presentation']).agg({
        'sum_click': ['sum', 'mean', 'std', 'max', 'min', 'count'],
        'date': ['min', 'max', 'nunique']
    }).reset_index()
    
    basic.columns = [
        'id_student', 'code_module', 'code_presentation',
        'total_clicks', 'avg_clicks', 'std_clicks', 'max_clicks', 'min_clicks', 'n_events',
        'first_day', 'last_day', 'active_days'
    ]
    
    # === FEATURES DE DESVINCULACIÓN ===
    # 1. Días sin actividad recientes
    basic['days_inactive'] = max_day - basic['last_day']
    basic['inactivity_ratio'] = basic['days_inactive'] / max_day
    
    # 2. Engagement decay: comparar últimos días vs primeros días
    half_point = max_day // 2
    
    first_half = df_vle_filtered[df_vle_filtered['date'] <= half_point].groupby(
        ['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()
    first_half.columns = ['id_student', 'code_module', 'code_presentation', 'clicks_early']
    
    second_half = df_vle_filtered[df_vle_filtered['date'] > half_point].groupby(
        ['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()
    second_half.columns = ['id_student', 'code_module', 'code_presentation', 'clicks_late']
    
    basic = basic.merge(first_half, on=['id_student', 'code_module', 'code_presentation'], how='left')
    basic = basic.merge(second_half, on=['id_student', 'code_module', 'code_presentation'], how='left')
    basic = basic.fillna(0)
    
    # Decay ratio: < 1 significa menos actividad en segunda mitad = señal de riesgo
    basic['engagement_decay'] = (basic['clicks_late'] + 1) / (basic['clicks_early'] + 1)
    basic['decay_flag'] = (basic['engagement_decay'] < 0.5).astype(int)  # 50% menos actividad
    
    # 3. Pattern de acceso: consecutivo vs esporádico
    def calc_access_pattern(group):
        days = sorted(group['date'].unique())
        if len(days) < 2:
            return pd.Series({'gap_mean': max_day, 'gap_max': max_day, 'regularity': 0})
        gaps = np.diff(days)
        return pd.Series({
            'gap_mean': gaps.mean(),
            'gap_max': gaps.max(),
            'regularity': 1 / (gaps.std() + 1)  # Mayor regularidad = menor std
        })
    
    patterns = df_vle_filtered.groupby(
        ['id_student', 'code_module', 'code_presentation']).apply(calc_access_pattern).reset_index()
    basic = basic.merge(patterns, on=['id_student', 'code_module', 'code_presentation'], how='left')
    
    # 4. Percentile dentro del curso (comparación con pares)
    for feature in ['total_clicks', 'active_days']:
        course_stats = basic.groupby(['code_module', 'code_presentation'])[feature].transform(
            lambda x: (x - x.mean()) / (x.std() + 1)
        )
        basic[f'{feature}_zscore'] = course_stats
    
    # 5. Click velocity (intensidad cuando está activo)
    basic['click_velocity'] = basic['total_clicks'] / (basic['active_days'] + 1)
    
    # 6. Early starter flag
    basic['early_start'] = (basic['first_day'] <= 7).astype(int)  # Empezó primera semana
    
    # 7. Consistency score
    expected_days = max_day / 2  # Esperamos actividad ~50% de los días
    basic['consistency'] = basic['active_days'] / expected_days
    
    # === MERGE CON TARGET ===
    df_info['burnout'] = df_info['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
    
    features = basic.merge(
        df_info[['id_student', 'code_module', 'code_presentation', 'burnout']],
        on=['id_student', 'code_module', 'code_presentation'],
        how='inner'
    )
    features = features.fillna(0)
    # sólo estudiantes aún matriculados en el horizonte (ver risk_set.py)
    features = apply_risk_set(features, max_day)
    features, _added = add_horizon_features(features, week)
    features = features.drop(columns=[c for c in features.columns
                                      if c.endswith("_dup") or c in
                                      ["last_submit", "gender", "age_band", "imd_band",
                                       "disability", "highest_education", "final_result"]],
                             errors="ignore")
    
    feature_cols = [c for c in features.columns if c not in 
                   ['id_student', 'code_module', 'code_presentation', 'burnout']]
    
    X = features[feature_cols]
    y = features['burnout']
    
    return X, y


def train_enhanced_model(X_train, y_train, X_val, y_val):
    """Entrenar modelo mejorado con optimización de hiperparámetros"""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Modelo 1: XGBoost optimizado
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )
    
    # Modelo 2: LightGBM optimizado
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1
    )
    
    # Modelo 3: GBM con diferentes hiperparámetros
    gbm_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    # Stacking
    stacking = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('gbm', gbm_model),
        ],
        final_estimator=LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced'),
        cv=5,
        n_jobs=-1,
        passthrough=True  # Incluir features originales
    )
    
    stacking.fit(X_train, y_train)
    
    return stacking


def run_enhanced_prediction():
    """Pipeline principal mejorado"""
    logger.info("="*70)
    logger.info("EWS-ENHANCED: PREDICCIÓN TEMPRANA MEJORADA")
    logger.info("="*70)
    
    # Cargar datos
    df_vle = pd.read_csv('data/studentVle.csv')
    df_info = pd.read_csv('data/studentInfo.csv')
    
    logger.info(f"  VLE: {len(df_vle):,} interacciones")
    logger.info(f"  Students: {len(df_info):,}")
    
    # Evaluar en diferentes semanas
    results = []
    
    for week in [2, 4, 6, 8]:
        logger.info(f"\n--- Semana {week} ---")
        
        X, y = extract_enhanced_features(df_vle, df_info, week)
        
        if X is None:
            continue
        
        logger.info(f"  Features: {X.shape[1]}, Samples: {len(X)}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        # Train
        model = train_enhanced_model(X_train_s, y_train, X_val_s, y_val)
        
        # Evaluate
        y_proba = model.predict_proba(X_test_s)[:, 1]
        y_pred = model.predict(X_test_s)
        
        auroc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'week': week,
            'auroc': auroc,
            'f1': f1,
            'accuracy': acc,
            'n_features': X.shape[1]
        })
        
        logger.info(f"  AUROC: {auroc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")
    
    # Tabla resumen
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Guardar
    results_df.to_csv('results/metrics/ews_enhanced_results.csv', index=False)
    
    # Comparación con baseline
    logger.info("\n" + "="*70)
    logger.info("COMPARACIÓN CON BASELINE")
    logger.info("="*70)
    
    baseline_week4 = 0.7701  # XGBoost básico
    enhanced_week4 = results_df[results_df['week'] == 4]['auroc'].values[0]
    improvement = (enhanced_week4 - baseline_week4) / baseline_week4 * 100
    
    logger.info(f"  Baseline (semana 4): AUROC = {baseline_week4:.4f}")
    logger.info(f"  Enhanced (semana 4): AUROC = {enhanced_week4:.4f}")
    logger.info(f"  Mejora: {improvement:+.2f}%")
    
    return results_df


if __name__ == '__main__':
    run_enhanced_prediction()
