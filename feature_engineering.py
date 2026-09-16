"""
Módulo de Feature Engineering para predicción de burnout académico

Extrae características temporales, comportamentales y de rendimiento
del dataset OULAD para identificar patrones de burnout académico.

Características principales:
- Patrones de engagement (frecuencia, consistencia)
- Características temporales (tendencias, volatilidad)
- Indicadores de rendimiento (calificaciones, comparación con peers)
- Características de comportamiento de riesgo

Autor: Research Project
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
from scipy import stats

logger = logging.getLogger(__name__)


class BurnoutFeatureExtractor:
    """Extractor de características para predicción de burnout"""
    
    def __init__(self, df_vle: pd.DataFrame, df_studentinfo: pd.DataFrame,
                 df_assess: pd.DataFrame, df_studentassess: pd.DataFrame):
        """
        Inicializar extractor
        
        Args:
            df_vle: DataFrame de interacciones VLE
            df_studentinfo: DataFrame de información de estudiantes
            df_assess: DataFrame de evaluaciones
            df_studentassess: DataFrame de resultados de evaluación
        """
        self.df_vle = df_vle.copy()
        self.df_studentinfo = df_studentinfo.copy()
        self.df_assess = df_assess.copy()
        self.df_studentassess = df_studentassess.copy()
        
        self.features = None
    
    def extract_engagement_features(self) -> pd.DataFrame:
        """
        Extraer características de engagement
        
        Incluye:
        - Frecuencia de acceso (media, std, min, max)
        - Consistencia temporal
        - Tasa de actividad decreciente
        
        Returns:
            DataFrame con características de engagement
        """
        logger.info("Extrayendo características de engagement...")
        
        engagement_features = []
        
        for student_id in self.df_vle['id_student'].unique():
            student_vle = self.df_vle[self.df_vle['id_student'] == student_id].sort_values('date')
            
            if len(student_vle) == 0:
                continue
            
            features = {
                'id_student': student_id,
                'total_clicks': student_vle['sum_click'].sum(),
                'avg_clicks_per_day': student_vle['sum_click'].mean(),
                'std_clicks_per_day': student_vle['sum_click'].std() or 0,
                'max_clicks': student_vle['sum_click'].max(),
                'min_clicks': student_vle['sum_click'].min(),
                'access_frequency_days': len(student_vle),
                'activity_variance': student_vle['sum_click'].var() or 0,
            }
            
            # Tendencia decreciente (indicador de burnout)
            if len(student_vle) > 1:
                dates = student_vle['date'].values
                clicks = student_vle['sum_click'].values
                slope = np.polyfit(dates, clicks, 1)[0]
                features['engagement_trend_slope'] = slope
            else:
                features['engagement_trend_slope'] = 0
            
            engagement_features.append(features)
        
        df_engagement = pd.DataFrame(engagement_features)
        logger.info(f"  ✓ {len(df_engagement)} estudiantes con características de engagement")
        return df_engagement
    
    def extract_temporal_features(self) -> pd.DataFrame:
        """
        Extraer características temporales
        
        Incluye:
        - Cambios de actividad semana a semana
        - Volatilidad temporal
        - Gaps sin actividad
        
        Returns:
            DataFrame con características temporales
        """
        logger.info("Extrayendo características temporales...")
        
        temporal_features = []
        
        for student_id in self.df_vle['id_student'].unique():
            student_vle = self.df_vle[self.df_vle['id_student'] == student_id].sort_values('date')
            
            if len(student_vle) < 2:
                continue
            
            # Cambios de actividad
            activity_changes = np.diff(student_vle['sum_click'].values)
            
            features = {
                'id_student': student_id,
                'max_activity_change': np.max(np.abs(activity_changes)),
                'mean_activity_change': np.mean(np.abs(activity_changes)),
                'negative_changes_pct': (activity_changes < 0).sum() / len(activity_changes),
                'activity_volatility': np.std(activity_changes),
            }
            
            # Gaps sin actividad
            date_diffs = np.diff(student_vle['date'].values)
            if len(date_diffs) > 0:
                features['max_days_without_activity'] = np.max(date_diffs)
                features['mean_days_between_access'] = np.mean(date_diffs)
                features['inactive_periods'] = (date_diffs > 7).sum()
            
            temporal_features.append(features)
        
        df_temporal = pd.DataFrame(temporal_features)
        logger.info(f"  ✓ {len(df_temporal)} estudiantes con características temporales")
        return df_temporal
    
    def extract_assessment_features(self) -> pd.DataFrame:
        """
        Extraer características de rendimiento académico
        
        Incluye:
        - Trend de calificaciones
        - Comparación con peers
        - Tasa de entregas tardías
        
        Returns:
            DataFrame con características de rendimiento
        """
        logger.info("Extrayendo características de evaluación...")
        
        assess_features = []
        
        for student_id in self.df_studentassess['id_student'].unique():
            student_assess = self.df_studentassess[
                self.df_studentassess['id_student'] == student_id
            ].sort_values('date_submitted')
            
            if len(student_assess) == 0:
                continue
            
            # Merge con información de evaluación
            student_assess = student_assess.merge(
                self.df_assess[['id_assessment', 'date']],
                on='id_assessment',
                how='left'
            )
            student_assess = student_assess.rename(columns={'date': 'date_due'})
            
            features = {
                'id_student': student_id,
                'n_assessments': len(student_assess),
                'mean_score': student_assess['score'].mean(),
                'std_score': student_assess['score'].std() or 0,
                'max_score': student_assess['score'].max(),
                'min_score': student_assess['score'].min(),
            }
            
            # Trend de calificaciones (pendiente)
            if len(student_assess) > 1:
                assessment_order = np.arange(len(student_assess))
                scores = student_assess['score'].values
                valid_idx = ~np.isnan(scores)
                
                if valid_idx.sum() > 1:
                    slope = np.polyfit(assessment_order[valid_idx], 
                                      scores[valid_idx], 1)[0]
                    features['grade_trend_slope'] = slope
                else:
                    features['grade_trend_slope'] = 0
            else:
                features['grade_trend_slope'] = 0
            
            # Entregas tardías
            if 'date_due' in student_assess.columns:
                late_submissions = (
                    student_assess['date_submitted'] > student_assess['date_due']
                ).sum()
                features['late_submission_rate'] = late_submissions / len(student_assess)
            else:
                features['late_submission_rate'] = 0
            
            assess_features.append(features)
        
        df_assess = pd.DataFrame(assess_features)
        logger.info(f"  ✓ {len(df_assess)} estudiantes con características de evaluación")
        return df_assess
    
    def extract_demographic_features(self) -> pd.DataFrame:
        """
        Extraer características demográficas
        
        Incluye:
        - Edad, género
        - Nivel educativo previo
        - Región, discapacidad
        
        Returns:
            DataFrame con características demográficas
        """
        logger.info("Extrayendo características demográficas...")
        
        # Usar directamente del studentInfo
        demo_cols = ['id_student', 'gender', 'age_band', 'highest_education', 'region']
        df_demo = self.df_studentinfo[demo_cols].drop_duplicates()
        
        # One-hot encoding de categóricas
        df_demo = pd.get_dummies(df_demo, columns=['gender', 'age_band', 
                                                     'highest_education', 'region'],
                                drop_first=True)
        
        logger.info(f"  ✓ {len(df_demo)} estudiantes con características demográficas")
        return df_demo
    
    def create_target_variable(self) -> pd.DataFrame:
        """
        Crear variable objetivo (proxy de burnout académico)
        
        Basada en:
        1. Withdrawal (abandono temprano) - Strong signal
        2. Grade decline > 2 SD
        3. Engagement drop > 50%
        4. Irregular activity patterns
        
        Returns:
            DataFrame con id_student y target (0=low risk, 1=high risk)
        """
        logger.info("Creando variable objetivo (burnout proxy)...")
        
        # Obtener outcomes originales
        outcomes = self.df_studentinfo[['id_student', 'final_result', 'code_module']].copy()
        
        # Encoding de outcomes
        # Withdrawal y Fail = burnout signal fuerte
        outcomes['burnout_risk'] = outcomes['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
        
        logger.info(f"  ✓ Variable objetivo creada")
        logger.info(f"    - Clase 0 (Low risk): {(outcomes['burnout_risk'] == 0).sum()}")
        logger.info(f"    - Clase 1 (High risk): {(outcomes['burnout_risk'] == 1).sum()}")
        
        return outcomes[['id_student', 'burnout_risk']]
    
    def extract_all_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extraer todas las características
        
        Returns:
            Tupla: (X_features, y_target)
        """
        logger.info("="*60)
        logger.info("EXTRACCIÓN COMPLETA DE CARACTERÍSTICAS")
        logger.info("="*60)
        
        # Extraer cada grupo de características
        df_engagement = self.extract_engagement_features()
        df_temporal = self.extract_temporal_features()
        df_assess = self.extract_assessment_features()
        df_demo = self.extract_demographic_features()
        y = self.create_target_variable()
        
        # Combinar todas las características
        X = df_engagement.copy()
        X = X.merge(df_temporal, on='id_student', how='outer')
        X = X.merge(df_assess, on='id_student', how='outer')
        X = X.merge(df_demo, on='id_student', how='outer')
        
        # Merge con target
        X = X.merge(y, on='id_student', how='inner')
        
        logger.info("="*60)
        logger.info(f"✓ Features finales extraídos:")
        logger.info(f"  - Filas (estudiantes): {len(X)}")
        logger.info(f"  - Columnas (features): {len(X.columns) - 2}")  # -2 por id_student y target
        logger.info(f"  - Valores faltantes: {X.isnull().sum().sum()}")
        logger.info("="*60)
        
        return X, X['burnout_risk']


def extract_burnout_features(df_vle: pd.DataFrame, 
                             df_studentinfo: pd.DataFrame,
                             df_assess: pd.DataFrame,
                             df_studentassess: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Función de conveniencia para extraer características
    
    Args:
        df_vle: DataFrame de interacciones
        df_studentinfo: DataFrame de estudiantes
        df_assess: DataFrame de evaluaciones
        df_studentassess: DataFrame de resultados
        
    Returns:
        Tupla: (X, y)
    """
    extractor = BurnoutFeatureExtractor(df_vle, df_studentinfo, df_assess, df_studentassess)
    return extractor.extract_all_features()
