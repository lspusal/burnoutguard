"""
Módulo de preprocesamiento de datos OULAD

Funciones principales:
- load_oulad_data: Cargar datos crudos
- clean_data: Limpieza y validación
- normalize_features: Normalización de características
- handle_missing_values: Manejo de valores faltantes

Autor: Research Project
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OULADProcessor:
    """Procesador de datos OULAD"""
    
    def __init__(self, data_dir: str = 'data/'):
        """
        Inicializar procesador
        
        Args:
            data_dir: Directorio donde están los datos OULAD
        """
        self.data_dir = Path(data_dir)
        self.df_vle = None
        self.df_studentinfo = None
        self.df_studentassess = None
        self.df_assess = None
        self.df_courses = None
        self.df_vle_static = None
    
    def load_data(self) -> bool:
        """
        Cargar todos los ficheros OULAD CSV
        
        Returns:
            True si carga exitosa
        """
        try:
            logger.info("Cargando datos OULAD...")
            
            self.df_vle = pd.read_csv(self.data_dir / 'studentVLE.csv')
            logger.info(f"  ✓ studentVLE: {self.df_vle.shape}")
            
            self.df_studentinfo = pd.read_csv(self.data_dir / 'studentInfo.csv')
            logger.info(f"  ✓ studentInfo: {self.df_studentinfo.shape}")
            
            self.df_studentassess = pd.read_csv(self.data_dir / 'studentAssessment.csv')
            logger.info(f"  ✓ studentAssessment: {self.df_studentassess.shape}")
            
            self.df_assess = pd.read_csv(self.data_dir / 'assessments.csv')
            logger.info(f"  ✓ assessments: {self.df_assess.shape}")
            
            self.df_courses = pd.read_csv(self.data_dir / 'courses.csv')
            logger.info(f"  ✓ courses: {self.df_courses.shape}")
            
            self.df_vle_static = pd.read_csv(self.data_dir / 'vle.csv')
            logger.info(f"  ✓ vle: {self.df_vle_static.shape}")
            
            logger.info("✓ Todos los datos cargados exitosamente")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"✗ Fichero no encontrado: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error cargando datos: {e}")
            return False
    
    def clean_data(self) -> pd.DataFrame:
        """
        Limpieza general de datos
        
        Returns:
            DataFrame limpio
        """
        logger.info("Limpiando datos...")
        
        # Eliminación de duplicados
        df = self.df_vle.drop_duplicates()
        logger.info(f"  ✓ Duplicados eliminados: {df.shape[0]} filas")
        
        # Manejo de valores faltantes
        df = self._handle_missing_values(df)
        
        # Validación de tipos
        df = self._validate_dtypes(df)
        
        logger.info("✓ Limpieza completada")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manejar valores faltantes"""
        missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        
        if len(missing_percent[missing_percent > 0]) > 0:
            logger.info("  Valores faltantes encontrados:")
            for col, pct in missing_percent[missing_percent > 0].items():
                logger.info(f"    - {col}: {pct:.2f}%")
            
            # Estrategia: eliminar filas con >20% de valores faltantes
            threshold = 0.2
            df = df.dropna(thresh=len(df.columns) * (1 - threshold))
            
            # Imputar numéricas con mediana
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Imputar categóricas con moda
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        return df
    
    def _validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validar y convertir tipos de datos"""
        dtype_mapping = {
            'code_module': 'category',
            'code_presentation': 'category',
            'id_student': 'int32',
            'date': 'int32',
            'sum_click': 'int32',
            'final_result': 'category'
        }
        
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except ValueError:
                    logger.warning(f"  ! No se puede convertir {col} a {dtype}")
        
        return df
    
    def normalize_features(self, X: pd.DataFrame, 
                          method: str = 'standard') -> Tuple[pd.DataFrame, object]:
        """
        Normalizar características
        
        Args:
            X: DataFrame de características
            method: 'standard' (media=0, std=1) o 'minmax' (0-1)
            
        Returns:
            DataFrame normalizado y objeto scaler
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método {method} no reconocido")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        logger.info(f"✓ Características normalizadas ({method})")
        return X_scaled, scaler
    
    def create_student_course_pairs(self) -> pd.DataFrame:
        """
        Crear matriz estudiante-curso para análisis
        
        Returns:
            DataFrame con pares estudiante-curso
        """
        logger.info("Creando pares estudiante-curso...")
        
        pairs = self.df_studentinfo[['id_student', 'code_module', 'code_presentation']].copy()
        pairs = pairs.drop_duplicates()
        
        logger.info(f"  ✓ {len(pairs)} pares estudiante-curso únicos")
        return pairs
    
    def get_summary_statistics(self) -> Dict:
        """
        Obtener estadísticas resumen
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'n_students': self.df_studentinfo['id_student'].nunique(),
            'n_courses': self.df_studentinfo['code_module'].nunique(),
            'n_interactions': len(self.df_vle),
            'date_range': (self.df_vle['date'].min(), self.df_vle['date'].max()),
            'outcome_distribution': self.df_studentinfo['final_result'].value_counts().to_dict()
        }
        return stats


# Funciones de alto nivel
def load_and_preprocess_oulad(data_dir: str = 'data/', 
                              normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Función de conveniencia para cargar y preprocesar OULAD
    
    Args:
        data_dir: Directorio de datos
        normalize: Si True, normaliza características numéricas
        
    Returns:
        Tupla: (X_features, y_target, scaler)
    """
    processor = OULADProcessor(data_dir)
    
    if not processor.load_data():
        raise RuntimeError("Error cargando datos OULAD")
    
    # Obtener estadísticas
    stats = processor.get_summary_statistics()
    logger.info("Estadísticas resumen:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Limpiar datos
    df_clean = processor.clean_data()
    
    # Normalizar si se solicita
    scaler = None
    if normalize:
        df_clean, scaler = processor.normalize_features(df_clean)
    
    return processor, df_clean, scaler
