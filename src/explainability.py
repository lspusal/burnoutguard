"""
Módulo de interpretabilidad para predicción de burnout académico

Incluye:
- Análisis SHAP (SHapley Additive exPlanations)
- Feature importance global y local
- Dependence plots
- Force plots y waterfall plots
- Validación de interpretabilidad

Autor: Research Project
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Estilo de plots
plt.style.use('seaborn-v0_8-whitegrid')


class SHAPAnalyzer:
    """Clase para análisis de interpretabilidad con SHAP"""
    
    def __init__(self, output_dir: str = 'results/figures/'):
        """
        Inicializar analizador SHAP
        
        Args:
            output_dir: Directorio para guardar figuras
        """
        if not HAS_SHAP:
            raise ImportError("SHAP no está instalado. Instalar con: pip install shap")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.X_sample = None
    
    def compute_shap_values(self,
                            model: Any,
                            X: pd.DataFrame,
                            sample_size: Optional[int] = 1000,
                            model_type: str = 'tree') -> np.ndarray:
        """
        Calcular valores SHAP para un modelo
        
        Args:
            model: Modelo entrenado
            X: DataFrame de features
            sample_size: Número de muestras a usar (None = todas)
            model_type: 'tree' para modelos de árbol, 'kernel' para otros
            
        Returns:
            Array de valores SHAP
        """
        logger.info("Calculando valores SHAP...")
        
        # Subsamplear si es necesario
        if sample_size and len(X) > sample_size:
            self.X_sample = X.sample(n=sample_size, random_state=42)
            logger.info(f"  Usando {sample_size} muestras de {len(X)}")
        else:
            self.X_sample = X
        
        self.feature_names = list(X.columns)
        
        # Crear explainer según tipo de modelo
        if model_type == 'tree':
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"TreeExplainer falló ({e}), usando Kernel")
                model_type = 'kernel'
        
        if model_type == 'kernel':
            # Para modelos no basados en árboles
            background = shap.sample(X, 100)
            self.explainer = shap.KernelExplainer(
                model.predict_proba, 
                background
            )
        
        # Calcular valores SHAP
        try:
            self.shap_values = self.explainer.shap_values(self.X_sample)
            
            # Para clasificación binaria, tomar la clase positiva
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]
            
            logger.info(f"  ✓ SHAP values calculados: shape={self.shap_values.shape}")
            
        except Exception as e:
            logger.error(f"Error calculando SHAP values: {e}")
            raise
        
        return self.shap_values
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Obtener importancia de features según SHAP
        
        Args:
            top_n: Número de features a retornar
            
        Returns:
            DataFrame con importancia de features
        """
        if self.shap_values is None:
            raise ValueError("Primero calcular SHAP values con compute_shap_values()")
        
        # Importancia media absoluta
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'importance_pct': mean_abs_shap / mean_abs_shap.sum() * 100
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info(f"\nTop {top_n} features por importancia SHAP:")
        for _, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_pct']:.1f}%")
        
        return importance_df.head(top_n)
    
    def summary_plot(self,
                     max_display: int = 20,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Generar summary plot (beeswarm)
        
        Args:
            max_display: Número máximo de features a mostrar
            save_path: Ruta para guardar figura
            
        Returns:
            Figura de matplotlib
        """
        if self.shap_values is None:
            raise ValueError("Primero calcular SHAP values con compute_shap_values()")
        
        logger.info("Generando summary plot...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Summary Plot - Feature Importance', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Summary plot guardado: {save_path}")
        
        return plt.gcf()
    
    def bar_plot(self,
                 max_display: int = 20,
                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Generar bar plot de importancia de features
        
        Args:
            max_display: Número máximo de features
            save_path: Ruta para guardar figura
            
        Returns:
            Figura de matplotlib
        """
        if self.shap_values is None:
            raise ValueError("Primero calcular SHAP values con compute_shap_values()")
        
        logger.info("Generando bar plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Feature Importance - Mean |SHAP Value|', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Bar plot guardado: {save_path}")
        
        return plt.gcf()
    
    def dependence_plot(self,
                        feature: str,
                        interaction_feature: Optional[str] = 'auto',
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Generar dependence plot para una feature
        
        Args:
            feature: Nombre de la feature principal
            interaction_feature: Feature de interacción ('auto' para automático)
            save_path: Ruta para guardar figura
            
        Returns:
            Figura de matplotlib
        """
        if self.shap_values is None:
            raise ValueError("Primero calcular SHAP values con compute_shap_values()")
        
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' no encontrada")
        
        logger.info(f"Generando dependence plot para '{feature}'...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.dependence_plot(
            feature,
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
            ax=ax
        )
        
        plt.title(f'SHAP Dependence Plot - {feature}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Dependence plot guardado: {save_path}")
        
        return fig
    
    def force_plot(self,
                   sample_idx: int = 0,
                   save_path: Optional[str] = None) -> Any:
        """
        Generar force plot para una predicción individual
        
        Args:
            sample_idx: Índice de la muestra
            save_path: Ruta para guardar (HTML)
            
        Returns:
            Objeto force plot de SHAP
        """
        if self.shap_values is None:
            raise ValueError("Primero calcular SHAP values con compute_shap_values()")
        
        logger.info(f"Generando force plot para muestra {sample_idx}...")
        
        # Base value
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1]  # Clase positiva
        else:
            expected_value = 0
        
        force = shap.force_plot(
            expected_value,
            self.shap_values[sample_idx, :],
            self.X_sample.iloc[sample_idx, :],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            if save_path.endswith('.html'):
                shap.save_html(save_path, force)
            else:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Force plot guardado: {save_path}")
        
        return force
    
    def waterfall_plot(self,
                       sample_idx: int = 0,
                       max_display: int = 15,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Generar waterfall plot para una predicción individual
        
        Args:
            sample_idx: Índice de la muestra
            max_display: Número máximo de features
            save_path: Ruta para guardar
            
        Returns:
            Figura de matplotlib
        """
        if self.shap_values is None:
            raise ValueError("Primero calcular SHAP values con compute_shap_values()")
        
        logger.info(f"Generando waterfall plot para muestra {sample_idx}...")
        
        # Crear Explanation object para SHAP >= 0.40
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]
        else:
            base_value = 0
        
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=base_value,
            data=self.X_sample.iloc[sample_idx].values,
            feature_names=self.feature_names
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Waterfall plot guardado: {save_path}")
        
        return plt.gcf()
    
    def plot_top_features_dependence(self,
                                     top_n: int = 6,
                                     save_dir: Optional[str] = None) -> List[plt.Figure]:
        """
        Generar dependence plots para las top N features
        
        Args:
            top_n: Número de features
            save_dir: Directorio para guardar
            
        Returns:
            Lista de figuras
        """
        importance_df = self.get_feature_importance(top_n)
        top_features = importance_df['feature'].tolist()
        
        figures = []
        
        for feature in top_features:
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f'dependence_{feature}.png'
            
            fig = self.dependence_plot(feature, save_path=save_path)
            figures.append(fig)
            plt.close(fig)
        
        return figures
    
    def generate_report(self,
                        model: Any,
                        X: pd.DataFrame,
                        sample_size: int = 1000,
                        save_dir: Optional[str] = None) -> str:
        """
        Generar reporte completo de interpretabilidad
        
        Args:
            model: Modelo entrenado
            X: Features
            sample_size: Tamaño de muestra
            save_dir: Directorio para guardar
            
        Returns:
            Texto del reporte
        """
        if save_dir:
            save_path = Path(save_dir)
        else:
            save_path = self.output_dir
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("GENERANDO REPORTE DE INTERPRETABILIDAD")
        logger.info("="*60)
        
        # Calcular SHAP values
        self.compute_shap_values(model, X, sample_size=sample_size)
        
        # Obtener importancia
        importance_df = self.get_feature_importance(top_n=20)
        importance_df.to_csv(save_path / 'feature_importance.csv', index=False)
        
        # Generar plots
        self.summary_plot(save_path=save_path / 'shap_summary.png')
        plt.close()
        
        self.bar_plot(save_path=save_path / 'shap_bar.png')
        plt.close()
        
        # Top 3 features dependence plots
        top_features = importance_df['feature'].head(3).tolist()
        for feature in top_features:
            self.dependence_plot(
                feature, 
                save_path=save_path / f'shap_dependence_{feature}.png'
            )
            plt.close()
        
        # Waterfall plot para ejemplo
        self.waterfall_plot(sample_idx=0, save_path=save_path / 'shap_waterfall_example.png')
        plt.close()
        
        # Generar texto de reporte
        top_3 = importance_df.head(3)
        report = f"""
================================================================================
REPORTE DE INTERPRETABILIDAD - ANÁLISIS SHAP
================================================================================

Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Muestras analizadas: {len(self.X_sample)}
Features: {len(self.feature_names)}

FEATURE IMPORTANCE (Top 10)
---------------------------
{importance_df.head(10).to_string(index=False)}

INSIGHTS PRINCIPALES
--------------------
1. Feature más importante: {top_3.iloc[0]['feature']} ({top_3.iloc[0]['importance_pct']:.1f}%)
2. Segunda más importante: {top_3.iloc[1]['feature']} ({top_3.iloc[1]['importance_pct']:.1f}%)
3. Tercera más importante: {top_3.iloc[2]['feature']} ({top_3.iloc[2]['importance_pct']:.1f}%)

Las top 3 features explican {top_3['importance_pct'].sum():.1f}% de la importancia total.

INTERPRETACIÓN CLÍNICA
----------------------
- engagement_trend_slope: Declive de participación indica burnout
- max_days_without_activity: Gaps largos = desvinculación
- grade_trend_slope: Deterioro académico progresivo

ARCHIVOS GENERADOS
------------------
  - feature_importance.csv
  - shap_summary.png
  - shap_bar.png
  - shap_dependence_*.png
  - shap_waterfall_example.png

================================================================================
"""
        
        # Guardar reporte
        with open(save_path / 'interpretability_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"  ✓ Reporte guardado en: {save_path}")
        
        return report


def analyze_model_explainability(model: Any,
                                 X: pd.DataFrame,
                                 output_dir: str = 'results/figures/') -> pd.DataFrame:
    """
    Función de conveniencia para análisis de interpretabilidad
    
    Args:
        model: Modelo entrenado
        X: Features
        output_dir: Directorio de salida
        
    Returns:
        DataFrame con importancia de features
    """
    analyzer = SHAPAnalyzer(output_dir)
    analyzer.generate_report(model, X, save_dir=output_dir)
    return analyzer.get_feature_importance()
