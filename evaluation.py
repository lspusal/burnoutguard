"""
Módulo de evaluación de modelos para predicción de burnout académico

Incluye:
- Métricas de clasificación (AUROC, AUPRC, F1, Precision, Recall)
- Curvas ROC y Precision-Recall
- Matrices de confusión
- Curvas de calibración
- Análisis de umbrales

Autor: Research Project
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Sklearn metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Estilo de plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ModelEvaluator:
    """Clase para evaluar modelos de predicción de burnout"""
    
    def __init__(self, output_dir: str = 'results/metrics/'):
        """
        Inicializar evaluador
        
        Args:
            output_dir: Directorio para guardar figuras y métricas
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def compute_metrics(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_proba: np.ndarray,
                        threshold: float = 0.5) -> Dict[str, float]:
        """
        Calcular todas las métricas de evaluación
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones binarias
            y_proba: Probabilidades predichas
            threshold: Umbral para clasificación
            
        Returns:
            Diccionario con métricas
        """
        # Aplicar umbral si se proporcionan probabilidades
        if y_pred is None:
            y_pred = (y_proba >= threshold).astype(int)
        
        # Métricas básicas
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'f2_score': self._f_beta_score(y_true, y_pred, beta=2),
            'specificity': self._specificity(y_true, y_pred),
        }
        
        # Métricas basadas en probabilidad
        if y_proba is not None:
            metrics['auroc'] = roc_auc_score(y_true, y_proba)
            metrics['auprc'] = average_precision_score(y_true, y_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        
        # Calcular intervalos de confianza con bootstrap
        ci = self._bootstrap_ci(y_true, y_proba)
        metrics['auroc_ci_lower'] = ci['auroc_lower']
        metrics['auroc_ci_upper'] = ci['auroc_upper']
        
        return metrics
    
    def _f_beta_score(self, y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2) -> float:
        """Calcular F-beta score"""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if precision + recall == 0:
            return 0.0
        
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    def _specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular especificidad (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _bootstrap_ci(self,
                      y_true: np.ndarray,
                      y_proba: np.ndarray,
                      n_bootstrap: int = 1000,
                      ci: float = 0.95) -> Dict[str, float]:
        """
        Calcular intervalos de confianza con bootstrap
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            n_bootstrap: Número de muestras bootstrap
            ci: Nivel de confianza
            
        Returns:
            Diccionario con límites del IC
        """
        np.random.seed(42)
        n_samples = len(y_true)
        
        auroc_scores = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_proba_boot = y_proba[indices]
            
            # Check if both classes are present
            if len(np.unique(y_true_boot)) < 2:
                continue
            
            try:
                auroc_scores.append(roc_auc_score(y_true_boot, y_proba_boot))
            except ValueError:
                continue
        
        if len(auroc_scores) == 0:
            return {'auroc_lower': 0.0, 'auroc_upper': 1.0}
        
        alpha = (1 - ci) / 2
        lower = np.percentile(auroc_scores, alpha * 100)
        upper = np.percentile(auroc_scores, (1 - alpha) * 100)
        
        return {'auroc_lower': lower, 'auroc_upper': upper}
    
    def evaluate_model(self,
                       model: Any,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       model_name: str = 'model') -> Dict[str, float]:
        """
        Evaluar un modelo completo
        
        Args:
            model: Modelo entrenado
            X_test: Features de test
            y_test: Target de test
            model_name: Nombre del modelo
            
        Returns:
            Diccionario con métricas
        """
        logger.info(f"Evaluando modelo: {model_name}")
        
        # Predicciones
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Métricas
        metrics = self.compute_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_proba=y_proba
        )
        
        self.results[model_name] = metrics
        
        # Log resultados
        logger.info(f"  AUROC: {metrics['auroc']:.4f} [{metrics['auroc_ci_lower']:.3f}-{metrics['auroc_ci_upper']:.3f}]")
        logger.info(f"  AUPRC: {metrics['auprc']:.4f}")
        logger.info(f"  F1: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        
        return metrics
    
    def evaluate_multiple_models(self,
                                 models: Dict[str, Any],
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluar múltiples modelos
        
        Args:
            models: Diccionario de modelos
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            DataFrame con resultados
        """
        logger.info("="*60)
        logger.info("EVALUACIÓN DE MODELOS")
        logger.info("="*60)
        
        for name, model in models.items():
            self.evaluate_model(model, X_test, y_test, name)
        
        return self.get_results_dataframe()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Obtener resultados como DataFrame
        
        Returns:
            DataFrame con métricas por modelo
        """
        return pd.DataFrame(self.results).T.round(4)
    
    def plot_roc_curves(self,
                        models: Dict[str, Any],
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plotear curvas ROC para múltiples modelos
        
        Args:
            models: Diccionario de modelos
            X_test: Features de test
            y_test: Target de test
            save_path: Ruta para guardar figura
            
        Returns:
            Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for (name, model), color in zip(models.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{name} (AUC = {auc:.3f})')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Burnout Prediction Models', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ ROC curves guardadas: {save_path}")
        
        return fig
    
    def plot_pr_curves(self,
                       models: Dict[str, Any],
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plotear curvas Precision-Recall
        
        Args:
            models: Diccionario de modelos
            X_test: Features de test
            y_test: Target de test
            save_path: Ruta para guardar figura
            
        Returns:
            Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        # Baseline (proporción de clase positiva)
        baseline = y_test.mean()
        
        for (name, model), color in zip(models.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{name} (AP = {ap:.3f})')
        
        # Baseline
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
                  label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - Burnout Prediction Models', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ PR curves guardadas: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str = 'Model',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plotear matriz de confusión
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            model_name: Nombre del modelo
            save_path: Ruta para guardar figura
            
        Returns:
            Figura de matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Confusion matrix guardada: {save_path}")
        
        return fig
    
    def plot_calibration_curve(self,
                               model: Any,
                               X_test: pd.DataFrame,
                               y_test: pd.Series,
                               model_name: str = 'Model',
                               n_bins: int = 10,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plotear curva de calibración
        
        Args:
            model: Modelo entrenado
            X_test: Features de test
            y_test: Target de test
            model_name: Nombre del modelo
            n_bins: Número de bins
            save_path: Ruta para guardar
            
        Returns:
            Figura de matplotlib
        """
        y_proba = model.predict_proba(X_test)[:, 1]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba, n_bins=n_bins
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        ax1.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=model_name, markersize=8)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.hist(y_proba, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predictions', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Calibration curve guardada: {save_path}")
        
        return fig
    
    def find_optimal_threshold(self,
                               y_true: np.ndarray,
                               y_proba: np.ndarray,
                               metric: str = 'f1') -> Tuple[float, float]:
        """
        Encontrar umbral óptimo para una métrica
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            metric: 'f1', 'recall', 'precision', 'youden'
            
        Returns:
            Umbral óptimo y valor de la métrica
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic = sensitivity + specificity - 1
                sensitivity = recall_score(y_true, y_pred, zero_division=0)
                specificity = self._specificity(y_true, y_pred)
                score = sensitivity + specificity - 1
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        logger.info(f"  Umbral óptimo ({metric}): {best_threshold:.2f} -> {metric}={best_score:.4f}")
        
        return best_threshold, best_score
    
    def generate_report(self,
                        models: Dict[str, Any],
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        save_dir: Optional[str] = None) -> str:
        """
        Generar reporte completo de evaluación
        
        Args:
            models: Diccionario de modelos
            X_test: Features de test
            y_test: Target de test
            save_dir: Directorio para guardar
            
        Returns:
            Reporte en texto
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.output_dir
        
        # Evaluar modelos
        results_df = self.evaluate_multiple_models(models, X_test, y_test)
        
        # Guardar CSV
        results_df.to_csv(save_path / 'model_metrics.csv')
        
        # Generar figuras
        self.plot_roc_curves(models, X_test, y_test, 
                            save_path / 'roc_curves.png')
        self.plot_pr_curves(models, X_test, y_test,
                           save_path / 'pr_curves.png')
        
        # Mejor modelo
        best_model_name = results_df['auroc'].idxmax()
        best_model = models[best_model_name]
        
        y_pred = best_model.predict(X_test)
        self.plot_confusion_matrix(y_test, y_pred, best_model_name,
                                  save_path / 'confusion_matrix.png')
        self.plot_calibration_curve(best_model, X_test, y_test, best_model_name,
                                   save_path / 'calibration_curve.png')
        
        # Generar texto de reporte
        report = f"""
================================================================================
REPORTE DE EVALUACIÓN DE MODELOS - PREDICCIÓN DE BURNOUT ACADÉMICO
================================================================================

Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Muestras de test: {len(y_test)}
Distribución de clases: {y_test.value_counts().to_dict()}

RESULTADOS POR MODELO
---------------------
{results_df.to_string()}

MEJOR MODELO: {best_model_name}
  - AUROC: {results_df.loc[best_model_name, 'auroc']:.4f}
  - AUPRC: {results_df.loc[best_model_name, 'auprc']:.4f}
  - F1 Score: {results_df.loc[best_model_name, 'f1_score']:.4f}

ARCHIVOS GENERADOS
------------------
  - model_metrics.csv
  - roc_curves.png
  - pr_curves.png
  - confusion_matrix.png
  - calibration_curve.png

================================================================================
"""
        
        # Guardar reporte
        with open(save_path / 'evaluation_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"  ✓ Reporte guardado en: {save_path}")
        
        return report


def evaluate_burnout_models(models: Dict[str, Any],
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            output_dir: str = 'results/metrics/') -> pd.DataFrame:
    """
    Función de conveniencia para evaluar todos los modelos
    
    Args:
        models: Diccionario de modelos
        X_test: Features de test
        y_test: Target de test
        output_dir: Directorio de salida
        
    Returns:
        DataFrame con resultados
    """
    evaluator = ModelEvaluator(output_dir)
    evaluator.generate_report(models, X_test, y_test)
    return evaluator.get_results_dataframe()
