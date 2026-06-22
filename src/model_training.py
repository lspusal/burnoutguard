"""
Módulo de entrenamiento de modelos para predicción de burnout académico

Incluye:
- Modelos baseline (Logistic Regression, Decision Tree, Random Forest)
- Modelos avanzados (XGBoost, LightGBM, CatBoost)
- Ensemble de modelos
- Tuning de hiperparámetros con Optuna
- Validación temporal y cross-validation

Autor: Research Project
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Sklearn
from sklearn.model_selection import (
    StratifiedKFold, 
    train_test_split,
    cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

# Gradient Boosting
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# Hyperparameter tuning
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BurnoutModelTrainer:
    """Clase para entrenar modelos de predicción de burnout"""
    
    def __init__(self, 
                 random_state: int = 42,
                 output_dir: str = 'results/models/'):
        """
        Inicializar trainer
        
        Args:
            random_state: Semilla para reproducibilidad
            output_dir: Directorio para guardar modelos
        """
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
        
        # Calcular class weight para desbalance
        self.class_weight = 'balanced'
        
    def _get_baseline_models(self) -> Dict[str, Any]:
        """
        Obtener modelos baseline
        
        Returns:
            Diccionario de modelos baseline
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                max_iter=1000,
                solver='lbfgs'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight=self.class_weight,
                max_depth=10
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=-1
            )
        }
        return models
    
    def _get_advanced_models(self, scale_pos_weight: float = 3.2) -> Dict[str, Any]:
        """
        Obtener modelos avanzados de gradient boosting
        
        Args:
            scale_pos_weight: Peso para clase positiva (desbalance)
            
        Returns:
            Diccionario de modelos avanzados
        """
        models = {}
        
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='auc',
                n_jobs=-1
            )
        else:
            logger.warning("XGBoost no disponible, instalando con: pip install xgboost")
        
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            logger.warning("LightGBM no disponible, instalando con: pip install lightgbm")
        
        if HAS_CATBOOST:
            models['catboost'] = CatBoostClassifier(
                iterations=300,
                depth=7,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                verbose=0
            )
        else:
            logger.warning("CatBoost no disponible, instalando con: pip install catboost")
        
        return models
    
    def temporal_split(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       date_column: str = 'date',
                       train_weeks: int = 20,
                       val_weeks: int = 6,
                       test_weeks: int = 4) -> Tuple:
        """
        Split temporal para simular despliegue real
        
        Args:
            X: Features
            y: Target
            date_column: Columna de fecha
            train_weeks: Semanas de training
            val_weeks: Semanas de validación
            test_weeks: Semanas de test
            
        Returns:
            Tupla con splits
        """
        if date_column in X.columns:
            # Convertir días a semanas (OULAD usa días desde inicio)
            X_copy = X.copy()
            X_copy['_week'] = X[date_column] // 7
            
            train_mask = X_copy['_week'] < train_weeks
            val_mask = (X_copy['_week'] >= train_weeks) & (X_copy['_week'] < train_weeks + val_weeks)
            test_mask = X_copy['_week'] >= train_weeks + val_weeks
            
            X_train = X[train_mask].drop(columns=[date_column], errors='ignore')
            X_val = X[val_mask].drop(columns=[date_column], errors='ignore')
            X_test = X[test_mask].drop(columns=[date_column], errors='ignore')
            
            y_train = y[train_mask]
            y_val = y[val_mask]
            y_test = y[test_mask]
            
            logger.info(f"Split temporal: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
        else:
            # Fallback a split aleatorio estratificado
            logger.warning(f"Columna '{date_column}' no encontrada, usando split aleatorio")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=self.random_state
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.18, stratify=y_temp, random_state=self.random_state
            )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_smote(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    sampling_strategy: float = 0.7) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplicar SMOTE para balancear clases
        
        Args:
            X_train: Features de training
            y_train: Target de training
            sampling_strategy: Ratio de oversampling
            
        Returns:
            Datos balanceados
        """
        if not HAS_IMBLEARN:
            logger.warning("imbalanced-learn no disponible, retornando datos originales")
            return X_train, y_train
        
        logger.info(f"Aplicando SMOTE (strategy={sampling_strategy})...")
        logger.info(f"  Antes: {y_train.value_counts().to_dict()}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"  Después: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
    
    def train_baseline_models(self,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_val: Optional[pd.DataFrame] = None,
                              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Entrenar modelos baseline
        
        Args:
            X_train: Features de training
            y_train: Target de training
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            
        Returns:
            Diccionario de modelos entrenados
        """
        logger.info("="*60)
        logger.info("ENTRENANDO MODELOS BASELINE")
        logger.info("="*60)
        
        baseline_models = self._get_baseline_models()
        
        for name, model in baseline_models.items():
            logger.info(f"\nEntrenando {name}...")
            
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Evaluar en train
            train_proba = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, train_proba)
            
            # Evaluar en val si disponible
            if X_val is not None and y_val is not None:
                val_proba = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_proba)
                logger.info(f"  ✓ {name}: Train AUROC={train_auc:.4f}, Val AUROC={val_auc:.4f}")
            else:
                logger.info(f"  ✓ {name}: Train AUROC={train_auc:.4f}")
        
        return self.trained_models
    
    def train_advanced_models(self,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_val: Optional[pd.DataFrame] = None,
                              y_val: Optional[pd.Series] = None,
                              early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Entrenar modelos avanzados con early stopping
        
        Args:
            X_train: Features de training
            y_train: Target de training
            X_val: Features de validación
            y_val: Target de validación
            early_stopping_rounds: Rounds para early stopping
            
        Returns:
            Diccionario de modelos entrenados
        """
        logger.info("="*60)
        logger.info("ENTRENANDO MODELOS AVANZADOS")
        logger.info("="*60)
        
        # Calcular scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        advanced_models = self._get_advanced_models(scale_pos_weight)
        
        for name, model in advanced_models.items():
            logger.info(f"\nEntrenando {name}...")
            
            if X_val is not None and y_val is not None:
                # Entrenar con early stopping
                if name == 'xgboost' and HAS_XGBOOST:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif name == 'lightgbm' and HAS_LIGHTGBM:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                    )
                elif name == 'catboost' and HAS_CATBOOST:
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            self.trained_models[name] = model
            
            # Evaluar
            train_proba = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, train_proba)
            
            if X_val is not None and y_val is not None:
                val_proba = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_proba)
                logger.info(f"  ✓ {name}: Train AUROC={train_auc:.4f}, Val AUROC={val_auc:.4f}")
            else:
                logger.info(f"  ✓ {name}: Train AUROC={train_auc:.4f}")
        
        return self.trained_models
    
    def create_ensemble(self,
                        model_names: List[str] = ['xgboost', 'lightgbm', 'catboost'],
                        weights: Optional[List[float]] = None) -> Any:
        """
        Crear ensemble de modelos
        
        Args:
            model_names: Nombres de modelos a incluir
            weights: Pesos para cada modelo
            
        Returns:
            Ensemble de modelos
        """
        logger.info("="*60)
        logger.info("CREANDO ENSEMBLE")
        logger.info("="*60)
        
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                estimators.append((name, self.trained_models[name]))
            else:
                logger.warning(f"Modelo {name} no encontrado, saltando...")
        
        if len(estimators) == 0:
            raise ValueError("No hay modelos disponibles para ensemble")
        
        if weights is None:
            weights = [1.0] * len(estimators)
        
        # Crear ensemble simple que promedia probabilidades
        class SimpleEnsemble:
            def __init__(self, models, model_weights):
                self.models = models
                self.weights = model_weights
                self.classes_ = np.array([0, 1])
            
            def predict_proba(self, X):
                probas = np.zeros((len(X), 2))
                total_weight = sum(self.weights)
                for (name, model), weight in zip(self.models, self.weights):
                    probas += model.predict_proba(X) * weight
                return probas / total_weight
            
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        
        ensemble = SimpleEnsemble(estimators, weights[:len(estimators)])
        
        self.trained_models['ensemble'] = ensemble
        logger.info(f"  ✓ Ensemble creado con modelos: {[name for name, _ in estimators]}")
        
        return ensemble
    
    def tune_xgboost_optuna(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series,
                            n_trials: int = 50) -> Dict:
        """
        Tuning de hiperparámetros con Optuna
        
        Args:
            X_train, y_train: Datos de training
            X_val, y_val: Datos de validación
            n_trials: Número de trials
            
        Returns:
            Mejores parámetros
        """
        if not HAS_OPTUNA:
            logger.warning("Optuna no disponible, usando parámetros por defecto")
            return {}
        
        if not HAS_XGBOOST:
            logger.warning("XGBoost no disponible para tuning")
            return {}
        
        logger.info(f"Tuning XGBoost con Optuna ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'auc'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            val_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, val_proba)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['xgboost'] = study.best_params
        logger.info(f"  ✓ Best AUROC: {study.best_value:.4f}")
        logger.info(f"  ✓ Best params: {study.best_params}")
        
        return study.best_params
    
    def cross_validate(self,
                       model: Any,
                       X: pd.DataFrame,
                       y: pd.Series,
                       n_folds: int = 5) -> Dict:
        """
        Cross-validation estratificada
        
        Args:
            model: Modelo a evaluar
            X: Features
            y: Target
            n_folds: Número de folds
            
        Returns:
            Diccionario con métricas
        """
        logger.info(f"Cross-validation con {n_folds} folds...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        
        results = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'scores': scores
        }
        
        logger.info(f"  ✓ AUROC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        
        return results
    
    def calibrate_model(self,
                        model_name: str,
                        X_cal: pd.DataFrame,
                        y_cal: pd.Series,
                        method: str = 'isotonic') -> Any:
        """
        Calibrar probabilidades del modelo
        
        Args:
            model_name: Nombre del modelo a calibrar
            X_cal: Datos de calibración
            y_cal: Target de calibración
            method: 'isotonic' o 'sigmoid'
            
        Returns:
            Modelo calibrado
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        logger.info(f"Calibrando {model_name} ({method})...")
        
        calibrated = CalibratedClassifierCV(
            self.trained_models[model_name],
            method=method,
            cv='prefit'
        )
        calibrated.fit(X_cal, y_cal)
        
        self.trained_models[f'{model_name}_calibrated'] = calibrated
        logger.info(f"  ✓ {model_name} calibrado")
        
        return calibrated
    
    def save_models(self, filename_prefix: str = 'burnout_model') -> List[str]:
        """
        Guardar modelos entrenados
        
        Args:
            filename_prefix: Prefijo para nombres de archivo
            
        Returns:
            Lista de paths guardados
        """
        saved_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.trained_models.items():
            filepath = self.output_dir / f'{filename_prefix}_{name}_{timestamp}.pkl'
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            saved_paths.append(str(filepath))
            logger.info(f"  ✓ Guardado: {filepath}")
        
        return saved_paths
    
    def load_model(self, filepath: str) -> Any:
        """
        Cargar modelo desde archivo
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Modelo cargado
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"  ✓ Modelo cargado desde: {filepath}")
        return model


def train_burnout_models(X: pd.DataFrame,
                         y: pd.Series,
                         use_smote: bool = True,
                         tune_hyperparams: bool = False) -> Tuple[Dict, BurnoutModelTrainer]:
    """
    Función de conveniencia para entrenar todos los modelos
    
    Args:
        X: Features
        y: Target
        use_smote: Aplicar SMOTE
        tune_hyperparams: Tuning con Optuna
        
    Returns:
        Diccionario de modelos y trainer
    """
    trainer = BurnoutModelTrainer()
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.temporal_split(X, y)
    
    # SMOTE
    if use_smote:
        X_train, y_train = trainer.apply_smote(X_train, y_train)
    
    # Entrenar baseline
    trainer.train_baseline_models(X_train, y_train, X_val, y_val)
    
    # Entrenar avanzados
    trainer.train_advanced_models(X_train, y_train, X_val, y_val)
    
    # Tuning opcional
    if tune_hyperparams and HAS_OPTUNA:
        trainer.tune_xgboost_optuna(X_train, y_train, X_val, y_val)
    
    # Crear ensemble
    trainer.create_ensemble()
    
    return trainer.trained_models, trainer
