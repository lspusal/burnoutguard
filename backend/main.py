"""
BurnoutGuard Backend - FastAPI Server

API para predicción de riesgo de abandono con explicaciones LLM.

Endpoints:
- POST /predict - Predecir riesgo para un estudiante
- POST /batch_predict - Predecir para múltiples estudiantes
- GET /health - Health check

Autor: Research Project
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cortes de nivel de riesgo sobre la probabilidad CALIBRADA. Por defecto
# "medio" coincide con la tasa base de abandono observada (~0.25).
THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.50"))
THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.25"))

app = FastAPI(
    title="BurnoutGuard API",
    description="API para predicción de riesgo de abandono académico con explicaciones IA",
    version="1.0.0"
)

# CORS para Moodle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== MODELOS PYDANTIC ==========

class StudentFeatures(BaseModel):
    """Features del estudiante extraídas de Moodle"""
    student_id: int
    student_name: Optional[str] = "Estudiante"
    course_id: int
    course_name: Optional[str] = "Curso"
    
    # Engagement features
    total_clicks: int = 0
    days_since_last_access: int = 0
    active_days: int = 0
    avg_clicks_per_day: float = 0.0
    
    # Academic features
    avg_grade: float = 0.0
    course_avg_grade: float = 70.0
    assignments_completed: int = 0
    assignments_total: int = 1
    
    # Temporal features
    days_enrolled: int = 30
    week_of_course: int = 4
    

class PredictionRequest(BaseModel):
    """Request para predicción"""
    student: StudentFeatures
    language: str = "es"  # es, en


class RiskFactor(BaseModel):
    """Factor de riesgo identificado"""
    factor: str
    description: str
    value: float
    impact: str  # bajo, medio, alto


class PredictionResponse(BaseModel):
    """Response con predicción y explicación"""
    student_id: int
    course_id: int
    risk_score: float          # probabilidad CALIBRADA si hay calibrador
    risk_level: str            # bajo, medio, alto
    raw_score: float           # salida cruda del modelo, antes de calibrar
    calibrated: bool           # False => risk_score sólo sirve para ordenar
    confidence: float
    explanation: str
    key_factors: List[RiskFactor]
    recommendations: List[str]
    generated_at: str


class BatchPredictionRequest(BaseModel):
    """Request para predicción batch"""
    students: List[StudentFeatures]
    language: str = "es"
    capacity: Optional[float] = None  # p.ej. 0.2 => marcar al 20% de mayor riesgo


# ========== PREDICTOR ML ==========

class Calibrator:
    """Calibrador post-hoc de las probabilidades del modelo.

    El modelo se entrena con remuestreo (SMOTE) y pesos de clase, lo que infla
    las probabilidades predichas. Sin calibrar, la puntuación sirve para ordenar
    estudiantes pero NO debe leerse como una probabilidad. Este calibrador
    (isotónico, ajustado sobre datos de validación retenidos) corrige la escala.
    """

    def __init__(self):
        self.model = None
        self.method = None
        self._load()

    def _load(self):
        try:
            import joblib
            path = os.path.join(os.path.dirname(__file__), 'models', 'calibrator.joblib')
            if os.path.exists(path):
                blob = joblib.load(path)
                self.model = blob.get('calibrator') if isinstance(blob, dict) else blob
                self.method = (blob.get('method') if isinstance(blob, dict) else None) or 'isotonic'
                logger.info(f"Calibrador cargado ({self.method})")
            else:
                logger.warning(
                    "No hay calibrador en models/calibrator.joblib. Las puntuaciones "
                    "sirven para ORDENAR estudiantes, no como probabilidades. "
                    "Ejecute fit_calibrator.py para generarlo.")
        except Exception as e:
            logger.warning(f"No se pudo cargar el calibrador: {e}")

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def apply(self, score: float) -> float:
        if self.model is None:
            return float(score)
        try:
            return float(np.clip(self.model.predict([score])[0], 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Fallo al calibrar, se devuelve la puntuación cruda: {e}")
            return float(score)


class BurnoutPredictor:
    """Modelo ML para predicción de riesgo"""
    
    def __init__(self):
        self.calibrator = Calibrator()
        self.model = None
        self.feature_names = [
            'total_clicks', 'days_since_last_access', 'active_days',
            'avg_clicks_per_day', 'avg_grade', 'grade_diff',
            'completion_rate', 'days_enrolled', 'week_of_course'
        ]
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo entrenado o usar uno por defecto"""
        try:
            import joblib
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.joblib')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Modelo cargado desde archivo")
            else:
                self._create_default_model()
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo: {e}. Usando modelo por defecto.")
            self._create_default_model()
    
    def _create_default_model(self):
        """Crear modelo simple basado en reglas si no hay modelo entrenado"""
        logger.info("Usando modelo basado en reglas")
        self.model = None
    
    def extract_features(self, student: StudentFeatures) -> np.ndarray:
        """Extraer features del estudiante"""
        grade_diff = student.avg_grade - student.course_avg_grade
        completion_rate = student.assignments_completed / max(student.assignments_total, 1)
        
        features = np.array([
            student.total_clicks,
            student.days_since_last_access,
            student.active_days,
            student.avg_clicks_per_day,
            student.avg_grade,
            grade_diff,
            completion_rate,
            student.days_enrolled,
            student.week_of_course
        ])
        
        return features.reshape(1, -1)
    
    def predict(self, student: StudentFeatures) -> tuple:
        """Predecir riesgo y detectar factores clave"""
        features = self.extract_features(student)
        
        if self.model is not None:
            # Usar modelo ML
            raw_score = float(self.model.predict_proba(features)[0][1])
        else:
            # Modelo basado en reglas
            raw_score = float(self._rule_based_prediction(student))

        # Calibración post-hoc: convierte la puntuación en una probabilidad
        # interpretable antes de mostrarla a un docente.
        risk_score = self.calibrator.apply(raw_score)

        # Identificar factores clave
        factors = self._identify_key_factors(student)

        # Nivel de riesgo. Los cortes se aplican sobre la probabilidad calibrada:
        # "medio" arranca en la tasa base de abandono de la cohorte, de modo que
        # marcar a un estudiante significa que su riesgo supera al del grupo.
        if risk_score >= THRESHOLD_HIGH:
            risk_level = "alto"
        elif risk_score >= THRESHOLD_MEDIUM:
            risk_level = "medio"
        else:
            risk_level = "bajo"

        return risk_score, risk_level, factors, raw_score
    
    def _rule_based_prediction(self, student: StudentFeatures) -> float:
        """Predicción basada en reglas cuando no hay modelo"""
        score = 0.3  # Base
        
        # Inactividad reciente (peso alto)
        if student.days_since_last_access >= 7:
            score += 0.25
        elif student.days_since_last_access >= 3:
            score += 0.15
        
        # Notas bajas
        grade_diff = student.avg_grade - student.course_avg_grade
        if grade_diff < -15:
            score += 0.2
        elif grade_diff < -5:
            score += 0.1
        
        # Tasa de completado baja
        completion = student.assignments_completed / max(student.assignments_total, 1)
        if completion < 0.5:
            score += 0.15
        elif completion < 0.7:
            score += 0.08
        
        # Baja actividad
        if student.total_clicks < 50:
            score += 0.1
        
        return min(score, 0.95)
    
    def _identify_key_factors(self, student: StudentFeatures) -> List[RiskFactor]:
        """Identificar factores clave de riesgo"""
        factors = []
        
        # Inactividad
        if student.days_since_last_access >= 3:
            impact = "alto" if student.days_since_last_access >= 7 else "medio"
            factors.append(RiskFactor(
                factor="inactividad",
                description=f"Sin acceder al curso en {student.days_since_last_access} días",
                value=float(student.days_since_last_access),
                impact=impact
            ))
        
        # Notas
        grade_diff = student.avg_grade - student.course_avg_grade
        if grade_diff < -5:
            impact = "alto" if grade_diff < -15 else "medio"
            factors.append(RiskFactor(
                factor="rendimiento_academico",
                description=f"Nota media ({student.avg_grade:.1f}%) por debajo del promedio del curso ({student.course_avg_grade:.1f}%)",
                value=float(grade_diff),
                impact=impact
            ))
        
        # Completado de tareas
        completion = student.assignments_completed / max(student.assignments_total, 1)
        if completion < 0.7:
            impact = "alto" if completion < 0.5 else "medio"
            factors.append(RiskFactor(
                factor="tareas_incompletas",
                description=f"Solo ha completado {student.assignments_completed} de {student.assignments_total} tareas ({completion*100:.0f}%)",
                value=float(completion),
                impact=impact
            ))
        
        # Baja actividad
        if student.total_clicks < 100 and student.week_of_course >= 3:
            factors.append(RiskFactor(
                factor="baja_participacion",
                description=f"Actividad muy baja ({student.total_clicks} clicks) para la semana {student.week_of_course}",
                value=float(student.total_clicks),
                impact="medio"
            ))
        
        # Ordenar por impacto
        impact_order = {"alto": 0, "medio": 1, "bajo": 2}
        factors.sort(key=lambda x: impact_order.get(x.impact, 2))
        
        return factors[:3]  # Top 3 factores


# ========== LLM EXPLAINER ==========

class LLMExplainer:
    """Generador de explicaciones usando LLM (OpenAI / Groq compatible)"""
    
    def __init__(self):
        # Soporta OpenAI o Groq (via OpenAI compatible API)
        self.api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.use_llm = bool(self.api_key)
        
        if self.use_llm:
            logger.info(f"LLM habilitado: {self.base_url} / {self.model}")
        else:
            logger.warning("No API key found. Using template-based explanations.")
    
    def generate_explanation(self, 
                            student: StudentFeatures,
                            risk_score: float,
                            risk_level: str,
                            factors: List[RiskFactor],
                            language: str = "es") -> tuple:
        """Generar explicación y recomendaciones"""
        
        if self.use_llm:
            return self._generate_with_llm(student, risk_score, risk_level, factors, language)
        else:
            return self._generate_with_template(student, risk_score, risk_level, factors, language)
    
    def _generate_with_template(self, 
                                student: StudentFeatures,
                                risk_score: float,
                                risk_level: str,
                                factors: List[RiskFactor],
                                language: str) -> tuple:
        """Generar explicación usando templates"""
        
        if language == "es":
            # Explicación
            explanation = f"El estudiante {student.student_name} presenta un nivel de riesgo {risk_level} "
            explanation += f"de abandono ({risk_score*100:.0f}%) en el curso '{student.course_name}'. "
            
            if factors:
                explanation += "Los principales factores de riesgo identificados son: "
                factor_texts = []
                for f in factors:
                    factor_texts.append(f"({len(factor_texts)+1}) {f.description}")
                explanation += "; ".join(factor_texts) + "."
            
            # Recomendaciones basadas en factores
            recommendations = []
            for f in factors:
                if f.factor == "inactividad":
                    recommendations.append("Enviar un mensaje de seguimiento personalizado al estudiante")
                    recommendations.append("Verificar si el estudiante tiene problemas de acceso técnico")
                elif f.factor == "rendimiento_academico":
                    recommendations.append("Ofrecer sesión de tutoría para reforzar conceptos")
                    recommendations.append("Proporcionar materiales de apoyo adicionales")
                elif f.factor == "tareas_incompletas":
                    recommendations.append("Recordar plazos de entrega pendientes")
                    recommendations.append("Revisar si las tareas son accesibles y claras")
                elif f.factor == "baja_participacion":
                    recommendations.append("Invitar a participar en foros de discusión")
                    recommendations.append("Programar una tutoría individual")
            
            # Limitar recomendaciones
            recommendations = list(dict.fromkeys(recommendations))[:3]
            
        else:  # English
            explanation = f"Student {student.student_name} shows {risk_level} risk level "
            explanation += f"of dropout ({risk_score*100:.0f}%) in course '{student.course_name}'. "
            
            if factors:
                explanation += "Key risk factors identified: "
                factor_texts = [f"({i+1}) {f.description}" for i, f in enumerate(factors)]
                explanation += "; ".join(factor_texts) + "."
            
            recommendations = [
                "Send a personalized follow-up message to the student",
                "Offer tutoring support session",
                "Review assignment accessibility"
            ]
        
        return explanation, recommendations
    
    def _generate_with_llm(self,
                           student: StudentFeatures,
                           risk_score: float,
                           risk_level: str,
                           factors: List[RiskFactor],
                           language: str) -> tuple:
        """Generar explicación usando OpenAI/Groq API"""
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Construir prompt
            factors_text = "\n".join([f"- {f.description} (impacto: {f.impact})" for f in factors])
            
            lang_instruction = "en español" if language == "es" else "in English"
            
            prompt = f"""Eres un asistente educativo que ayuda a profesores universitarios 
a entender por qué un estudiante puede estar en riesgo de abandono.

Datos del estudiante:
- Nombre: {student.student_name}
- Curso: {student.course_name}
- Semana del curso: {student.week_of_course}
- Última actividad: hace {student.days_since_last_access} días
- Nota media: {student.avg_grade}% (promedio curso: {student.course_avg_grade}%)
- Tareas completadas: {student.assignments_completed}/{student.assignments_total}
- Actividad total: {student.total_clicks} clicks

Riesgo predicho: {risk_score*100:.0f}% ({risk_level})

Factores de riesgo detectados:
{factors_text}

Genera {lang_instruction}:
1. Una explicación clara y empática de 2-3 frases para el profesor
2. Exactamente 3 recomendaciones concretas de acción

Formato de respuesta (JSON):
{{"explanation": "...", "recommendations": ["rec1", "rec2", "rec3"]}}"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["explanation"], result["recommendations"]
            
        except Exception as e:
            logger.error(f"Error en LLM: {e}")
            return self._generate_with_template(student, risk_score, risk_level, factors, language)


# ========== INSTANCIAS GLOBALES ==========

predictor = BurnoutPredictor()
explainer = LLMExplainer()


# ========== ENDPOINTS ==========

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "calibrated": predictor.calibrator.is_fitted,
        "calibration_method": predictor.calibrator.method,
        "llm_enabled": explainer.use_llm,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest):
    """Predecir riesgo para un estudiante"""
    try:
        student = request.student
        
        # Predicción ML
        risk_score, risk_level, factors, raw_score = predictor.predict(student)
        
        # Explicación LLM
        explanation, recommendations = explainer.generate_explanation(
            student, risk_score, risk_level, factors, request.language
        )
        
        return PredictionResponse(
            student_id=student.student_id,
            course_id=student.course_id,
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            raw_score=round(raw_score, 3),
            calibrated=predictor.calibrator.is_fitted,
            confidence=0.85,  # Placeholder
            explanation=explanation,
            key_factors=factors,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Predecir riesgo para múltiples estudiantes"""
    results = []
    
    for student in request.students:
        try:
            risk_score, risk_level, factors, raw_score = predictor.predict(student)
            explanation, recommendations = explainer.generate_explanation(
                student, risk_score, risk_level, factors, request.language
            )
            
            results.append({
                "student_id": student.student_id,
                "risk_score": round(risk_score, 3),
                "raw_score": round(raw_score, 3),
                "risk_level": risk_level,
                "explanation": explanation,
                "key_factors": [f.dict() for f in factors],
                "recommendations": recommendations
            })
        except Exception as e:
            results.append({
                "student_id": student.student_id,
                "error": str(e)
            })
    
    # Regla de capacidad: si el centro sólo puede atender a una fracción de los
    # estudiantes, marcamos a los de mayor riesgo hasta agotar esa capacidad en
    # lugar de usar un corte fijo de probabilidad.
    flagged = None
    if request.capacity is not None and 0 < request.capacity <= 1:
        scored = [r for r in results if "risk_score" in r]
        scored.sort(key=lambda r: r["risk_score"], reverse=True)
        n_flag = max(1, int(round(request.capacity * len(scored))))
        for i, r in enumerate(scored):
            r["flagged"] = i < n_flag
        flagged = n_flag

    payload = {"predictions": results, "total": len(results),
               "calibrated": predictor.calibrator.is_fitted}
    if flagged is not None:
        payload["capacity"] = request.capacity
        payload["flagged"] = flagged
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
