# BurnoutGuard - Sistema de Alerta Temprana con IA Explicable

## 📋 Descripción

Plugin de Moodle que predice el riesgo de abandono/burnout académico y proporciona explicaciones en lenguaje natural usando IA generativa.

**Contribuciones técnicas:**
1. Integración ML + LLM en plataforma LMS
2. Explicaciones automáticas en español/inglés
3. Dashboard visual para profesores
4. API REST reutilizable

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                         MOODLE                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Plugin: local_burnoutguard                │ │
│  │  • Extrae logs, notas, actividad                       │ │
│  │  • Dashboard visual de riesgo                          │ │
│  └─────────────────────────┬──────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON BACKEND (FastAPI)                 │
│  ┌─────────────────┐   ┌─────────────────────────────────┐ │
│  │ ML Predictor    │   │ LLM Explainer                   │ │
│  │ (XGBoost/Rules) │   │ (Groq/Llama or OpenAI;          │ │
│  │                 │   │  fallback to templates)         │ │
│  └─────────────────┘   └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Archivos

```
burnoutguard/
├── backend/                    # Python FastAPI
│   ├── main.py                 # API: predicción, calibración y explicaciones
│   ├── fit_calibrator.py       # Entrena el modelo y ajusta su calibrador
│   ├── models/                 # Modelo y calibrador ya entrenados
│   ├── requirements.txt        # Dependencias
│   └── Dockerfile
│
├── docker-compose.yml          # Moodle + backend para pruebas locales
└── moodle_plugin/              # Plugin Moodle
    └── local/burnoutguard/
        ├── version.php         # Versión
        ├── lib.php             # Funciones principales
        ├── dashboard.php       # UI para profesores
        ├── classes/
        │   └── api.php         # Cliente API (200 líneas)
        ├── db/
        │   └── access.php      # Permisos
        └── lang/
            ├── es/             # Español
            └── en/             # Inglés
```

---

## 🚀 Instalación

### 1. Backend Python

```bash
cd burnoutguard/backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt   # fastapi, uvicorn, pydantic, numpy,
                                  # scikit-learn, xgboost, joblib, openai

# (Opcional) Configurar OpenAI para explicaciones LLM
export OPENAI_API_KEY="sk-..."

# Ejecutar servidor
python main.py
# o: uvicorn main:app --reload --port 8000
```

El servidor estará en: http://localhost:8000

### 2. Plugin Moodle

```bash
# Copiar a Moodle
cp -r moodle_plugin/local/burnoutguard /path/to/moodle/local/

# Ir a Moodle como admin
# Notificaciones > Actualizar base de datos
```

### 3. Configuración

En Moodle:
- Ir a: Administración > Plugins > Locales > BurnoutGuard
- Configurar URL del backend: `http://localhost:8000`

---

## 📡 API Endpoints

### POST /predict
Predecir riesgo para un estudiante.

**Request:**
```json
{
  "student": {
    "student_id": 123,
    "student_name": "María García",
    "course_id": 45,
    "course_name": "Matemáticas I",
    "total_clicks": 150,
    "days_since_last_access": 5,
    "avg_grade": 65.0,
    "assignments_completed": 3,
    "assignments_total": 5
  },
  "language": "es"
}
```

**Response:**
```json
{
  "student_id": 123,
  "risk_score": 0.72,
  "risk_level": "alto",
  "explanation": "María presenta un nivel de riesgo alto porque: 
    (1) no ha accedido al curso en 5 días, 
    (2) su nota media (65%) está por debajo del promedio (72%).",
  "key_factors": [
    {"factor": "inactividad", "description": "Sin acceder en 5 días", "impact": "alto"}
  ],
  "recommendations": [
    "Enviar mensaje de seguimiento al estudiante",
    "Ofrecer tutoría personalizada"
  ]
}
```

### POST /batch_predict
Predecir para múltiples estudiantes.

### GET /health
Verificar estado del servicio.

---

## 🖥️ Ejemplo de Dashboard

El dashboard muestra a los profesores:

- 📊 **Resumen estadístico**: Estudiantes en riesgo alto/medio/bajo
- 🚨 **Tarjetas de riesgo alto**: Con explicación IA y recomendaciones
- ⚡ **Tarjetas de riesgo medio**: Seguimiento preventivo
- ✅ **Estudiantes estables**: Resumen rápido

---

## 🎯 Calibración y punto de operación

El modelo se entrena con SMOTE y pesos de clase, lo que **infla las probabilidades
predichas**: sin corregirlo, una puntuación cruda de 0.81 puede corresponder a una
probabilidad real de abandono de 0.41. Como el panel muestra niveles de riesgo y no
un simple ranking, el backend aplica un **calibrador isotónico post-hoc** ajustado
sobre datos de validación retenidos.

El repositorio incluye `backend/models/calibrator.joblib` y `backend/models/xgboost_model.joblib`
ya entrenados, de modo que el backend calibra desde el primer arranque. Para
regenerarlos con tus propios datos:

```bash
cd backend
python fit_calibrator.py --data ../../data --week 8
```

En OULAD, la calibración reduce el ECE de **0.207 a 0.015** y el Brier de 0.187 a
0.133, dejando el AUROC prácticamente intacto (0.798 → 0.797): no mejora la
capacidad de ordenar, pero hace que las probabilidades signifiquen lo que dicen.

`GET /health` indica si hay calibrador cargado (`"calibrated": true`). Si falta, el
backend sigue respondiendo pero registra un aviso: en ese caso la puntuación sólo
sirve para **ordenar** estudiantes, no como probabilidad.

**Niveles de riesgo.** Los cortes se aplican sobre la probabilidad calibrada y son
configurables por variables de entorno (`RISK_THRESHOLD_HIGH`, por defecto 0.50;
`RISK_THRESHOLD_MEDIUM`, por defecto 0.25 ≈ la tasa base de abandono).

**Regla de capacidad.** Si el centro sólo puede atender a una fracción de los
estudiantes, `POST /batch_predict` acepta `capacity` (p. ej. `0.2`) y marca al 20%
de mayor riesgo en lugar de usar un corte fijo:

```json
{ "students": [...], "language": "es", "capacity": 0.2 }
```

La respuesta añade `flagged` a cada estudiante y devuelve `"calibrated"` y el número
de estudiantes marcados.

---

## 🔧 Personalización

### Añadir nuevo factor de riesgo

En `backend/main.py`, método `_identify_key_factors()`:

```python
# Ejemplo: añadir factor de participación en foros
if student.forum_posts < 2 and student.week_of_course >= 4:
    factors.append(RiskFactor(
        factor="baja_participacion_foros",
        description=f"Solo {student.forum_posts} mensajes en foros",
        value=float(student.forum_posts),
        impact="medio"
    ))
```

### Reentrenar el modelo

`backend/fit_calibrator.py` entrena el modelo y ajusta su calibrador en un solo
paso, y deja ambos en `backend/models/`, que es donde el backend los busca al
arrancar:

```bash
cd backend
python fit_calibrator.py --data ../../data --week 8
```

Si prefieres usar un modelo propio, guárdalo como
`backend/models/xgboost_model.joblib` con `joblib.dump()`; debe aceptar las nueve
features en el orden de `BurnoutPredictor.extract_features()`. Recuerda ajustar
también su calibrador, o el panel mostrará probabilidades infladas.

---

## 📄 Licencia

GNU GPL v3 - Para uso académico e investigación.

---

## 📚 Citar

```bibtex
@software{burnoutguard2024,
  title = {BurnoutGuard: Early Warning System with Explainable AI},
  year = {2024},
  author = {Research Project},
  url = {https://github.com/lspusal/burnoutguard}
}
```
