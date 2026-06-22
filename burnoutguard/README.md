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
│   ├── main.py                 # API principal (450 líneas)
│   ├── requirements.txt        # Dependencias
│   └── models/                 # Modelos entrenados
│
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
pip install fastapi uvicorn pydantic openai numpy

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

### Usar modelo ML entrenado

```bash
# Entrenar y guardar modelo
python
>>> from sklearn.externals import joblib
>>> joblib.dump(model, 'backend/models/xgboost_model.joblib')
```

El backend lo cargará automáticamente.

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
  url = {https://github.com/...}
}
```
