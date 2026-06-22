# Dataset OULAD

## Descripción

Open University Learning Analytics Dataset (OULAD) es el dataset más grande de learning analytics disponible públicamente.

### Fuente
- **Institución**: Open University (UK)
- **Período**: 2013-2014
- **Estudiantes**: 32,593
- **Cursos**: 22
- **Interacciones**: 10,1 millones

### Componentes

#### 1. studentVLE.csv (10,1M filas)
Registro de cada interacción de un estudiante con recursos en Virtual Learning Environment

**Columnas:**
- `id_student`: ID anónimo del estudiante
- `code_module`: Código del módulo/curso
- `code_presentation`: Semestre/año (e.g., '2014A')
- `id_site`: ID del recurso VLE específico
- `date`: Día del semestre (0-299)
- `sum_click`: Número de clicks en ese día para ese recurso

**Ejemplo:**
```
id_student,code_module,code_presentation,id_site,date,sum_click
11391,AAA,2014J,13073,6,10
11391,AAA,2014J,13073,7,9
11391,AAA,2014J,13073,8,24
```

#### 2. studentInfo.csv (32,593 filas)
Información demográfica y resultados finales de estudiantes

**Columnas:**
- `id_student`: ID del estudiante
- `code_module`: Módulo en que está matriculado
- `code_presentation`: Semestre
- `gender`: Género (M/F)
- `age_band`: Grupo de edad (13-17, 18-24, 25-35, 35+)
- `highest_education`: Nivel educativo previo
- `region`: Región geográfica
- `disability`: Si tiene discapacidad registrada
- `num_of_prior_attempts`: Intentos previos en este módulo
- `studied_credits`: Créditos que estaban estudiando
- `final_result`: **[VARIABLE CLAVE]** Resultado final
  - `Distinction`: Sobresaliente (>70%)
  - `Pass`: Aprobado (40-70%)
  - `Fail`: No aprobado (<40%)
  - `Withdrawn`: Abandonó

#### 3. studentAssessment.csv (173,912 filas)
Resultados en evaluaciones individuales

**Columnas:**
- `id_student`: ID del estudiante
- `id_assessment`: ID de la evaluación
- `date_submitted`: Día del semestre en que fue entregada
- `is_banked`: Si fue guardada para uso futuro (0/1)
- `score`: Puntuación obtenida

#### 4. assessments.csv (607 filas)
Metadatos de evaluaciones

**Columnas:**
- `code_module`: Módulo
- `code_presentation`: Semestre
- `id_assessment`: ID de evaluación
- `assessment_type`: Tipo (TMA: Tutor Marked Assignment, CMA: Computer Marked Assignment, Exam)
- `date`: Día del semestre cuando se abre
- `weight`: Peso en calificación final (0-100)

#### 5. vle.csv (32,593 filas)
Metadatos de recursos VLE (Virtual Learning Environment)

**Columnas:**
- `code_module`: Módulo
- `code_presentation`: Semestre
- `id_site`: ID del recurso
- `activity_type`: Tipo de recurso (Page, Quiz, Discussion Forum, Resource, ...)
- `week_from`: Semana en que aparece
- `week_to`: Última semana disponible

#### 6. courses.csv (22 filas)
Información agregada de módulos/cursos

**Columnas:**
- `code_module`: Código del módulo
- `code_presentation`: Semestre
- `module_presentation_length`: Duración en días

## Descarga

```bash
python data/download_oulad.py --output data/ --extract True
```

Este script descargará automáticamente ~10GB de datos y los descomprimirá en CSV.

## Consideraciones Éticas

- ✅ Datos completamente anonimizados (no hay nombres, emails, IPs)
- ✅ Acceso público a través de Analyse KMi
- ✅ Publicado bajo términos de uso académico
- ✅ Usado en 200+ estudios publicados

## Referencia

Kuzilek, J., Vitek, M., & Klimes, V. (2015). Open University Learning Analytics Dataset. Scientific Data, 2, 150005.

https://analyse.kmi.open.ac.uk/open_dataset
