<?php
// This file is part of Moodle - http://moodle.org/
//
// BurnoutGuard - Spanish language strings
//
// @package    local_burnoutguard
// @copyright  2024 Research Project
// @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

$string['pluginname'] = 'BurnoutGuard - Sistema de Alerta Temprana';
$string['dashboard'] = 'Panel de Riesgo de Abandono';

// Risk levels
$string['high_risk'] = 'Riesgo Alto';
$string['medium_risk'] = 'Riesgo Medio';
$string['low_risk'] = 'Riesgo Bajo';
$string['high'] = 'Alto';
$string['medium'] = 'Medio';
$string['low'] = 'Bajo';

// Dashboard
$string['total_students'] = 'Total Estudiantes';
$string['high_risk_students'] = 'Estudiantes en Riesgo Alto';
$string['medium_risk_students'] = 'Estudiantes en Riesgo Medio';
$string['low_risk_students'] = 'Estudiantes en Riesgo Bajo';

// Card content
$string['key_factors'] = 'Factores de riesgo identificados';
$string['recommendations'] = 'Recomendaciones de acción';

// Errors
$string['api_error'] = 'Error de conexión con el servicio de predicción. Por favor, contacte al administrador.';

// Capabilities
$string['burnoutguard:viewrisk'] = 'Ver predicciones de riesgo de abandono';
$string['burnoutguard:manage'] = 'Gestionar configuración de BurnoutGuard';

// Settings
$string['settings'] = 'Configuración de BurnoutGuard';
$string['api_url'] = 'URL del API Backend';
$string['api_url_desc'] = 'URL del servidor Python que ejecuta el modelo de predicción (ej: http://localhost:8000)';
$string['api_timeout'] = 'Timeout API (segundos)';
$string['api_timeout_desc'] = 'Tiempo máximo de espera para respuestas del API';
$string['groq_api_key'] = 'Clave API de Groq';
$string['groq_api_key_desc'] = 'Tu clave API de Groq para generar explicaciones con IA (obtener en console.groq.com)';
$string['llm_model'] = 'Modelo LLM';
$string['llm_model_desc'] = 'Modelo de lenguaje a utilizar para generar explicaciones';
$string['enable_llm'] = 'Habilitar explicaciones IA';
$string['enable_llm_desc'] = 'Activar la generación de explicaciones automáticas con IA';
