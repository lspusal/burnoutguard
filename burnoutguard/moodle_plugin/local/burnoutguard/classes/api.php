<?php
// This file is part of Moodle - http://moodle.org/
//
// BurnoutGuard - API Client
//
// @package    local_burnoutguard
// @copyright  2024 Research Project
// @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

namespace local_burnoutguard;

defined('MOODLE_INTERNAL') || die();

/**
 * Cliente API para comunicarse con el backend Python de BurnoutGuard
 */
class api
{

    /** @var string URL del backend */
    private $base_url;

    /** @var int Timeout en segundos */
    private $timeout;

    /**
     * Constructor
     */
    public function __construct()
    {
        $this->base_url = get_config('local_burnoutguard', 'api_url') ?: 'http://localhost:8000';
        $this->timeout = get_config('local_burnoutguard', 'api_timeout') ?: 30;
    }

    /**
     * Predecir riesgo para un estudiante
     * 
     * @param int $studentid ID del estudiante
     * @param int $courseid ID del curso
     * @return object|false Respuesta del API o false si error
     */
    public function predict_student_risk($studentid, $courseid)
    {
        // Extraer features del estudiante
        $features = $this->extract_student_features($studentid, $courseid);

        if (!$features) {
            return false;
        }

        // Llamar al API
        $response = $this->call_api('/predict', [
            'student' => $features,
            'language' => current_language() == 'es' ? 'es' : 'en'
        ]);

        return $response;
    }

    /**
     * Predecir riesgo para todos los estudiantes de un curso
     * 
     * @param int $courseid ID del curso
     * @return array Lista de predicciones
     */
    public function predict_course_risk($courseid)
    {
        global $DB;

        // Obtener estudiantes del curso
        $context = \context_course::instance($courseid);
        $students = get_enrolled_users($context, 'mod/assign:submit');

        $features_list = [];
        foreach ($students as $student) {
            $features = $this->extract_student_features($student->id, $courseid);
            if ($features) {
                $features_list[] = $features;
            }
        }

        if (empty($features_list)) {
            return [];
        }

        // Batch prediction
        $response = $this->call_api('/batch_predict', [
            'students' => $features_list,
            'language' => current_language() == 'es' ? 'es' : 'en'
        ]);

        return $response->predictions ?? [];
    }

    /**
     * Extraer features del estudiante desde Moodle
     * 
     * @param int $studentid
     * @param int $courseid
     * @return array|false
     */
    private function extract_student_features($studentid, $courseid)
    {
        global $DB;

        try {
            $student = $DB->get_record('user', ['id' => $studentid], 'id, firstname, lastname');
            $course = $DB->get_record('course', ['id' => $courseid], 'id, fullname, startdate');

            if (!$student || !$course) {
                return false;
            }

            // Calcular días desde inicio del curso
            $now = time();
            $days_enrolled = max(1, floor(($now - $course->startdate) / 86400));
            $week_of_course = ceil($days_enrolled / 7);

            // === ENGAGEMENT: Logs de actividad ===
            $sql = "SELECT COUNT(*) as total_clicks, MAX(timecreated) as last_access
                    FROM {logstore_standard_log}
                    WHERE userid = :userid AND courseid = :courseid";
            $log_stats = $DB->get_record_sql($sql, ['userid' => $studentid, 'courseid' => $courseid]);

            $total_clicks = $log_stats->total_clicks ?? 0;
            $last_access = $log_stats->last_access ?? $course->startdate;
            $days_since_last_access = floor(($now - $last_access) / 86400);

            // Días activos
            $sql = "SELECT COUNT(DISTINCT DATE(FROM_UNIXTIME(timecreated))) as active_days
                    FROM {logstore_standard_log}
                    WHERE userid = :userid AND courseid = :courseid";
            $active_stats = $DB->get_record_sql($sql, ['userid' => $studentid, 'courseid' => $courseid]);
            $active_days = $active_stats->active_days ?? 0;

            // === ACADEMIC: Notas ===
            $sql = "SELECT AVG(gg.finalgrade) as avg_grade
                    FROM {grade_grades} gg
                    JOIN {grade_items} gi ON gg.itemid = gi.id
                    WHERE gg.userid = :userid AND gi.courseid = :courseid
                    AND gi.itemtype != 'course'";
            $grade_stats = $DB->get_record_sql($sql, ['userid' => $studentid, 'courseid' => $courseid]);
            $avg_grade = $grade_stats->avg_grade ?? 0;

            // Promedio del curso
            $sql = "SELECT AVG(gg.finalgrade) as course_avg
                    FROM {grade_grades} gg
                    JOIN {grade_items} gi ON gg.itemid = gi.id
                    WHERE gi.courseid = :courseid
                    AND gi.itemtype != 'course'
                    AND gg.finalgrade IS NOT NULL";
            $course_avg_stats = $DB->get_record_sql($sql, ['courseid' => $courseid]);
            $course_avg_grade = $course_avg_stats->course_avg ?? 70;

            // === ASSIGNMENTS ===
            $sql = "SELECT COUNT(*) as total
                    FROM {assign} a
                    WHERE a.course = :courseid";
            $total_assigns = $DB->count_records_sql($sql, ['courseid' => $courseid]);

            $sql = "SELECT COUNT(DISTINCT a.id) as completed
                    FROM {assign} a
                    JOIN {assign_submission} s ON a.id = s.assignment
                    WHERE a.course = :courseid AND s.userid = :userid
                    AND s.status = 'submitted'";
            $completed = $DB->count_records_sql($sql, ['courseid' => $courseid, 'userid' => $studentid]);

            return [
                'student_id' => $studentid,
                'student_name' => $student->firstname . ' ' . $student->lastname,
                'course_id' => $courseid,
                'course_name' => $course->fullname,
                'total_clicks' => (int) $total_clicks,
                'days_since_last_access' => (int) $days_since_last_access,
                'active_days' => (int) $active_days,
                'avg_clicks_per_day' => $days_enrolled > 0 ? round($total_clicks / $days_enrolled, 2) : 0,
                'avg_grade' => round($avg_grade, 1),
                'course_avg_grade' => round($course_avg_grade, 1),
                'assignments_completed' => (int) $completed,
                'assignments_total' => max(1, (int) $total_assigns),
                'days_enrolled' => (int) $days_enrolled,
                'week_of_course' => (int) $week_of_course
            ];

        } catch (\Exception $e) {
            debugging('BurnoutGuard: Error extracting features: ' . $e->getMessage(), DEBUG_DEVELOPER);
            return false;
        }
    }

    /**
     * Llamar al API backend
     * 
     * @param string $endpoint
     * @param array $data
     * @return object|false
     */
    private function call_api($endpoint, $data)
    {
        $url = $this->base_url . $endpoint;

        $options = [
            'http' => [
                'header' => "Content-type: application/json\r\n",
                'method' => 'POST',
                'content' => json_encode($data),
                'timeout' => $this->timeout
            ]
        ];

        $context = stream_context_create($options);

        try {
            $response = file_get_contents($url, false, $context);

            if ($response === false) {
                debugging('BurnoutGuard: API call failed to ' . $url, DEBUG_DEVELOPER);
                return false;
            }

            return json_decode($response);

        } catch (\Exception $e) {
            debugging('BurnoutGuard: API exception: ' . $e->getMessage(), DEBUG_DEVELOPER);
            return false;
        }
    }

    /**
     * Verificar conexión con el backend
     * 
     * @return bool
     */
    public function health_check()
    {
        $url = $this->base_url . '/health';

        try {
            $response = @file_get_contents($url);
            $data = json_decode($response);
            return isset($data->status) && $data->status == 'healthy';
        } catch (\Exception $e) {
            return false;
        }
    }
}
