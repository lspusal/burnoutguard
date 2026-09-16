<?php
// This file is part of Moodle - http://moodle.org/
//
// BurnoutGuard - Dashboard Page
//
// @package    local_burnoutguard
// @copyright  2024 Research Project
// @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

require_once(__DIR__ . '/../../config.php');
require_once($CFG->libdir . '/adminlib.php');

$courseid = required_param('courseid', PARAM_INT);

$course = $DB->get_record('course', ['id' => $courseid], '*', MUST_EXIST);
$context = context_course::instance($courseid);

require_login($course);
require_capability('local/burnoutguard:viewrisk', $context);

$PAGE->set_url('/local/burnoutguard/dashboard.php', ['courseid' => $courseid]);
$PAGE->set_context($context);
$PAGE->set_title(get_string('dashboard', 'local_burnoutguard'));
$PAGE->set_heading($course->fullname . ' - ' . get_string('pluginname', 'local_burnoutguard'));
$PAGE->set_pagelayout('incourse');

// Obtener predicciones
$api = new \local_burnoutguard\api();

// Verificar conexión con backend
if (!$api->health_check()) {
    $error = get_string('api_error', 'local_burnoutguard');
    $predictions = [];
} else {
    $predictions = $api->predict_course_risk($courseid);
}

// Ordenar por riesgo (mayor primero)
usort($predictions, function ($a, $b) {
    return ($b->risk_score ?? 0) <=> ($a->risk_score ?? 0);
});

// Categorizar
$high_risk = array_filter($predictions, fn($p) => ($p->risk_level ?? '') == 'alto');
$medium_risk = array_filter($predictions, fn($p) => ($p->risk_level ?? '') == 'medio');
$low_risk = array_filter($predictions, fn($p) => ($p->risk_level ?? '') == 'bajo');

// Renderizar
echo $OUTPUT->header();

// Stats cards
$total = count($predictions);
$high_count = count($high_risk);
$medium_count = count($medium_risk);
$low_count = count($low_risk);

?>

<style>
    .burnoutguard-stats {
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }

    .stat-card {
        background: #fff;
        border-radius: 8px;
        padding: 20px;
        min-width: 150px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .stat-card.high {
        border-left: 4px solid #dc3545;
    }

    .stat-card.medium {
        border-left: 4px solid #ffc107;
    }

    .stat-card.low {
        border-left: 4px solid #28a745;
    }

    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
    }

    .stat-label {
        color: #666;
        margin-top: 5px;
    }

    .student-card {
        background: #fff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .student-card.high {
        border-left: 4px solid #dc3545;
    }

    .student-card.medium {
        border-left: 4px solid #ffc107;
    }

    .student-card.low {
        border-left: 4px solid #28a745;
    }

    .student-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }

    .student-name {
        font-size: 1.2em;
        font-weight: bold;
    }

    .risk-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
    }

    .risk-badge.high {
        background: #dc3545;
    }

    .risk-badge.medium {
        background: #ffc107;
        color: #333;
    }

    .risk-badge.low {
        background: #28a745;
    }

    .explanation {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        font-style: italic;
    }

    .factors-list,
    .recommendations-list {
        margin: 10px 0;
        padding-left: 20px;
    }

    .factor-item {
        margin: 5px 0;
    }

    .factor-impact {
        font-size: 0.85em;
        padding: 2px 8px;
        border-radius: 3px;
        margin-left: 10px;
    }

    .factor-impact.alto {
        background: #ffe0e0;
        color: #c00;
    }

    .factor-impact.medio {
        background: #fff3cd;
        color: #856404;
    }

    .factor-impact.bajo {
        background: #d4edda;
        color: #155724;
    }

    .section-title {
        font-size: 1.3em;
        font-weight: bold;
        margin: 30px 0 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #ddd;
    }
</style>

<div class="burnoutguard-dashboard">

    <div class="burnoutguard-stats">
        <div class="stat-card high">
            <div class="stat-number"><?php echo $high_count; ?></div>
            <div class="stat-label"><?php echo get_string('high_risk', 'local_burnoutguard'); ?></div>
        </div>
        <div class="stat-card medium">
            <div class="stat-number"><?php echo $medium_count; ?></div>
            <div class="stat-label"><?php echo get_string('medium_risk', 'local_burnoutguard'); ?></div>
        </div>
        <div class="stat-card low">
            <div class="stat-number"><?php echo $low_count; ?></div>
            <div class="stat-label"><?php echo get_string('low_risk', 'local_burnoutguard'); ?></div>
        </div>
        <div class="stat-card">
            <div class="stat-number"><?php echo $total; ?></div>
            <div class="stat-label"><?php echo get_string('total_students', 'local_burnoutguard'); ?></div>
        </div>
    </div>

    <?php if (!empty($error)): ?>
        <div class="alert alert-danger"><?php echo $error; ?></div>
    <?php endif; ?>

    <?php if ($high_count > 0): ?>
        <h3 class="section-title" style="color: #dc3545;">
            ⚠️ <?php echo get_string('high_risk_students', 'local_burnoutguard'); ?>
        </h3>

        <?php foreach ($high_risk as $prediction): ?>
            <?php echo render_student_card($prediction, 'high'); ?>
        <?php endforeach; ?>
    <?php endif; ?>

    <?php if ($medium_count > 0): ?>
        <h3 class="section-title" style="color: #ffc107;">
            ⚡ <?php echo get_string('medium_risk_students', 'local_burnoutguard'); ?>
        </h3>

        <?php foreach ($medium_risk as $prediction): ?>
            <?php echo render_student_card($prediction, 'medium'); ?>
        <?php endforeach; ?>
    <?php endif; ?>

    <?php if ($low_count > 0): ?>
        <h3 class="section-title" style="color: #28a745;">
            ✅ <?php echo get_string('low_risk_students', 'local_burnoutguard'); ?>
        </h3>

        <?php foreach (array_slice($low_risk, 0, 5) as $prediction): ?>
            <?php echo render_student_card($prediction, 'low'); ?>
        <?php endforeach; ?>

        <?php if ($low_count > 5): ?>
            <p class="text-muted">... y <?php echo $low_count - 5; ?> estudiantes más con bajo riesgo.</p>
        <?php endif; ?>
    <?php endif; ?>

</div>

<?php

echo $OUTPUT->footer();

/**
 * Render a student card
 */
function render_student_card($prediction, $level)
{
    global $DB;

    $student = $DB->get_record('user', ['id' => $prediction->student_id]);
    $name = $student ? fullname($student) : 'Estudiante #' . $prediction->student_id;

    $risk_percent = round(($prediction->risk_score ?? 0) * 100);
    $explanation = $prediction->explanation ?? '';
    $factors = $prediction->key_factors ?? [];
    $recommendations = $prediction->recommendations ?? [];

    $level_text = [
        'high' => get_string('high', 'local_burnoutguard'),
        'medium' => get_string('medium', 'local_burnoutguard'),
        'low' => get_string('low', 'local_burnoutguard')
    ][$level] ?? $level;

    $html = '<div class="student-card ' . $level . '">';
    $html .= '<div class="student-header">';
    $html .= '<span class="student-name">' . $name . '</span>';
    $html .= '<span class="risk-badge ' . $level . '">' . $risk_percent . '% - ' . $level_text . '</span>';
    $html .= '</div>';

    if ($explanation) {
        $html .= '<div class="explanation">"' . $explanation . '"</div>';
    }

    if (!empty($factors)) {
        $html .= '<strong>' . get_string('key_factors', 'local_burnoutguard') . ':</strong>';
        $html .= '<ul class="factors-list">';
        foreach ($factors as $factor) {
            $impact = $factor->impact ?? 'medio';
            $html .= '<li class="factor-item">';
            $html .= $factor->description ?? $factor->factor;
            $html .= '<span class="factor-impact ' . $impact . '">' . $impact . '</span>';
            $html .= '</li>';
        }
        $html .= '</ul>';
    }

    if (!empty($recommendations)) {
        $html .= '<strong>' . get_string('recommendations', 'local_burnoutguard') . ':</strong>';
        $html .= '<ul class="recommendations-list">';
        foreach ($recommendations as $rec) {
            $html .= '<li>' . $rec . '</li>';
        }
        $html .= '</ul>';
    }

    $html .= '</div>';

    return $html;
}
