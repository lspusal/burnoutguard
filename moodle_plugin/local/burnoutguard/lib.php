<?php
// This file is part of Moodle - http://moodle.org/
//
// BurnoutGuard - Library functions
//
// @package    local_burnoutguard
// @copyright  2024 Research Project
// @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

defined('MOODLE_INTERNAL') || die();

/**
 * Extend navigation to add BurnoutGuard link
 */
function local_burnoutguard_extend_navigation(global_navigation $navigation)
{
    global $PAGE, $COURSE;

    if ($PAGE->context->contextlevel >= CONTEXT_COURSE) {
        $courseid = $COURSE->id;
        if ($courseid && has_capability('local/burnoutguard:viewrisk', $PAGE->context)) {
            $url = new moodle_url('/local/burnoutguard/dashboard.php', ['courseid' => $courseid]);
            $navigation->add(
                get_string('pluginname', 'local_burnoutguard'),
                $url,
                navigation_node::TYPE_CUSTOM,
                null,
                'burnoutguard',
                new pix_icon('i/report', '')
            );
        }
    }
}

/**
 * Add link to course administration
 */
function local_burnoutguard_extend_settings_navigation(settings_navigation $settingsnav, context $context)
{
    global $PAGE;

    if ($context->contextlevel != CONTEXT_COURSE) {
        return;
    }

    if (!has_capability('local/burnoutguard:viewrisk', $context)) {
        return;
    }

    $courseid = $context->instanceid;

    if ($settingnode = $settingsnav->find('courseadmin', navigation_node::TYPE_COURSE)) {
        $url = new moodle_url('/local/burnoutguard/dashboard.php', ['courseid' => $courseid]);
        $node = navigation_node::create(
            get_string('pluginname', 'local_burnoutguard'),
            $url,
            navigation_node::TYPE_SETTING,
            null,
            'burnoutguard',
            new pix_icon('i/report', '')
        );
        $settingnode->add_node($node);
    }
}
