<?php
// This file is part of Moodle - http://moodle.org/
//
// BurnoutGuard - Admin Settings
//
// @package    local_burnoutguard
// @copyright  2024 Research Project
// @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage('local_burnoutguard', get_string('pluginname', 'local_burnoutguard'));

    // API URL
    $settings->add(new admin_setting_configtext(
        'local_burnoutguard/api_url',
        get_string('api_url', 'local_burnoutguard'),
        get_string('api_url_desc', 'local_burnoutguard'),
        'http://localhost:8000',
        PARAM_URL
    ));

    // API Timeout
    $settings->add(new admin_setting_configtext(
        'local_burnoutguard/api_timeout',
        get_string('api_timeout', 'local_burnoutguard'),
        get_string('api_timeout_desc', 'local_burnoutguard'),
        '30',
        PARAM_INT
    ));

    // Groq API Key
    $settings->add(new admin_setting_configpasswordunmask(
        'local_burnoutguard/groq_api_key',
        get_string('groq_api_key', 'local_burnoutguard'),
        get_string('groq_api_key_desc', 'local_burnoutguard'),
        ''
    ));

    // LLM Model
    $settings->add(new admin_setting_configselect(
        'local_burnoutguard/llm_model',
        get_string('llm_model', 'local_burnoutguard'),
        get_string('llm_model_desc', 'local_burnoutguard'),
        'llama-3.3-70b-versatile',
        [
            'llama-3.3-70b-versatile' => 'Llama 3.3 70B (Recomendado)',
            'llama-3.1-8b-instant' => 'Llama 3.1 8B (Más rápido)',
            'mixtral-8x7b-32768' => 'Mixtral 8x7B',
            'gemma2-9b-it' => 'Gemma 2 9B'
        ]
    ));

    // Enable/Disable LLM
    $settings->add(new admin_setting_configcheckbox(
        'local_burnoutguard/enable_llm',
        get_string('enable_llm', 'local_burnoutguard'),
        get_string('enable_llm_desc', 'local_burnoutguard'),
        1
    ));

    $ADMIN->add('localplugins', $settings);
}
