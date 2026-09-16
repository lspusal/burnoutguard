<?php
// This file is part of Moodle - http://moodle.org/
//
// BurnoutGuard - English language strings
//
// @package    local_burnoutguard
// @copyright  2024 Research Project
// @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

$string['pluginname'] = 'BurnoutGuard - Early Warning System';
$string['dashboard'] = 'Dropout Risk Dashboard';

// Risk levels
$string['high_risk'] = 'High Risk';
$string['medium_risk'] = 'Medium Risk';
$string['low_risk'] = 'Low Risk';
$string['high'] = 'High';
$string['medium'] = 'Medium';
$string['low'] = 'Low';

// Dashboard
$string['total_students'] = 'Total Students';
$string['high_risk_students'] = 'High Risk Students';
$string['medium_risk_students'] = 'Medium Risk Students';
$string['low_risk_students'] = 'Low Risk Students';

// Card content
$string['key_factors'] = 'Key risk factors identified';
$string['recommendations'] = 'Recommended actions';

// Errors
$string['api_error'] = 'Connection error with prediction service. Please contact administrator.';

// Capabilities
$string['burnoutguard:viewrisk'] = 'View dropout risk predictions';
$string['burnoutguard:manage'] = 'Manage BurnoutGuard settings';

// Settings
$string['settings'] = 'BurnoutGuard Settings';
$string['api_url'] = 'API Backend URL';
$string['api_url_desc'] = 'URL of the Python server running the prediction model (e.g. http://localhost:8000)';
$string['api_timeout'] = 'API Timeout (seconds)';
$string['api_timeout_desc'] = 'Maximum wait time for API responses';
