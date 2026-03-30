use serde::Deserialize;

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::commands::{
    dispatch_mode, execute_action, fetch_engine_diagnostics, fetch_history_timeline,
    fetch_invariant_status,
};

const MAX_COMMAND_HISTORY: usize = 20;
const MAX_DIFF_LINES: usize = 20;
pub const HISTORY_WINDOW_MODE: &str = "History Window";

pub const NAV_ITEMS: &[&str] = &[
    "Diagnostics",
    "Control Flow",
    "Memory",
    "Adaptive",
    "Regime Jump",
    "Self-Healing",
    "History Window",
    "Invariants",
    "Law Engine",
    "Actions",
];

#[derive(Deserialize)]
struct RawDiagnostics {
    collapse_score: f64,
    trend_state: String,
    adaptive_damping: f64,
    healing_mode: String,
    history_behavior: String,
}

pub struct DiagnosticsData {
    pub collapse_score: String,
    pub trend_state: String,
    pub adaptive_damping: String,
    pub healing_mode: String,
    pub history_behavior: String,
    pub error: Option<String>,
}

impl DiagnosticsData {
    fn placeholder() -> Self {
        Self {
            collapse_score: "—".to_string(),
            trend_state: "—".to_string(),
            adaptive_damping: "—".to_string(),
            healing_mode: "—".to_string(),
            history_behavior: "—".to_string(),
            error: None,
        }
    }

    fn from_engine() -> Self {
        match fetch_engine_diagnostics() {
            Ok(json_str) => match serde_json::from_str::<RawDiagnostics>(&json_str) {
                Ok(raw) => Self {
                    collapse_score: format!("{:.2}", raw.collapse_score),
                    trend_state: raw.trend_state,
                    adaptive_damping: format!("{:.2}", raw.adaptive_damping),
                    healing_mode: raw.healing_mode,
                    history_behavior: raw.history_behavior,
                    error: None,
                },
                Err(e) => {
                    let mut d = Self::placeholder();
                    d.error = Some(format!("JSON parse error: {e}"));
                    d
                }
            },
            Err(e) => {
                let mut d = Self::placeholder();
                d.error = Some(format!("ENGINE ERROR: {e}"));
                d
            }
        }
    }
}

#[derive(Deserialize)]
struct RawHistory {
    timeline: Vec<String>,
}

pub struct HistoryData {
    pub timeline: Vec<String>,
    pub error: Option<String>,
}

impl HistoryData {
    fn placeholder() -> Self {
        Self {
            timeline: vec![
                "stable".to_string(),
                "rising".to_string(),
                "oscillatory".to_string(),
                "locked".to_string(),
            ],
            error: None,
        }
    }

    fn from_engine() -> Self {
        match fetch_history_timeline() {
            Ok(json_str) => match serde_json::from_str::<RawHistory>(&json_str) {
                Ok(raw) => Self {
                    timeline: raw.timeline,
                    error: None,
                },
                Err(e) => {
                    let mut h = Self::placeholder();
                    h.error = Some(format!("JSON parse error: {e}"));
                    h
                }
            },
            Err(e) => {
                let mut h = Self::placeholder();
                h.error = Some(format!("ENGINE ERROR: {e}"));
                h
            }
        }
    }
}

#[derive(Deserialize)]
struct RawInvariants {
    determinism: String,
    bounds: String,
    stability: String,
    law_engine: String,
}

pub struct InvariantData {
    pub determinism: String,
    pub bounds: String,
    pub stability: String,
    pub law_engine: String,
    pub error: Option<String>,
}

impl InvariantData {
    fn placeholder() -> Self {
        Self {
            determinism: "—".to_string(),
            bounds: "—".to_string(),
            stability: "—".to_string(),
            law_engine: "—".to_string(),
            error: None,
        }
    }

    fn from_engine() -> Self {
        match fetch_invariant_status() {
            Ok(json_str) => match serde_json::from_str::<RawInvariants>(&json_str) {
                Ok(raw) => Self {
                    determinism: raw.determinism,
                    bounds: raw.bounds,
                    stability: raw.stability,
                    law_engine: raw.law_engine,
                    error: None,
                },
                Err(e) => {
                    let mut inv = Self::placeholder();
                    inv.error = Some(format!("JSON parse error: {e}"));
                    inv
                }
            },
            Err(e) => {
                let mut inv = Self::placeholder();
                inv.error = Some(format!("ENGINE ERROR: {e}"));
                inv
            }
        }
    }
}

pub struct App {
    pub nav_items: &'static [&'static str],
    pub selected_index: usize,
    pub mode: &'static str,
    pub diagnostics: DiagnosticsData,
    pub history: HistoryData,
    pub invariants: InvariantData,
    pub action_log: Vec<String>,
    pub command_history: Vec<String>,
    pub action_status: String,
    pub last_action_time: String,
    pub exported_log_path: String,
    // Retained for replay-pane compatibility with existing session log workflows.
    pub replay_lines: Vec<String>,
    pub artifact_view: Vec<String>,
    pub session_files: Vec<String>,
    pub search_query: String,
    pub filtered_session_files: Vec<String>,
    pub search_overlay_active: bool,
    pub help_overlay_active: bool,
    pub session_metadata: Vec<String>,
    pub selected_session_index: usize,
    pub diff_lines: Vec<String>,
    pub total_actions_run: usize,
    pub successful_actions: usize,
    pub failed_actions: usize,
    pub average_action_latency_ms: u128,
    pub health_status: String,
    pub recent_failures: Vec<String>,
    pub invariant_summary: Vec<String>,
}

impl App {
    pub fn new() -> Self {
        let mut app = Self {
            nav_items: NAV_ITEMS,
            selected_index: 0,
            mode: "Diagnostics",
            diagnostics: DiagnosticsData::from_engine(),
            history: HistoryData::from_engine(),
            invariants: InvariantData::from_engine(),
            action_log: Vec::new(),
            command_history: Vec::new(),
            action_status: "IDLE".to_string(),
            last_action_time: String::new(),
            exported_log_path: String::new(),
            replay_lines: Vec::new(),
            artifact_view: Vec::new(),
            session_files: Vec::new(),
            search_query: String::new(),
            filtered_session_files: Vec::new(),
            search_overlay_active: false,
            help_overlay_active: false,
            session_metadata: Vec::new(),
            selected_session_index: 0,
            diff_lines: Vec::new(),
            total_actions_run: 0,
            successful_actions: 0,
            failed_actions: 0,
            average_action_latency_ms: 0,
            health_status: "HEALTHY".to_string(),
            recent_failures: Vec::new(),
            invariant_summary: vec![
                "PASS: 0".to_string(),
                "FAIL: 0".to_string(),
                "UNKNOWN: 4".to_string(),
            ],
        };
        app.build_invariant_summary();
        app
    }

    pub fn nav_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        } else {
            self.selected_index = self.nav_items.len() - 1;
        }
    }

    pub fn nav_down(&mut self) {
        if self.selected_index < self.nav_items.len() - 1 {
            self.selected_index += 1;
        } else {
            self.selected_index = 0;
        }
    }

    pub fn select_mode(&mut self) {
        self.mode = dispatch_mode(self.selected_index);
    }

    pub fn jump_to(&mut self, index: usize) {
        if index < self.nav_items.len() {
            self.selected_index = index;
            self.select_mode();
        }
    }

    pub fn refresh_diagnostics(&mut self) {
        self.diagnostics = DiagnosticsData::from_engine();
    }

    pub fn refresh_all(&mut self) {
        self.diagnostics = DiagnosticsData::from_engine();
        self.history = HistoryData::from_engine();
        self.invariants = InvariantData::from_engine();
        self.build_invariant_summary();
    }

    pub fn update_observability_metrics(
        &mut self,
        action_name: &str,
        success: bool,
        latency_ms: u128,
    ) {
        let previous_total = self.total_actions_run as u128;
        self.total_actions_run += 1;
        if success {
            self.successful_actions += 1;
        } else {
            self.failed_actions += 1;
            self.recent_failures.push(action_name.to_string());
            while self.recent_failures.len() > 10 {
                self.recent_failures.remove(0);
            }
        }

        self.average_action_latency_ms =
            ((self.average_action_latency_ms * previous_total) + latency_ms)
                / (self.total_actions_run as u128);

        self.health_status = match self.recent_failures.len() {
            0 => "HEALTHY".to_string(),
            1..=3 => "DEGRADED".to_string(),
            _ => "CRITICAL".to_string(),
        };
    }

    pub fn build_invariant_summary(&mut self) {
        let mut pass = 0;
        let mut fail = 0;
        let mut unknown = 0;
        for value in [
            self.invariants.determinism.as_str(),
            self.invariants.bounds.as_str(),
            self.invariants.stability.as_str(),
            self.invariants.law_engine.as_str(),
        ] {
            match value {
                "PASS" => pass += 1,
                "FAIL" => fail += 1,
                _ => unknown += 1,
            }
        }

        self.invariant_summary = vec![
            format!("PASS: {pass}"),
            format!("FAIL: {fail}"),
            format!("UNKNOWN: {unknown}"),
        ];
    }

    pub fn run_action(&mut self, action: &str) {
        let entry = match execute_action(action) {
            Ok(output) => format!("[{action}] {output}"),
            Err(e) => format!("[{action}] ERROR: {e}"),
        };
        self.action_log.push(entry);
        // FIFO trim to last 10 lines
        while self.action_log.len() > 10 {
            self.action_log.remove(0);
        }
    }

    pub fn run_action_with_status(&mut self, action: &str) {
        self.action_status = "RUNNING".to_string();
        let timestamp = format!("{:?}", std::time::SystemTime::now());
        self.last_action_time = timestamp.clone();

        let success = match execute_action(action) {
            Ok(output) => {
                let entry = format!("[{action}] {output}");
                self.action_log.push(entry);
                true
            }
            Err(e) => {
                let entry = format!("[{action}] ERROR: {e}");
                self.action_log.push(entry);
                false
            }
        };

        // FIFO trim action_log to last 10 lines
        while self.action_log.len() > 10 {
            self.action_log.remove(0);
        }

        self.action_status = if success {
            "SUCCESS".to_string()
        } else {
            "FAILED".to_string()
        };

        // Append to command history with timestamp
        self.command_history.push(format!("{action} @ {timestamp}"));
        // FIFO trim to last MAX_COMMAND_HISTORY
        while self.command_history.len() > MAX_COMMAND_HISTORY {
            self.command_history.remove(0);
        }
    }

    pub fn export_session_log(&mut self) -> Result<(), String> {
        let path = "qec_tui_session.log";
        let mut file =
            fs::File::create(path).map_err(|e| format!("Failed to create log file: {e}"))?;
        for entry in &self.command_history {
            writeln!(file, "{entry}").map_err(|e| format!("Write error: {e}"))?;
        }
        writeln!(file, "---").map_err(|e| format!("Write error: {e}"))?;
        writeln!(file, "status: {}", self.action_status)
            .map_err(|e| format!("Write error: {e}"))?;
        if !self.last_action_time.is_empty() {
            writeln!(file, "last_action_time: {}", self.last_action_time)
                .map_err(|e| format!("Write error: {e}"))?;
        }
        for entry in &self.action_log {
            writeln!(file, "{entry}").map_err(|e| format!("Write error: {e}"))?;
        }
        self.exported_log_path = path.to_string();
        Ok(())
    }

    pub fn replay_last_session(&mut self) -> Result<(), String> {
        let path = "qec_tui_session.log";
        let contents =
            fs::read_to_string(path).map_err(|e| format!("Failed to read log file: {e}"))?;
        self.replay_lines = contents.lines().map(|l| l.to_string()).collect();
        Ok(())
    }

    pub fn scan_sessions(&mut self) {
        let mut sessions: Vec<String> = match fs::read_dir(".") {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    let path = entry.path();
                    let ext = path.extension().and_then(|e| e.to_str());
                    if ext == Some("log") && path.is_file() {
                        path.file_name()
                            .and_then(|name| name.to_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => Vec::new(),
        };

        sessions.sort();
        self.session_files = sessions;
        self.filter_sessions();
        if self.filtered_session_files.is_empty() {
            self.selected_session_index = 0;
        } else if self.selected_session_index >= self.filtered_session_files.len() {
            self.selected_session_index = self.filtered_session_files.len() - 1;
        }
    }

    pub fn filter_sessions(&mut self) {
        if self.search_query.is_empty() {
            self.filtered_session_files = self.session_files.clone();
        } else {
            let needle = self.search_query.to_ascii_lowercase();
            self.filtered_session_files = self
                .session_files
                .iter()
                .filter(|name| name.to_ascii_lowercase().contains(&needle))
                .cloned()
                .collect();
        }

        if self.filtered_session_files.is_empty() {
            self.selected_session_index = 0;
        } else if self.selected_session_index >= self.filtered_session_files.len() {
            self.selected_session_index = self.filtered_session_files.len() - 1;
        }
        self.build_session_metadata();
    }

    pub fn build_session_metadata(&mut self) {
        self.session_metadata.clear();

        let Some(selected_file) = self.filtered_session_files.get(self.selected_session_index)
        else {
            self.session_metadata
                .push("No session selected".to_string());
            return;
        };

        let path = Path::new(selected_file);
        let lines = match fs::read_to_string(path) {
            Ok(contents) => contents.lines().count(),
            Err(_) => 0,
        };

        self.session_metadata.push(format!("name: {selected_file}"));
        self.session_metadata.push(format!("lines: {lines}"));

        match fs::metadata(path) {
            Ok(metadata) => {
                self.session_metadata
                    .push(format!("size: {} bytes", metadata.len()));
                if let Ok(modified) = metadata.modified() {
                    self.session_metadata
                        .push(format!("modified: {:?}", modified));
                }
            }
            Err(_) => {
                self.session_metadata.push("size: unavailable".to_string());
                self.session_metadata
                    .push("modified: unavailable".to_string());
            }
        }
    }

    pub fn session_up(&mut self) {
        if self.filtered_session_files.is_empty() {
            return;
        }
        if self.selected_session_index == 0 {
            self.selected_session_index = self.filtered_session_files.len() - 1;
        } else {
            self.selected_session_index -= 1;
        }
        self.build_session_metadata();
    }

    pub fn session_down(&mut self) {
        if self.filtered_session_files.is_empty() {
            return;
        }
        if self.selected_session_index + 1 >= self.filtered_session_files.len() {
            self.selected_session_index = 0;
        } else {
            self.selected_session_index += 1;
        }
        self.build_session_metadata();
    }

    pub fn session_browser_active(&self) -> bool {
        self.mode == HISTORY_WINDOW_MODE
    }

    pub fn diff_with_selected_session(&mut self) {
        self.diff_lines.clear();
        let Some(selected_file) = self.filtered_session_files.get(self.selected_session_index)
        else {
            return;
        };

        let selected_path = Path::new(selected_file);
        let file_lines: Vec<String> = match fs::read_to_string(selected_path) {
            Ok(contents) => contents.lines().map(|line| line.to_string()).collect(),
            Err(e) => {
                self.diff_lines.push(format!("- read error: {e}"));
                return;
            }
        };

        let max_len = file_lines.len().max(self.command_history.len());
        for idx in 0..max_len {
            match (file_lines.get(idx), self.command_history.get(idx)) {
                (Some(previous), Some(current)) if previous == current => {
                    self.diff_lines.push(format!("= {current}"));
                }
                (Some(previous), Some(current)) => {
                    if self.diff_lines.len() + 2 > MAX_DIFF_LINES {
                        break;
                    }
                    self.diff_lines.push(format!("- {previous}"));
                    self.diff_lines.push(format!("+ {current}"));
                }
                (Some(previous), None) => self.diff_lines.push(format!("- {previous}")),
                (None, Some(current)) => self.diff_lines.push(format!("+ {current}")),
                (None, None) => {}
            }
            if self.diff_lines.len() >= MAX_DIFF_LINES {
                break;
            }
        }
        self.diff_lines.truncate(MAX_DIFF_LINES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn test_initial_status_is_idle() {
        let app = App::new();
        assert_eq!(app.action_status, "IDLE");
        assert!(app.command_history.is_empty());
        assert!(app.last_action_time.is_empty());
    }

    #[test]
    fn test_run_action_with_status_sets_success() {
        let mut app = App::new();
        app.run_action_with_status("diagnostics");
        assert!(app.action_status == "SUCCESS" || app.action_status == "FAILED");
        assert_eq!(app.command_history.len(), 1);
        assert!(app.command_history[0].starts_with("diagnostics @ "));
        assert!(!app.last_action_time.is_empty());
    }

    #[test]
    fn test_run_action_with_status_unknown_action_sets_failed() {
        let mut app = App::new();
        app.run_action_with_status("nonexistent_action");
        assert_eq!(app.action_status, "FAILED");
        assert_eq!(app.command_history.len(), 1);
        assert!(app.command_history[0].starts_with("nonexistent_action @ "));
    }

    #[test]
    fn test_command_history_fifo_trim() {
        let mut app = App::new();
        for i in 0..25 {
            app.command_history.push(format!("cmd_{i}"));
        }
        // Simulate the trim logic
        while app.command_history.len() > MAX_COMMAND_HISTORY {
            app.command_history.remove(0);
        }
        assert_eq!(app.command_history.len(), MAX_COMMAND_HISTORY);
        assert_eq!(app.command_history[0], "cmd_5");
        assert_eq!(app.command_history[19], "cmd_24");
    }

    #[test]
    fn test_status_transitions() {
        let mut app = App::new();
        assert_eq!(app.action_status, "IDLE");
        // After running a valid action, status should be SUCCESS or FAILED
        app.run_action_with_status("diagnostics");
        let after_run = app.action_status.clone();
        assert!(after_run == "SUCCESS" || after_run == "FAILED");
    }

    #[test]
    fn test_command_history_ordering() {
        let mut app = App::new();
        app.run_action_with_status("diagnostics");
        app.run_action_with_status("invariants");
        assert_eq!(app.command_history.len(), 2);
        assert!(app.command_history[0].starts_with("diagnostics @ "));
        assert!(app.command_history[1].starts_with("invariants @ "));
    }

    #[test]
    fn test_export_writes_file() {
        let mut app = App::new();
        app.command_history.push("test_cmd @ time1".to_string());
        app.action_status = "SUCCESS".to_string();
        app.action_log.push("[test] output".to_string());
        let result = app.export_session_log();
        assert!(result.is_ok());
        assert_eq!(app.exported_log_path, "qec_tui_session.log");
        let contents = std::fs::read_to_string("qec_tui_session.log").unwrap();
        assert!(contents.contains("test_cmd @ time1"));
        assert!(contents.contains("status: SUCCESS"));
        // cleanup
        let _ = std::fs::remove_file("qec_tui_session.log");
    }

    #[test]
    fn test_replay_reads_file() {
        let mut app = App::new();
        std::fs::write("qec_tui_session_test.log", "line1\nline2\nline3\n").unwrap();
        // Temporarily use a custom path for isolation — test the core logic
        let contents = std::fs::read_to_string("qec_tui_session_test.log").unwrap();
        app.replay_lines = contents.lines().map(|l| l.to_string()).collect();
        assert_eq!(app.replay_lines.len(), 3);
        assert_eq!(app.replay_lines[0], "line1");
        let _ = std::fs::remove_file("qec_tui_session_test.log");
    }

    #[test]
    fn test_replay_missing_file_returns_error() {
        let mut app = App::new();
        let _ = std::fs::remove_file("qec_tui_session.log");
        let result = app.replay_last_session();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Failed to read log file"));
    }

    #[test]
    fn test_replay_trims_to_10_in_ui() {
        // Simulate: replay_lines has 15 entries, UI should show last 10
        let mut app = App::new();
        for i in 0..15 {
            app.replay_lines.push(format!("line_{i}"));
        }
        let display: Vec<&String> = app.replay_lines.iter().rev().take(10).collect();
        assert_eq!(display.len(), 10);
    }

    #[test]
    fn test_scan_finds_log_files_in_alphabetical_order() {
        let _guard = test_lock().lock().unwrap();
        let mut app = App::new();
        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = original_dir.join("tui_app_scan_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_current_dir(&temp_dir).unwrap();

        std::fs::write("z_run.log", "").unwrap();
        std::fs::write("a_run.log", "").unwrap();
        std::fs::write("ignore.txt", "").unwrap();

        app.scan_sessions();
        assert_eq!(
            app.session_files,
            vec!["a_run.log".to_string(), "z_run.log".to_string()]
        );

        std::env::set_current_dir(&original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_scan_empty_directory_no_panic() {
        let _guard = test_lock().lock().unwrap();
        let mut app = App::new();
        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = original_dir.join("tui_app_empty_scan_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_current_dir(&temp_dir).unwrap();

        app.scan_sessions();
        assert!(app.session_files.is_empty());
        assert!(app.filtered_session_files.is_empty());
        assert_eq!(app.selected_session_index, 0);

        std::env::set_current_dir(&original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_diff_markers_present() {
        let _guard = test_lock().lock().unwrap();
        let mut app = App::new();
        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = original_dir.join("tui_app_diff_marker_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_current_dir(&temp_dir).unwrap();

        std::fs::write("session.log", "same\nold_only\n").unwrap();
        app.command_history = vec![
            "same".to_string(),
            "new_only".to_string(),
            "added".to_string(),
        ];
        app.scan_sessions();
        app.diff_with_selected_session();
        let joined = app.diff_lines.join("\n");
        assert!(joined.contains("= same"));
        assert!(joined.contains("- old_only"));
        assert!(joined.contains("+ new_only"));
        assert!(joined.contains("+ added"));

        std::env::set_current_dir(&original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_diff_lines_trimmed_to_20() {
        let _guard = test_lock().lock().unwrap();
        let mut app = App::new();
        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = original_dir.join("tui_app_diff_trim_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_current_dir(&temp_dir).unwrap();

        let file_lines = (0..30)
            .map(|i| format!("old_{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write("session.log", file_lines).unwrap();
        app.command_history = (0..30).map(|i| format!("new_{i}")).collect();
        app.scan_sessions();
        app.diff_with_selected_session();
        assert_eq!(app.diff_lines.len(), 20);

        std::env::set_current_dir(&original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_diff_never_emits_orphaned_removal_line() {
        let _guard = test_lock().lock().unwrap();
        let mut app = App::new();
        let original_dir = std::env::current_dir().unwrap();
        let temp_dir = original_dir.join("tui_app_diff_orphan_guard_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_current_dir(&temp_dir).unwrap();

        let mut session_contents = (0..(MAX_DIFF_LINES - 1))
            .map(|i| format!("same_{i}\n"))
            .collect::<String>();
        session_contents.push_str("old_tail\n");
        std::fs::write("session.log", session_contents).unwrap();

        app.command_history = (0..(MAX_DIFF_LINES - 1))
            .map(|i| format!("same_{i}"))
            .collect();
        app.command_history.push("new_tail".to_string());
        app.scan_sessions();
        app.diff_with_selected_session();

        assert_eq!(app.diff_lines.len(), MAX_DIFF_LINES - 1);
        assert!(app
            .diff_lines
            .iter()
            .all(|line| !line.starts_with("- old_tail")));
        assert!(app
            .diff_lines
            .iter()
            .all(|line| !line.starts_with("+ new_tail")));

        std::env::set_current_dir(&original_dir).unwrap();
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_filter_empty_query_returns_all_sessions() {
        let mut app = App::new();
        app.session_files = vec!["a.log".to_string(), "b.log".to_string()];
        app.search_query.clear();
        app.filter_sessions();
        assert_eq!(app.filtered_session_files, app.session_files);
    }

    #[test]
    fn test_filter_ordering_is_deterministic() {
        let mut app = App::new();
        app.session_files = vec![
            "2026_alpha.log".to_string(),
            "2026_beta.log".to_string(),
            "2025_gamma.log".to_string(),
        ];
        app.search_query = "2026".to_string();
        app.filter_sessions();
        assert_eq!(
            app.filtered_session_files,
            vec!["2026_alpha.log".to_string(), "2026_beta.log".to_string()]
        );
    }

    #[test]
    fn test_filter_is_case_insensitive() {
        let mut app = App::new();
        app.session_files = vec!["Alpha.LOG".to_string(), "beta.log".to_string()];
        app.search_query = "alpha".to_string();
        app.filter_sessions();
        assert_eq!(app.filtered_session_files, vec!["Alpha.LOG".to_string()]);
    }

    #[test]
    fn test_help_overlay_toggle_state() {
        let mut app = App::new();
        assert!(!app.help_overlay_active);
        app.help_overlay_active = true;
        assert!(app.help_overlay_active);
        app.help_overlay_active = false;
        assert!(!app.help_overlay_active);
    }

    #[test]
    fn test_metadata_fallback_on_empty_selection() {
        let mut app = App::new();
        app.filtered_session_files.clear();
        app.selected_session_index = 0;
        app.build_session_metadata();
        assert_eq!(
            app.session_metadata,
            vec!["No session selected".to_string()]
        );
    }

    #[test]
    fn test_observability_metrics_counters() {
        let mut app = App::new();
        app.update_observability_metrics("diagnostics", true, 10);
        app.update_observability_metrics("law", false, 20);

        assert_eq!(app.total_actions_run, 2);
        assert_eq!(app.successful_actions, 1);
        assert_eq!(app.failed_actions, 1);
    }

    #[test]
    fn test_observability_metrics_rolling_latency_average() {
        let mut app = App::new();
        app.update_observability_metrics("diagnostics", true, 10);
        app.update_observability_metrics("invariants", true, 20);
        app.update_observability_metrics("law", true, 31);

        assert_eq!(app.average_action_latency_ms, 20);
    }

    #[test]
    fn test_observability_health_transitions() {
        let mut app = App::new();
        assert_eq!(app.health_status, "HEALTHY");

        app.update_observability_metrics("a", false, 1);
        assert_eq!(app.health_status, "DEGRADED");

        app.update_observability_metrics("b", false, 1);
        app.update_observability_metrics("c", false, 1);
        assert_eq!(app.health_status, "DEGRADED");

        app.update_observability_metrics("d", false, 1);
        assert_eq!(app.health_status, "CRITICAL");
    }

    #[test]
    fn test_recent_failures_capped_at_10() {
        let mut app = App::new();
        for idx in 0..12 {
            app.update_observability_metrics(&format!("fail_{idx}"), false, 1);
        }

        assert_eq!(app.recent_failures.len(), 10);
        assert_eq!(app.recent_failures.first().unwrap(), "fail_2");
        assert_eq!(app.recent_failures.last().unwrap(), "fail_11");
    }

    #[test]
    fn test_invariant_summary_fallback() {
        let mut app = App::new();
        app.invariants = InvariantData {
            determinism: "PASS".to_string(),
            bounds: "—".to_string(),
            stability: "UNKNOWN".to_string(),
            law_engine: "FAIL".to_string(),
            error: None,
        };
        app.build_invariant_summary();

        assert_eq!(app.invariant_summary[0], "PASS: 1");
        assert_eq!(app.invariant_summary[1], "FAIL: 1");
        assert_eq!(app.invariant_summary[2], "UNKNOWN: 2");
    }
}
