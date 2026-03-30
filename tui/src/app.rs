use serde::Deserialize;

use crate::commands::{dispatch_mode, execute_action, fetch_engine_diagnostics, fetch_history_timeline, fetch_invariant_status};

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
}

impl App {
    pub fn new() -> Self {
        Self {
            nav_items: NAV_ITEMS,
            selected_index: 0,
            mode: "Diagnostics",
            diagnostics: DiagnosticsData::from_engine(),
            history: HistoryData::from_engine(),
            invariants: InvariantData::from_engine(),
            action_log: Vec::new(),
        }
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
}
