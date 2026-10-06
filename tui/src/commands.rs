use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Stub dispatcher — returns mode name string for a given nav index.
/// No engine calls. Display only.
pub fn dispatch_mode(index: usize) -> &'static str {
    match index {
        0 => "Diagnostics",
        1 => "Control Flow",
        2 => "Memory",
        3 => "Adaptive",
        4 => "Regime Jump",
        5 => "Self-Healing",
        6 => "History Window",
        7 => "Invariants",
        8 => "Law Engine",
        9 => "Phase Dynamics",
        10 => "Actions",
        _ => "Unknown",
    }
}

/// Demo fixtures are explicit and never stand in for a failed live request.
pub fn demo_enabled() -> bool {
    std::env::var("QEC_TUI_DEMO").as_deref() == Ok("1")
}

fn python_candidates(explicit: Option<OsString>, venv: Option<OsString>) -> Vec<OsString> {
    if let Some(python) = explicit {
        // An explicit configuration is authoritative, even when invalid.
        return vec![python];
    }
    if let Some(venv) = venv.filter(|value| !value.is_empty()) {
        let relative = if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" };
        return vec![PathBuf::from(venv).join(relative).into_os_string()];
    }
    vec![OsString::from("python3"), OsString::from("python")]
}

fn invoke_python(module: &str, candidates: &[OsString]) -> Result<String, String> {
    for python in candidates {
        let result = Command::new(python)
            .args(["-m", module])
            .stdin(Stdio::null())
            .output();
        match result {
            Err(error) if error.kind() == ErrorKind::NotFound && candidates.len() > 1 => {
                continue;
            }
            Err(error) => {
                return Err(format!(
                    "Cannot invoke Python {:?}: {error}. Set QEC_PYTHON to your QEC virtual environment's Python executable.",
                    python
                ));
            }
            Ok(output) if !output.status.success() => {
                return Err(format!(
                    "Python {:?} -m {module} failed ({}): {}. Activate the QEC environment and check that this CLI adapter is installed.",
                    python, output.status, String::from_utf8_lossy(&output.stderr).trim()
                ));
            }
            Ok(output) => {
                let stdout = String::from_utf8(output.stdout)
                    .map_err(|error| format!("{module} returned invalid UTF-8: {error}"))?;
                if stdout.trim().is_empty() {
                    return Err(format!("{module} returned empty output"));
                }
                return Ok(stdout.trim().to_string());
            }
        }
    }
    Err("No Python interpreter found (tried python3, python). Install Python 3 or set QEC_PYTHON to your QEC virtual environment's Python executable.".to_string())
}

fn run_module(module: &str) -> Result<String, String> {
    invoke_python(module, &python_candidates(std::env::var_os("QEC_PYTHON"), std::env::var_os("VIRTUAL_ENV")))
}

fn panel_data(module: &str, fixture: &str) -> Result<String, String> {
    if demo_enabled() {
        Ok(fixture.to_string())
    } else {
        run_module(module)
    }
}

pub fn fetch_engine_diagnostics() -> Result<String, String> {
    panel_data("qec.cli.diagnostics", r#"{"collapse_score":0.12,"trend_state":"stable","adaptive_damping":0.8,"healing_mode":"hold","history_behavior":"stable_window"}"#)
}

pub fn fetch_history_timeline() -> Result<String, String> {
    panel_data("qec.cli.history", r#"{"timeline":["stable","rising","oscillatory","locked"]}"#)
}

pub fn fetch_invariant_status() -> Result<String, String> {
    panel_data("qec.cli.invariants", r#"{"determinism":"PASS","bounds":"PASS","stability":"PASS","law_engine":"PASS"}"#)
}

pub fn fetch_phase_diagnostics() -> Result<String, String> {
    panel_data("qec.cli.phase_diagnostics", r#"{"attractor_state":"fixed_point","attractor_cycle_length":1,"phase_transition_index":0.0,"attractor_entry_cycle":0,"transition_sharpness_score":1.0,"attractor_confidence_score":1.0,"detected_cycle_period":1,"cycle_spectrum_class":"mono"}"#)
}

fn collect_refresh(mut run: impl FnMut(&str) -> Result<String, String>) -> Result<String, String> {
    let mut combined = String::new();
    let mut failed = false;
    for action in ["diagnostics", "invariants", "law", "phase_diagnostics"] {
        match run(action) {
            Ok(output) => combined.push_str(&format!("[{action}] {}\n", output.trim())),
            Err(error) => {
                failed = true;
                combined.push_str(&format!("[{action}] ERROR: {error}\n"));
            }
        }
    }
    if failed { Err(combined) } else { Ok(combined) }
}

pub fn execute_action(action: &str) -> Result<String, String> {
    let module = match action {
        "diagnostics" => "qec.cli.diagnostics",
        "invariants" => "qec.cli.invariants",
        "law" => "qec.cli.law_engine",
        "phase_diagnostics" => "qec.cli.phase_diagnostics",
        "refresh" => return collect_refresh(execute_action),
        other => return Err(format!("Unknown action: {other}")),
    };
    if demo_enabled() {
        return Ok(format!("DEMO: simulated {action}; no Python engine was invoked"));
    }
    run_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_python_is_authoritative() {
        assert_eq!(python_candidates(Some("/missing/custom python".into()), Some("/venv".into())), vec![OsString::from("/missing/custom python")]);
    }

    #[test]
    fn active_virtual_environment_is_preferred() {
        let candidates = python_candidates(None, Some("/my venv".into()));
        let relative = if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" };
        assert_eq!(candidates, vec![PathBuf::from("/my venv").join(relative).into_os_string()]);
        assert_eq!(python_candidates(None, None), vec![OsString::from("python3"), OsString::from("python")]);
    }

    #[test]
    fn missing_interpreter_is_an_error() {
        let error = invoke_python("qec.cli.diagnostics", &["/qec-missing-interpreter".into()]).unwrap_err();
        assert!(error.contains("QEC_PYTHON"));
        assert!(error.contains("qec-missing-interpreter"));
    }

    #[test]
    fn refresh_propagates_partial_failure_and_retains_all_outputs() {
        let error = collect_refresh(|action| {
            if action == "invariants" { Err("missing adapter".into()) } else { Ok("real result".into()) }
        }).unwrap_err();
        assert!(error.contains("[invariants] ERROR: missing adapter"));
        assert!(error.contains("[phase_diagnostics] real result"));
        assert!(collect_refresh(|_| Ok("real result".into())).is_ok());
    }
}
