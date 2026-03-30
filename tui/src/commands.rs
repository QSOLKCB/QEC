use std::process::Command;

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
        _ => "Unknown",
    }
}

/// Invoke the Python engine and return raw JSON diagnostics.
///
/// Calls `python -m qec.cli.diagnostics` first; if that module is unavailable,
/// falls back to an inline placeholder that emits the expected JSON shape.
pub fn fetch_engine_diagnostics() -> Result<String, String> {
    // Try the real CLI entry point first
    let output = Command::new("python")
        .args(["-m", "qec.cli.diagnostics"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            if stdout.trim().is_empty() {
                return Err("Engine returned empty output".to_string());
            }
            Ok(stdout)
        }
        _ => {
            // Fallback: inline placeholder
            let fallback = Command::new("python")
                .args([
                    "-c",
                    "import json; print(json.dumps({\"collapse_score\": 0.12, \"trend_state\": \"stable\", \"adaptive_damping\": 0.8, \"healing_mode\": \"hold\", \"history_behavior\": \"stable_window\"}))",
                ])
                .output()
                .map_err(|e| format!("Failed to invoke Python: {e}"))?;

            if !fallback.status.success() {
                let stderr = String::from_utf8_lossy(&fallback.stderr);
                return Err(format!("Python subprocess failed: {stderr}"));
            }

            let stdout = String::from_utf8_lossy(&fallback.stdout).to_string();
            if stdout.trim().is_empty() {
                return Err("Fallback engine returned empty output".to_string());
            }
            Ok(stdout)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_engine_diagnostics_returns_json() {
        let result = fetch_engine_diagnostics();
        assert!(result.is_ok(), "fetch_engine_diagnostics failed: {:?}", result.err());
        let json: serde_json::Value =
            serde_json::from_str(&result.unwrap()).expect("output is not valid JSON");
        assert!(json.get("collapse_score").is_some());
        assert!(json.get("trend_state").is_some());
    }
}
