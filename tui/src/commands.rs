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
