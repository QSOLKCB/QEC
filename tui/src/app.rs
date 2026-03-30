use crate::commands::dispatch_mode;

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
];

pub struct App {
    pub nav_items: &'static [&'static str],
    pub selected_index: usize,
    pub mode: &'static str,
}

impl App {
    pub fn new() -> Self {
        Self {
            nav_items: NAV_ITEMS,
            selected_index: 0,
            mode: "Diagnostics",
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
}
