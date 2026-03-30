mod app;
mod commands;
mod ui;

use std::io;
use std::time::Instant;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::{CrosstermBackend, Terminal};

use app::App;

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let mut app = App::new();

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if app.search_overlay_active {
                match key.code {
                    KeyCode::Esc => {
                        app.search_overlay_active = false;
                        app.help_overlay_active = false;
                    }
                    KeyCode::Backspace => {
                        app.search_query.pop();
                    }
                    KeyCode::Enter => app.filter_sessions(),
                    KeyCode::Char(c) => app.search_query.push(c),
                    _ => {}
                }
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => break,
                KeyCode::Esc => {
                    app.search_overlay_active = false;
                    app.help_overlay_active = false;
                }
                KeyCode::Char('?') => app.help_overlay_active = !app.help_overlay_active,
                KeyCode::Char('/') => {
                    app.search_overlay_active = true;
                    app.help_overlay_active = false;
                }
                _ if app.help_overlay_active => {}
                KeyCode::Up => {
                    if app.session_browser_active() {
                        app.session_up();
                    } else {
                        app.nav_up();
                    }
                }
                KeyCode::Down => {
                    if app.session_browser_active() {
                        app.session_down();
                    } else {
                        app.nav_down();
                    }
                }
                KeyCode::Enter => app.select_mode(),
                _ if app.mode == "Actions" => match key.code {
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        let started = Instant::now();
                        app.run_action_with_status("diagnostics");
                        app.update_observability_metrics(
                            "diagnostics",
                            app.action_status == "SUCCESS",
                            started.elapsed().as_millis(),
                        );
                    }
                    KeyCode::Char('i') | KeyCode::Char('I') => {
                        let started = Instant::now();
                        app.run_action_with_status("invariants");
                        app.update_observability_metrics(
                            "invariants",
                            app.action_status == "SUCCESS",
                            started.elapsed().as_millis(),
                        );
                    }
                    KeyCode::Char('l') | KeyCode::Char('L') => {
                        let started = Instant::now();
                        app.run_action_with_status("law");
                        app.update_observability_metrics(
                            "law",
                            app.action_status == "SUCCESS",
                            started.elapsed().as_millis(),
                        );
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        let started = Instant::now();
                        app.run_action_with_status("refresh");
                        app.update_observability_metrics(
                            "refresh",
                            app.action_status == "SUCCESS",
                            started.elapsed().as_millis(),
                        );
                        app.refresh_all();
                    }
                    KeyCode::Char('x') | KeyCode::Char('X') => app.jump_to(9),
                    _ => {}
                },
                // Direct shortcuts (non-Actions mode)
                KeyCode::Char('d') | KeyCode::Char('D') => app.jump_to(0),
                KeyCode::Char('c') | KeyCode::Char('C') => app.jump_to(1),
                KeyCode::Char('m') | KeyCode::Char('M') => app.jump_to(2),
                KeyCode::Char('a') | KeyCode::Char('A') => app.jump_to(3),
                KeyCode::Char('r') | KeyCode::Char('R') => app.refresh_all(),
                KeyCode::Char('h') | KeyCode::Char('H') => app.jump_to(5),
                // 'w' for history Window to avoid conflict
                KeyCode::Char('w') | KeyCode::Char('W') => app.jump_to(6),
                KeyCode::Char('i') | KeyCode::Char('I') => app.jump_to(7),
                KeyCode::Char('l') | KeyCode::Char('L') => app.jump_to(8),
                KeyCode::Char('x') | KeyCode::Char('X') => app.jump_to(9),
                KeyCode::Char('e') | KeyCode::Char('E') => match app.export_session_log() {
                    Ok(()) => app.action_status = "EXPORTED".to_string(),
                    Err(e) => {
                        app.action_log.push(format!("[export] ERROR: {e}"));
                        app.action_status = "FAILED".to_string();
                    }
                },
                KeyCode::Char('p') | KeyCode::Char('P') => match app.replay_last_session() {
                    Ok(()) => app.action_status = "REPLAY LOADED".to_string(),
                    Err(e) => {
                        app.action_log.push(format!("[replay] ERROR: {e}"));
                        app.action_status = "FAILED".to_string();
                    }
                },
                KeyCode::Char('1') => app.set_operator_view(0),
                KeyCode::Char('2') => app.set_operator_view(1),
                KeyCode::Char('3') => app.set_operator_view(2),
                KeyCode::Char('t') | KeyCode::Char('T') => app.cycle_alert_threshold_profile(),
                KeyCode::Char('s') | KeyCode::Char('S') => app.scan_sessions(),
                KeyCode::Char('v') | KeyCode::Char('V') => app.diff_with_selected_session(),
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
