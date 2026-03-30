mod app;
mod commands;
mod ui;

use std::io;

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
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => break,
                KeyCode::Up => app.nav_up(),
                KeyCode::Down => app.nav_down(),
                KeyCode::Enter => app.select_mode(),
                // Direct shortcuts
                KeyCode::Char('d') | KeyCode::Char('D') => app.jump_to(0),
                KeyCode::Char('c') | KeyCode::Char('C') => app.jump_to(1),
                KeyCode::Char('m') | KeyCode::Char('M') => app.jump_to(2),
                KeyCode::Char('a') | KeyCode::Char('A') => app.jump_to(3),
                KeyCode::Char('r') | KeyCode::Char('R') => app.jump_to(4),
                KeyCode::Char('h') | KeyCode::Char('H') => app.jump_to(5),
                // 'w' for history Window to avoid conflict
                KeyCode::Char('w') | KeyCode::Char('W') => app.jump_to(6),
                KeyCode::Char('i') | KeyCode::Char('I') => app.jump_to(7),
                KeyCode::Char('l') | KeyCode::Char('L') => app.jump_to(8),
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
