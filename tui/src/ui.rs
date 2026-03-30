use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App) {
    // Main vertical split: body + footer
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(f.size());

    // Body: left nav | center workspace | right status
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(22),
            Constraint::Min(40),
            Constraint::Length(28),
        ])
        .split(outer[0]);

    draw_nav(f, app, body[0]);
    draw_workspace(f, app, body[1]);
    draw_status(f, app, body[2]);
    draw_footer(f, outer[1]);
}

fn draw_nav(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .nav_items
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let prefix = if i == app.selected_index { ">> " } else { "   " };
            let style = if i == app.selected_index {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(Span::styled(
                format!("{prefix}{name}"),
                style,
            )))
        })
        .collect();

    let nav = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Navigation "),
    );
    f.render_widget(nav, area);
}

fn draw_workspace(f: &mut Frame, app: &App, area: Rect) {
    let ws_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(8),
        ])
        .split(area);

    let content = workspace_content(app);
    let paragraph = Paragraph::new(content).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" {} ", app.mode)),
    );
    f.render_widget(paragraph, ws_layout[0]);

    draw_command_history(f, app, ws_layout[1]);
    draw_sessions_pane(f, app, ws_layout[2]);
    draw_diff_pane(f, app, ws_layout[3]);
}

fn draw_command_history(f: &mut Frame, app: &App, area: Rect) {
    let status_color = match app.action_status.as_str() {
        "RUNNING" => Color::Yellow,
        "SUCCESS" => Color::Green,
        "FAILED" => Color::Red,
        _ => Color::White,
    };

    let mut lines: Vec<Line<'static>> = vec![
        Line::from(Span::styled(
            format!("  STATUS: {}", app.action_status),
            Style::default().fg(status_color).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];

    if app.command_history.is_empty() {
        lines.push(Line::from("  (no commands run yet)"));
    } else {
        for entry in &app.command_history {
            lines.push(Line::from(format!("  {entry}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Recent Commands "),
    );
    f.render_widget(panel, area);
}

fn workspace_content(app: &App) -> Vec<Line<'static>> {
    let mode = app.mode;
    match mode {
        "Diagnostics" => return diagnostics_content(app),
        "Control Flow" => vec![
            Line::from(""),
            Line::from("  control_mode:     auto"),
            Line::from("  damping_target:   0.80"),
            Line::from("  override:         none"),
        ],
        "Memory" => vec![
            Line::from(""),
            Line::from("  memory_slots:     16"),
            Line::from("  active_entries:   3"),
            Line::from("  bounded:          true"),
        ],
        "Adaptive" => vec![
            Line::from(""),
            Line::from("  strategy:         conservative"),
            Line::from("  score_bias:       0.00"),
            Line::from("  feedback_loop:    idle"),
        ],
        "Regime Jump" => vec![
            Line::from(""),
            Line::from("  current_regime:   stable"),
            Line::from("  jump_detected:    false"),
            Line::from("  transition_prob:  0.02"),
        ],
        "Self-Healing" => vec![
            Line::from(""),
            Line::from("  healing_state:    hold"),
            Line::from("  repairs_queued:   0"),
            Line::from("  last_action:      none"),
        ],
        "History Window" => return history_content(app),
        "Invariants" => return invariants_content(app),
        "Law Engine" => vec![
            Line::from(""),
            Line::from("  laws_loaded:      12"),
            Line::from("  violations:       0"),
            Line::from("  enforcement:      active"),
        ],
        "Actions" => return actions_content(app),
        _ => vec![Line::from("  (no data)")],
    }
}

fn status_color(value: &str) -> Color {
    if value == "FAIL" { Color::Red } else { Color::Green }
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let inv = &app.invariants;
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  ENGINE:       READY",
            Style::default().fg(Color::Green),
        )),
        Line::from(Span::styled(
            format!("  DETERMINISM:  {}", inv.determinism),
            Style::default().fg(status_color(&inv.determinism)),
        )),
        Line::from(Span::styled(
            format!("  BOUNDS:       {}", inv.bounds),
            Style::default().fg(status_color(&inv.bounds)),
        )),
        Line::from(Span::styled(
            format!("  STABILITY:    {}", inv.stability),
            Style::default().fg(status_color(&inv.stability)),
        )),
        Line::from(Span::styled(
            format!("  LAW ENGINE:   {}", inv.law_engine),
            Style::default().fg(status_color(&inv.law_engine)),
        )),
    ];

    let status = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Status "),
    );
    f.render_widget(status, area);
}

fn diagnostics_content(app: &App) -> Vec<Line<'static>> {
    let d = &app.diagnostics;
    if let Some(ref err) = d.error {
        return vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {err}"),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  Press [R] to retry"),
        ];
    }
    vec![
        Line::from(""),
        Line::from(format!("  collapse_score:         {}", d.collapse_score)),
        Line::from(format!("  trend_state:            {}", d.trend_state)),
        Line::from(format!("  adaptive_damping:       {}", d.adaptive_damping)),
        Line::from(format!("  healing_mode:           {}", d.healing_mode)),
        Line::from(format!("  history_behavior:       {}", d.history_behavior)),
    ]
}

fn history_content(app: &App) -> Vec<Line<'static>> {
    let h = &app.history;
    if let Some(ref err) = h.error {
        return vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {err}"),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  Press [R] to retry"),
        ];
    }
    let timeline_str = h.timeline.join(" → ");
    vec![
        Line::from(""),
        Line::from(format!("  Timeline: {timeline_str}")),
    ]
}

fn invariants_content(app: &App) -> Vec<Line<'static>> {
    let inv = &app.invariants;
    if let Some(ref err) = inv.error {
        return vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {err}"),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  Press [R] to retry"),
        ];
    }
    vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  DETERMINISM   {}", inv.determinism),
            Style::default().fg(status_color(&inv.determinism)),
        )),
        Line::from(Span::styled(
            format!("  BOUNDS        {}", inv.bounds),
            Style::default().fg(status_color(&inv.bounds)),
        )),
        Line::from(Span::styled(
            format!("  STABILITY     {}", inv.stability),
            Style::default().fg(status_color(&inv.stability)),
        )),
        Line::from(Span::styled(
            format!("  LAW ENGINE    {}", inv.law_engine),
            Style::default().fg(status_color(&inv.law_engine)),
        )),
    ]
}

fn actions_content(app: &App) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Available Actions:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  [D] Run Diagnostics"),
        Line::from("  [I] Run Invariants"),
        Line::from("  [L] Run Law Engine"),
        Line::from("  [R] Refresh All"),
        Line::from(""),
        Line::from(Span::styled(
            "  Recent Action Output:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
    ];
    if app.action_log.is_empty() {
        lines.push(Line::from("  (no actions run yet)"));
    } else {
        for entry in &app.action_log {
            lines.push(Line::from(format!("  {entry}")));
        }
    }
    lines
}

fn draw_sessions_pane(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.session_files.is_empty() {
        lines.push(Line::from("  No sessions scanned"));
    } else {
        for (index, session_file) in app.session_files.iter().enumerate() {
            let prefix = if index == app.selected_session_index { ">" } else { " " };
            lines.push(Line::from(format!("{prefix} {session_file}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Saved Sessions "),
    );
    f.render_widget(panel, area);
}

fn draw_diff_pane(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.diff_lines.is_empty() {
        lines.push(Line::from("  No diff loaded"));
    } else {
        for line in &app.diff_lines {
            lines.push(Line::from(format!("  {line}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Session Diff "),
    );
    f.render_widget(panel, area);
}

fn draw_footer(f: &mut Frame, area: Rect) {
    let legend = Line::from(vec![
        Span::styled(" [D]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Diagnostics  "),
        Span::styled("[C]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Control  "),
        Span::styled("[M]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Memory  "),
        Span::styled("[A]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Adaptive  "),
        Span::styled("[R]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Refresh All  "),
        Span::styled("[H]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Healing  "),
        Span::styled("[I]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Invariants  "),
        Span::styled("[L]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Law  "),
        Span::styled("[X]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Actions  "),
        Span::styled("[E]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Export  "),
        Span::styled("[P]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Replay  "),
        Span::styled("[S]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Scan  "),
        Span::styled("[V]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Diff  "),
        Span::styled("[Q]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::raw(" Quit"),
    ]);

    let footer = Paragraph::new(legend).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Hotkeys "),
    );
    f.render_widget(footer, area);
}
