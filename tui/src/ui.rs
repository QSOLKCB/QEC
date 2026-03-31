use std::fmt::Display;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
    Frame,
};

use crate::app::{App, HealthStatus, HISTORY_WINDOW_MODE, MAX_INCIDENT_TIMELINE};

pub fn draw(f: &mut Frame, app: &App) {
    // Main vertical split: KPI strip + body + footer
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.size());

    draw_kpi_strip(f, app, outer[0]);

    // Body: left nav | center workspace | right status
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(22),
            Constraint::Min(40),
            Constraint::Length(28),
        ])
        .split(outer[1]);

    draw_nav(f, app, body[0]);
    draw_workspace(f, app, body[1]);
    draw_status(f, app, body[2]);
    if app.search_overlay_active {
        draw_search_overlay(f, app);
    }
    if app.help_overlay_active {
        draw_help_overlay(f);
    }
    draw_footer(f, outer[2]);
}

fn draw_kpi_strip(f: &mut Frame, app: &App, area: Rect) {
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(12),
            Constraint::Length(28),
            Constraint::Length(28),
            Constraint::Min(28),
        ])
        .split(area);

    draw_kpi_card(f, cards[0], "Actions", app.total_actions_run, Color::White);
    draw_ratio_gauge(
        f,
        cards[1],
        "Success Ratio",
        app.success_ratio_percent(),
        Color::Green,
    );
    draw_ratio_gauge(
        f,
        cards[2],
        "Failure Ratio",
        app.failure_ratio_percent(),
        Color::Red,
    );
    draw_ratio_gauge(
        f,
        cards[3],
        "Latency Gauge",
        app.latency_threshold_percent(),
        Color::Cyan,
    );
}

fn draw_kpi_card(f: &mut Frame, area: Rect, title: &str, value: impl Display, color: Color) {
    let line = Line::from(Span::styled(
        format!("  {value}"),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    let panel = Paragraph::new(vec![Line::from(""), line]).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" {title} ")),
    );
    f.render_widget(panel, area);
}

fn draw_nav(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .nav_items
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let prefix = if i == app.selected_index {
                ">> "
            } else {
                "   "
            };
            let style = if i == app.selected_index {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(Span::styled(format!("{prefix}{name}"), style)))
        })
        .collect();

    let nav = List::new(items).block(Block::default().borders(Borders::ALL).title(" Navigation "));
    f.render_widget(nav, area);
}

fn draw_workspace(f: &mut Frame, app: &App, area: Rect) {
    let ws_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(7),
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
    draw_metadata_pane(f, app, ws_layout[3]);
    draw_diff_pane(f, app, ws_layout[4]);
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
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
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
        HISTORY_WINDOW_MODE => return history_content(app),
        "Invariants" => return invariants_content(app),
        "Law Engine" => vec![
            Line::from(""),
            Line::from("  laws_loaded:      12"),
            Line::from("  violations:       0"),
            Line::from("  enforcement:      active"),
        ],
        "Phase Dynamics" => return phase_dynamics_content(app),
        "Actions" => return actions_content(app),
        _ => vec![Line::from("  (no data)")],
    }
}

fn status_color(value: &str) -> Color {
    if value == "FAIL" {
        Color::Red
    } else {
        Color::Green
    }
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let status_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Min(6),
        ])
        .split(area);

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

    let status =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Status "));
    f.render_widget(status, status_layout[0]);

    draw_health_and_view(f, app, status_layout[1]);
    draw_recent_failures(f, app, status_layout[2]);
    draw_invariant_kpis(f, app, status_layout[3]);
    draw_phase_health(f, app, status_layout[4]);
    draw_operator_audit(f, app, status_layout[5]);
    draw_incident_timeline(f, app, status_layout[6]);
}

fn draw_health_and_view(f: &mut Frame, app: &App, area: Rect) {
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Length(4)])
        .split(area);

    let health_color = match app.health_status {
        HealthStatus::Healthy => Color::Green,
        HealthStatus::Degraded => Color::Yellow,
        HealthStatus::Critical => Color::Red,
    };
    let health_panel = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", app.health_status),
            Style::default()
                .fg(health_color)
                .add_modifier(Modifier::BOLD),
        )),
    ])
    .block(Block::default().borders(Borders::ALL).title(" Health "));
    f.render_widget(health_panel, split[0]);

    let view_panel = Paragraph::new(vec![
        Line::from(""),
        Line::from(format!("  Current: {}", app.current_view)),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Operator View "),
    );
    f.render_widget(view_panel, split[1]);
}

fn draw_recent_failures(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.recent_failures.is_empty() {
        lines.push(Line::from("  No recent failures"));
    } else {
        for failure in app.recent_failures.iter().rev() {
            lines.push(Line::from(format!("  {failure}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Recent Failures "),
    );
    f.render_widget(panel, area);
}

fn draw_invariant_kpis(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.invariant_summary.is_empty() {
        lines.push(Line::from("  PASS: 0"));
        lines.push(Line::from("  FAIL: 0"));
        lines.push(Line::from("  UNKNOWN: 0"));
    } else {
        for line in &app.invariant_summary {
            lines.push(Line::from(format!("  {line}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Invariant Health "),
    );
    f.render_widget(panel, area);
}

fn draw_operator_audit(f: &mut Frame, app: &App, area: Rect) {
    let lines: Vec<Line<'static>> = vec![
        Line::from(format!("  Mode: {}", app.alert_profile)),
        Line::from(format!(
            "  Latency Warn: {} ms",
            app.latency_warning_threshold_ms
        )),
        Line::from(format!(
            "  Latency Crit: {} ms",
            app.latency_critical_threshold_ms
        )),
        Line::from(format!("  Failure Warn: {}", app.failure_warning_threshold)),
        Line::from(format!(
            "  Failure Crit: {}",
            app.failure_critical_threshold
        )),
    ];

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Alert Profile "),
    );
    f.render_widget(panel, area);
}

fn draw_incident_timeline(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.incident_timeline.is_empty() {
        lines.push(Line::from("  No recent incidents"));
    } else {
        for entry in app
            .incident_timeline
            .iter()
            .rev()
            .take(MAX_INCIDENT_TIMELINE)
        {
            lines.push(Line::from(format!("  {entry}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Incident Timeline (UTC) "),
    );
    f.render_widget(panel, area);
}

fn draw_ratio_gauge(f: &mut Frame, area: Rect, title: &str, percent: u16, color: Color) {
    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" {title} ")),
        )
        .gauge_style(Style::default().fg(color).add_modifier(Modifier::BOLD))
        .percent(percent)
        .label(format!("{percent}%"));
    f.render_widget(gauge, area);
}

fn draw_phase_health(f: &mut Frame, app: &App, area: Rect) {
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(area);
    let phase = &app.phase_diagnostics;
    let summary = Paragraph::new(vec![Line::from(format!(
        "  period: {}  sharpness: {:.4}  confidence: {:.4}",
        phase.detected_cycle_period,
        phase.transition_sharpness_score,
        phase.attractor_confidence_score
    ))])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Phase Health "),
    );
    f.render_widget(summary, split[0]);

    let confidence = bounded_percent(phase.attractor_confidence_score);
    let sharpness = bounded_percent(phase.transition_sharpness_score);
    draw_ratio_gauge(f, split[1], "Confidence", confidence, Color::Green);
    draw_ratio_gauge(f, split[2], "Sharpness", sharpness, Color::Cyan);
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

fn phase_dynamics_content(app: &App) -> Vec<Line<'static>> {
    let phase = &app.phase_diagnostics;
    if let Some(ref err) = phase.error {
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
            "  PHASE DYNAMICS",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  --------------------------------"),
        Line::from(format!("  state: {}", phase.attractor_state)),
        Line::from(format!("  cycle_length: {}", phase.attractor_cycle_length)),
        Line::from(format!("  entry: {}", phase.attractor_entry_cycle)),
        Line::from(format!("  transition: {:.4}", phase.phase_transition_index)),
        Line::from(format!(
            "  sharpness: {:.4}",
            phase.transition_sharpness_score
        )),
        Line::from(format!(
            "  confidence: {:.4}",
            phase.attractor_confidence_score
        )),
        Line::from(format!("  period: {}", phase.detected_cycle_period)),
        Line::from(format!("  spectrum: {}", phase.cycle_spectrum_class)),
    ]
}

fn actions_content(app: &App) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Available Actions:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  [D] Run Diagnostics"),
        Line::from("  [I] Run Invariants"),
        Line::from("  [L] Run Law Engine"),
        Line::from("  [T] Run Phase Diagnostics"),
        Line::from("  [R] Refresh All"),
        Line::from(""),
        Line::from(Span::styled(
            "  Recent Action Output:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
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
    if app.filtered_session_files.is_empty() {
        lines.push(Line::from("  No sessions scanned"));
    } else {
        for (index, session_file) in app.filtered_session_files.iter().enumerate() {
            let prefix = if index == app.selected_session_index {
                ">"
            } else {
                " "
            };
            lines.push(Line::from(format!("{prefix} {session_file}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Saved Sessions (/ {}) ", app.search_query)),
    );
    f.render_widget(panel, area);
}

fn draw_metadata_pane(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.session_metadata.is_empty() {
        lines.push(Line::from("  No session selected"));
    } else {
        for line in &app.session_metadata {
            lines.push(Line::from(format!("  {line}")));
        }
    }

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Session Metadata "),
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
        Span::styled(
            " [D]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Diagnostics  "),
        Span::styled(
            "[C]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Control  "),
        Span::styled(
            "[M]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Memory  "),
        Span::styled(
            "[A]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Adaptive  "),
        Span::styled(
            "[R]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Refresh All  "),
        Span::styled(
            "[H]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Healing  "),
        Span::styled(
            "[I]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Invariants  "),
        Span::styled(
            "[L]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Law  "),
        Span::styled(
            "[T]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Phase  "),
        Span::styled(
            "[X]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Actions  "),
        Span::styled(
            "[E]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Export  "),
        Span::styled(
            "[P]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Replay  "),
        Span::styled(
            "[S]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Scan  "),
        Span::styled(
            "[1/2/3]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Views  "),
        Span::styled(
            "[G]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Thresholds  "),
        Span::styled(
            "[V]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Diff  "),
        Span::styled(
            "[Q]",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Quit"),
    ]);

    let footer =
        Paragraph::new(legend).block(Block::default().borders(Borders::ALL).title(" Hotkeys "));
    f.render_widget(footer, area);
}

fn draw_search_overlay(f: &mut Frame, app: &App) {
    let area = centered_rect(60, 20, f.size());
    let lines = vec![
        Line::from(""),
        Line::from("  Search Sessions"),
        Line::from("  ---------------"),
        Line::from(format!("  / {}", app.search_query)),
    ];
    let popup =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Search "));
    f.render_widget(popup, area);
}

fn draw_help_overlay(f: &mut Frame) {
    let area = centered_rect(60, 45, f.size());
    let lines = vec![
        Line::from(""),
        Line::from("  Hotkeys"),
        Line::from("  -------"),
        Line::from("  /  Search"),
        Line::from("  ?  Help"),
        Line::from("  S  Scan"),
        Line::from("  V  Diff"),
        Line::from("  T  Phase Dynamics"),
        Line::from("  G  Cycle Alert Profile"),
        Line::from("  E  Export"),
        Line::from("  P  Replay"),
        Line::from("  ↑↓ Navigate"),
        Line::from("  Q  Quit"),
    ];
    let popup = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Help "));
    f.render_widget(popup, area);
}

fn bounded_percent(value: f64) -> u16 {
    if !value.is_finite() {
        return 0;
    }
    let clamped = value.clamp(0.0, 1.0);
    (clamped * 100.0).round() as u16
}

#[cfg(test)]
mod tests {
    use super::{bounded_percent, phase_dynamics_content};
    use crate::app::{App, PhaseDiagnosticsData};

    #[test]
    fn test_bounded_percent_non_finite_guard() {
        assert_eq!(bounded_percent(f64::NAN), 0);
        assert_eq!(bounded_percent(f64::INFINITY), 0);
    }

    #[test]
    fn test_phase_dynamics_renders_negative_entry_cycle() {
        let mut app = App::new();
        app.phase_diagnostics = PhaseDiagnosticsData {
            attractor_state: "drifting".to_string(),
            attractor_cycle_length: 0,
            phase_transition_index: -1.0,
            attractor_entry_cycle: -1,
            transition_sharpness_score: 0.0,
            attractor_confidence_score: 0.0,
            detected_cycle_period: 0,
            cycle_spectrum_class: "aperiodic".to_string(),
            error: None,
        };
        let rendered = phase_dynamics_content(&app)
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("entry: -1"));
        assert!(rendered.contains("transition: -1.0000"));
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}
