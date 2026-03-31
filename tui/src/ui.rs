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
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Min(4),
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
    let phase = &app.phase_diagnostics;
    let snapshots = recent_phase_snapshots(app, 12);
    let confidence_values: Vec<f64> = snapshots
        .iter()
        .map(|snapshot| snapshot.attractor_confidence_score)
        .collect();
    let sharpness_values: Vec<f64> = snapshots
        .iter()
        .map(|snapshot| snapshot.transition_sharpness_score)
        .collect();
    let period_values: Vec<u64> = snapshots
        .iter()
        .map(|snapshot| snapshot.detected_cycle_period)
        .collect();

    let confidence_delta = latest_delta_marker(&confidence_values);
    let period_delta = latest_u64_delta_marker(&period_values);
    let sharpness_delta = latest_delta_marker(&sharpness_values);
    let inflection =
        if has_recent_inflection(&confidence_values) || has_recent_inflection(&sharpness_values) {
        " !"
    } else {
        ""
    };

    let lines = vec![
        Line::from(format!(
            "  period: {}  sharpness: {:.4}  confidence: {:.4}",
            phase.detected_cycle_period,
            phase.transition_sharpness_score,
            phase.attractor_confidence_score
        )),
        Line::from(format!(
            "  confidence trend: {}  {}",
            sparkline(confidence_values.iter().copied()),
            confidence_delta
        )),
        Line::from(format!(
            "  period timeline: {}  {}",
            period_timeline(period_values.iter().copied()),
            period_delta
        )),
        Line::from(format!(
            "  sharpness trend:  {}  {}",
            sparkline(sharpness_values.iter().copied()),
            sharpness_delta
        )),
        Line::from(format!(
            "  signature strip:  {}",
            signature_timeline(
                snapshots
                    .iter()
                    .map(|snapshot| attractor_signature(&snapshot.attractor_state))
            )
        )),
        Line::from(format!(
            "  confidence velocity: {}{}",
            confidence_velocity_strip(&confidence_values, 12),
            inflection
        )),
    ];

    let panel = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Phase Health "),
    );
    f.render_widget(panel, area);
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
    if !app.replay_lines.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Replay mode:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(format!(
            "  timeline: {}",
            replay_timeline_strip(app.replay_lines.len())
        )));
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

    let footer = Paragraph::new(legend).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Hotkeys + trend observability / phase history "),
    );
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
        Line::from("  trend observability: confidence/sharpness"),
        Line::from("  phase history: period + attractor signature"),
        Line::from("  G  Cycle Alert Profile"),
        Line::from("  E  Export"),
        Line::from("  P  Replay"),
        Line::from("  ↑↓ Navigate"),
        Line::from("  Q  Quit"),
    ];
    let popup = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Help "));
    f.render_widget(popup, area);
}

fn recent_phase_snapshots<'a>(app: &'a App, window: usize) -> &'a [crate::app::PhaseSnapshot] {
    let len = app.phase_snapshots.len();
    let start = len.saturating_sub(window);
    &app.phase_snapshots[start..]
}

const INSUFFICIENT_HISTORY_PLACEHOLDER: &str = "(history n<2)";

fn sparkline<I>(values: I) -> String
where
    I: Iterator<Item = f64>,
{
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut out = String::new();
    let mut count = 0usize;
    for value in values {
        let glyph = if value.is_finite() {
            let clamped = value.clamp(0.0, 1.0);
            let idx = (clamped * (BARS.len() as f64 - 1.0)).round() as usize;
            BARS[idx]
        } else {
            '?'
        };
        out.push(glyph);
        count += 1;
    }
    if count < 2 {
        return INSUFFICIENT_HISTORY_PLACEHOLDER.to_string();
    }
    out
}

fn period_timeline<I>(values: I) -> String
where
    I: Iterator<Item = u64>,
{
    let mut out = String::new();
    let mut count = 0usize;
    for value in values {
        if count > 0 {
            out.push(' ');
        }
        out.push_str(&value.to_string());
        count += 1;
    }
    if count < 2 {
        return INSUFFICIENT_HISTORY_PLACEHOLDER.to_string();
    }
    out
}

fn attractor_signature(state: &str) -> char {
    match state {
        "fixed_point" => 'F',
        "period_two" => 'P',
        "drifting_phase" => 'D',
        "intervention_phase" => 'I',
        "fixed_cycle" => 'F',
        "low_period_cycle" => 'P',
        "medium_period_cycle" => 'P',
        "drifting_cycle" => 'D',
        _ => 'A',
    }
}

fn signature_timeline<I>(values: I) -> String
where
    I: Iterator<Item = char>,
{
    let mut out = String::new();
    let mut count = 0usize;
    for value in values {
        if count > 0 {
            out.push(' ');
        }
        out.push(value);
        count += 1;
    }
    if count < 2 {
        return INSUFFICIENT_HISTORY_PLACEHOLDER.to_string();
    }
    out
}

fn bounded_percent(value: f64) -> u16 {
    if !value.is_finite() {
        return 0;
    }
    let clamped = value.clamp(0.0, 1.0);
    (clamped * 100.0).round() as u16
}

fn ordering_marker(is_greater: bool, is_less: bool) -> char {
    if is_greater {
        '+'
    } else if is_less {
        '-'
    } else {
        '='
    }
}

fn latest_pair<T: Copy>(values: &[T]) -> Option<(T, T)> {
    if values.len() < 2 {
        None
    } else {
        Some((values[values.len() - 2], values[values.len() - 1]))
    }
}

fn delta_marker(previous: f64, latest: f64) -> char {
    if !previous.is_finite() || !latest.is_finite() {
        return '=';
    }
    ordering_marker(latest > previous, latest < previous)
}

fn latest_delta_marker(values: &[f64]) -> char {
    match latest_pair(values) {
        Some((previous, latest)) => delta_marker(previous, latest),
        None => '=',
    }
}

fn latest_u64_delta_marker(values: &[u64]) -> char {
    match latest_pair(values) {
        Some((previous, latest)) => ordering_marker(latest > previous, latest < previous),
        None => '=',
    }
}

fn has_recent_inflection(values: &[f64]) -> bool {
    if values.len() < 3 {
        return false;
    }
    let recent = &values[values.len().saturating_sub(4)..];
    let mut first_sign: Option<i8> = None;
    for pair in recent.windows(2) {
        if !pair[0].is_finite() || !pair[1].is_finite() {
            continue;
        }
        let sign = if pair[1] > pair[0] {
            1
        } else if pair[1] < pair[0] {
            -1
        } else {
            0
        };
        if sign == 0 {
            continue;
        }
        if let Some(existing) = first_sign {
            if existing != sign {
                return true;
            }
        } else {
            first_sign = Some(sign);
        }
    }
    false
}

fn confidence_velocity_strip(values: &[f64], window: usize) -> String {
    if values.len() < 2 {
        return INSUFFICIENT_HISTORY_PLACEHOLDER.to_string();
    }
    let start = values.len().saturating_sub(window);
    let mut out = String::new();
    let mut count = 0usize;
    for pair in values[start..].windows(2) {
        if count > 0 {
            out.push(' ');
        }
        out.push(delta_marker(pair[0], pair[1]));
        count += 1;
    }
    if count == 0 {
        return INSUFFICIENT_HISTORY_PLACEHOLDER.to_string();
    }
    out
}

fn replay_timeline_strip(frame_count: usize) -> String {
    if frame_count == 0 {
        return INSUFFICIENT_HISTORY_PLACEHOLDER.to_string();
    }
    let width = frame_count.min(10).max(2);
    let mut out = String::from("[");
    for idx in 0..width {
        if idx + 1 == width {
            out.push('|');
        } else {
            out.push('=');
        }
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::{
        attractor_signature, bounded_percent, confidence_velocity_strip, has_recent_inflection,
        latest_delta_marker, latest_u64_delta_marker, period_timeline, phase_dynamics_content,
        replay_timeline_strip, signature_timeline, sparkline, INSUFFICIENT_HISTORY_PLACEHOLDER,
    };
    use crate::app::{App, PhaseDiagnosticsData};

    #[test]
    fn test_bounded_percent_non_finite_guard() {
        assert_eq!(bounded_percent(f64::NAN), 0);
        assert_eq!(bounded_percent(f64::INFINITY), 0);
    }

    #[test]
    fn test_sparkline_deterministic_rendering() {
        assert_eq!(sparkline([0.0, 0.5, 1.0].into_iter()), "▁▅█");
    }

    #[test]
    fn test_history_placeholder_for_insufficient_points() {
        assert_eq!(
            sparkline([0.3].into_iter()),
            INSUFFICIENT_HISTORY_PLACEHOLDER
        );
        assert_eq!(
            period_timeline([3].into_iter()),
            INSUFFICIENT_HISTORY_PLACEHOLDER
        );
        assert_eq!(
            signature_timeline(['F'].into_iter()),
            INSUFFICIENT_HISTORY_PLACEHOLDER
        );
    }

    #[test]
    fn test_sparkline_anomaly_glyph_rendering() {
        assert_eq!(sparkline([0.8, f64::NAN, 0.6].into_iter()), "▇?▅");
        assert_eq!(sparkline([0.8, f64::INFINITY, 0.6].into_iter()), "▇?▅");
    }

    #[test]
    fn test_attractor_signature_mapping() {
        assert_eq!(attractor_signature("fixed_point"), 'F');
        assert_eq!(attractor_signature("period_two"), 'P');
        assert_eq!(attractor_signature("drifting_phase"), 'D');
        assert_eq!(attractor_signature("intervention_phase"), 'I');
        assert_eq!(attractor_signature("fixed_cycle"), 'F');
        assert_eq!(attractor_signature("low_period_cycle"), 'P');
        assert_eq!(attractor_signature("medium_period_cycle"), 'P');
        assert_eq!(attractor_signature("drifting_cycle"), 'D');
        assert_eq!(attractor_signature("unknown"), 'A');
    }

    #[test]
    fn test_delta_markers_and_velocity_strip() {
        assert_eq!(latest_delta_marker(&[0.2, 0.4]), '+');
        assert_eq!(latest_delta_marker(&[0.4, 0.2]), '-');
        assert_eq!(latest_delta_marker(&[0.4, 0.4]), '=');
        assert_eq!(latest_u64_delta_marker(&[2, 5]), '+');
        assert_eq!(latest_u64_delta_marker(&[5, 2]), '-');
        assert_eq!(
            confidence_velocity_strip(&[0.1, 0.2, 0.2, 0.1, 0.3], 12),
            "+ = - +"
        );
    }

    #[test]
    fn test_inflection_and_replay_timeline_strip() {
        assert!(has_recent_inflection(&[0.1, 0.4, 0.2]));
        assert!(!has_recent_inflection(&[0.1, 0.2, 0.3, 0.4]));
        assert_eq!(replay_timeline_strip(0), INSUFFICIENT_HISTORY_PLACEHOLDER);
        assert_eq!(replay_timeline_strip(1), "[=|]");
        assert_eq!(replay_timeline_strip(5), "[====|]");
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
