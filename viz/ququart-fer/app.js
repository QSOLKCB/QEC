(() => {
  "use strict";

  const loading = document.querySelector("#loading");
  const reportNode = document.querySelector("#report");
  const data = window.QEC_QUQUART_REPORT;
  if (!data) {
    loading.textContent = "Generated evidence is missing. Run qec-ququart-bench with --output viz/ququart-fer/results, then reload.";
    loading.classList.add("fail");
    return;
  }

  loading.hidden = true;
  reportNode.hidden = false;
  const number = value => Number(value);
  const sci = value => {
    const n = number(value);
    if (!Number.isFinite(n)) return "—";
    return n === 0 ? "0" : n.toExponential(3);
  };
  const pct = value => `${(100 * number(value)).toFixed(2)}%`;
  const pretty = value => String(value).replaceAll("_", " ");

  document.querySelector("#patterns").textContent = data.summary.exact_patterns.toLocaleString();
  document.querySelector("#claim-state").textContent = data.claim_validation.passed ? "PASS" : "FAIL";
  document.querySelector("#lane-state").textContent = data.lane_symmetry_certificate.weight_enumerators_equal ? "VERIFIED" : "FAILED";
  document.querySelector("#replication-state").textContent = pretty(data.qbraid_replication_receipt.verification.deterministic_artifacts);
  document.querySelector("#seed").textContent = data.summary.seed;
  document.querySelector("#hash").textContent =
    `methodology sha256: ${data.methodology_sha256} · claim validation: ${data.claim_validation.sha256} · v170.0 certificate: ${data.certificate_sha256}`;

  const weightHeaders = [
    ["weight", "w"],
    ["patterns", "patterns"],
    ["corrected", "corrected"],
    ["detected_uncorrectable", "detected"],
    ["logical_failure", "logical"],
  ];
  const weightTable = document.querySelector("#weight-table");
  function drawWeightTable(channel) {
    const rows = data.exact_channel_weight_enumerator.filter(row => row.channel === channel);
    weightTable.innerHTML = `<thead><tr>${weightHeaders.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead><tbody>${
      rows.map(row =>
        `<tr>${weightHeaders.map(([key]) => `<td>${Number(row[key]).toLocaleString()}</td>`).join("")}</tr>`
      ).join("")
    }</tbody>`;
  }

  const faultHeaders = [
    ["fault_case", "case"],
    ["accepted", "accept"],
    ["receiver_rejections", "receiver reject"],
    ["decoder_rejections", "decoder reject"],
    ["successful", "success"],
    ["receiver_false_trust", "false trust"],
  ];
  const faultTable = document.querySelector("#fault-table");
  faultTable.innerHTML = `<thead><tr>${faultHeaders.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead><tbody>${
    data.harmonic_fault_matrix.map(row => {
      const expectedPass = row.expected === "accept_and_correct_all"
        ? row.successful === row.errors_tested && row.false_accepts === 0
        : row.accepted === 0 && row.receiver_false_trust === 0;
      return `<tr class="${expectedPass ? "pass" : "fail"}">${
        faultHeaders.map(([key]) => `<td>${row[key]}</td>`).join("")
      }</tr>`;
    }).join("")
  }</tbody>`;

  const channels = [...new Set(data.exact_channel_fer.map(row => row.channel))];
  const channelSelect = document.querySelector("#channel");
  channelSelect.innerHTML = channels.map(channel =>
    `<option value="${channel}">${pretty(channel)}</option>`
  ).join("");

  function logBounds(values, fallback) {
    const positive = values
      .map(Number)
      .filter(value => Number.isFinite(value) && value > 0);
    if (positive.length === 0) return [...fallback];
    const minimum = Math.min(...positive);
    const maximum = Math.max(...positive);
    if (minimum === maximum) return [minimum / 10, maximum * 10];
    return [minimum, maximum];
  }

  function drawChart(channel) {
    const exact = data.exact_channel_fer.filter(row => row.channel === channel).map(row => ({
      x: number(row.physical_error_rate),
      y: number(row.frame_error_rate),
    })).filter(point => point.x > 0 && point.y > 0);
    const mc = data.monte_carlo_fer.filter(row => row.channel === channel).map(row => ({
      x: number(row.physical_error_rate),
      y: number(row.frame_error_rate),
    })).filter(point => point.x > 0 && point.y > 0);

    const svg = document.querySelector("#fer-chart");
    const width = 960, height = 460;
    const margin = {left: 78, right: 28, top: 35, bottom: 62};
    const allX = [...exact, ...mc].map(point => point.x);
    const allY = [...exact, ...mc].map(point => point.y);
    const [xmin, xmax] = logBounds(allX, [1e-5, 1]);
    const [yminRaw, ymaxRaw] = logBounds(allY, [1e-10, 1]);
    const ymin = Math.min(yminRaw, 1e-10);
    const ymax = Math.max(ymaxRaw, ymin * 10);
    const lx = value => Math.log10(value);
    const xScale = value => margin.left + (lx(value) - lx(xmin)) / (lx(xmax) - lx(xmin)) * (width - margin.left - margin.right);
    const yScale = value => height - margin.bottom - (lx(value) - lx(ymin)) / (lx(ymax) - lx(ymin)) * (height - margin.top - margin.bottom);
    const line = points => points.map((point, index) => `${index ? "L" : "M"}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`).join(" ");

    const xTicks = [-5, -4, -3, -2, -1, 0].map(power => 10 ** power).filter(v => v >= xmin && v <= xmax);
    const yTicks = Array.from({length: 11}, (_, i) => 10 ** (-10 + i)).filter(v => v >= ymin && v <= ymax);
    svg.innerHTML = `
      ${xTicks.map(v => `<line class="gridline" x1="${xScale(v)}" x2="${xScale(v)}" y1="${margin.top}" y2="${height-margin.bottom}"/><text class="chart-label" x="${xScale(v)}" y="${height-30}" text-anchor="middle">${v.toExponential(0)}</text>`).join("")}
      ${yTicks.map(v => `<line class="gridline" x1="${margin.left}" x2="${width-margin.right}" y1="${yScale(v)}" y2="${yScale(v)}"/><text class="chart-label" x="${margin.left-12}" y="${yScale(v)+4}" text-anchor="end">${v.toExponential(0)}</text>`).join("")}
      <line class="axis" x1="${margin.left}" x2="${width-margin.right}" y1="${height-margin.bottom}" y2="${height-margin.bottom}"/>
      <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height-margin.bottom}"/>
      <path class="exact-line" d="${line(exact)}"/>
      ${mc.length > 1 ? `<path class="mc-line" d="${line(mc)}"/>` : ""}
      ${mc.map(point => `<circle class="point" cx="${xScale(point.x)}" cy="${yScale(point.y)}" r="4"/>`).join("")}
      <text class="chart-label" x="${width/2}" y="${height-7}" text-anchor="middle">independent physical error probability p per ququart</text>
      <text class="chart-label" transform="translate(18 ${height/2}) rotate(-90)" text-anchor="middle">frame error rate</text>
      <text class="chart-label legend-exact" x="${width-250}" y="28">━━ exact selected channel</text>
      <text class="chart-label legend-mc" x="${width-115}" y="28">┄ MC selected</text>
    `;
    document.querySelector("#chart-caption").textContent =
      `Exact and sampled evidence now use the same declared ${pretty(channel)} channel. No cross-channel overlay is implied.`;
    drawWeightTable(channel);
  }
  channelSelect.addEventListener("change", () => drawChart(channelSelect.value));
  drawChart(channelSelect.value);

  const facts = data.evidence_facts;
  document.querySelector("#claim-facts").innerHTML = [
    ["End-to-end cells", facts.end_to_end_cells],
    ["Deterministic evaluations", facts.deterministic_fault_evaluations],
    ["Expected accepts", facts.expected_accept_evaluations],
    ["Adversarial rejects", facts.adversarial_rejection_evaluations],
    ["Receiver false trust", facts.receiver_false_trust],
    ["Threshold claim", data.report_claims.threshold_claim ? "permitted" : "forbidden"],
  ].map(([term, value]) => `<div><dt>${term}</dt><dd>${value}</dd></div>`).join("");

  const replication = data.qbraid_replication_receipt;
  document.querySelector("#replication-facts").innerHTML = [
    ["Environment", replication.environment.platform],
    ["Seed", replication.parameters.seed],
    ["Monte Carlo trials", replication.parameters.monte_carlo_trials_per_cell],
    ["Harmonic trials", replication.parameters.harmonic_trials_per_cell],
    ["Deterministic evidence", pretty(replication.verification.deterministic_artifacts)],
    ["Sampled evidence", pretty(replication.verification.sampled_artifacts)],
  ].map(([term, value]) => `<div><dt>${term}</dt><dd>${value}</dd></div>`).join("");

  const rates = [...new Set(data.harmonic_end_to_end.map(row => row.physical_error_rate))];
  const heatRate = document.querySelector("#heat-rate");
  heatRate.innerHTML = rates.map(rate => `<option value="${rate}">${rate}</option>`).join("");
  function drawHarmonicCards(rate) {
    const rows = data.harmonic_end_to_end.filter(row => row.physical_error_rate === rate);
    document.querySelector("#harmonic-cards").innerHTML = rows.map(row => `
      <article>
        <span>σ = ${row.harmonic_noise_sigma}</span>
        <strong>${pct(row.frame_error_rate)}</strong>
        <small>${row.receiver_rejections} receiver reject · ${row.decoder_rejections} decoder reject</small>
        <small>${row.receiver_false_trust} false trust · ${row.accepted_logical_residual} logical residual</small>
        <small>95% CI ${sci(row.wilson95_low)} — ${sci(row.wilson95_high)}</small>
      </article>
    `).join("");
  }
  heatRate.addEventListener("change", () => drawHarmonicCards(heatRate.value));
  drawHarmonicCards(heatRate.value);
})();
