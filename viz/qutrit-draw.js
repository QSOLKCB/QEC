// Canvas receivers for the qutrit visual lab.
(function (global, factory) {
  global.QutritDraw = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const css = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function fit(id) {
    const canvas = byId(id);
    const bounds = canvas.getBoundingClientRect();
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    canvas.width = Math.max(320, Math.round(bounds.width * dpr));
    canvas.height = Math.max(120, Math.round(bounds.height * dpr));
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { context, width: bounds.width, height: bounds.height };
  }

  function topology(result) {
    const { context, width, height } = fit("topology");
    const center = [width / 2, height / 2 + 8];
    const radius = Math.min(width, height) * 0.34;
    const points = Array.from({ length: result.code.n }, (_, site) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * site / result.code.n;
      return [
        center[0] + radius * Math.cos(angle),
        center[1] + radius * Math.sin(angle)
      ];
    });
    context.clearRect(0, 0, width, height);
    context.strokeStyle = css("--line");
    context.lineWidth = 1;

    result.code.stabilizers.slice(0, 6).forEach((check) => {
      const support = points.filter(
        (_, site) => check[site] || check[result.code.n + site]
      );
      if (support.length < 2) return;
      context.beginPath();
      support.forEach(([x, y], index) =>
        index ? context.lineTo(x, y) : context.moveTo(x, y)
      );
      context.closePath();
      context.globalAlpha = 0.24;
      context.stroke();
    });
    context.globalAlpha = 1;

    points.forEach(([x, y], site) => {
      const hasError = result.error.x[site] || result.error.z[site];
      const hasCorrection = result.correction &&
        (result.correction.x[site] || result.correction.z[site]);
      const hasResidual = result.residual.x[site] || result.residual.z[site];
      context.beginPath();
      context.arc(x, y, 17, 0, 2 * Math.PI);
      context.fillStyle = hasResidual ? css("--green") : css("--raised");
      context.fill();
      context.strokeStyle = hasError ? css("--oxide") : css("--line");
      context.lineWidth = hasError ? 4 : 1;
      context.stroke();
      if (hasCorrection) {
        context.beginPath();
        context.arc(x, y, 23, 0, 2 * Math.PI);
        context.strokeStyle = css("--mineral");
        context.lineWidth = 2;
        context.stroke();
      }
      context.fillStyle = css("--ink");
      context.font = "11px " + css("--mono");
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("q" + site, x, y);
      if (hasError) {
        context.fillStyle = css("--oxide");
        context.fillText(
          "X" + result.error.x[site] + " Z" + result.error.z[site],
          x,
          y + 34
        );
      }
    });

    context.fillStyle = css("--muted");
    context.font = "11px " + css("--mono");
    context.textAlign = "center";
    context.fillText(result.code.name, center[0], center[1] - 8);
    context.fillStyle = result.success ? css("--green") : css("--red");
    context.font = "700 15px " + css("--mono");
    context.fillText(
      result.success ? "LOGICAL STATE CLEAN" : "NO CLEAN CORRECTION",
      center[0],
      center[1] + 14
    );
  }

  function harmonics(result) {
    const { context, width, height } = fit("harmonics");
    const rows = [
      ["H1", result.observation.h1],
      ["H2", result.observation.h2],
      ["H3", result.observation.h3]
    ];
    const left = 42;
    const cell = (width - left - 12) / Math.max(1, result.exactSyndrome.length);
    context.clearRect(0, 0, width, height);
    rows.forEach(([label, samples], row) => {
      const y = 48 + row * 68;
      context.fillStyle = css("--muted");
      context.font = "11px " + css("--mono");
      context.textAlign = "left";
      context.fillText(label, 12, y + 4);
      samples.forEach(([real, imag], index) => {
        const x = left + cell * (index + 0.5);
        const radius = Math.min(13, cell * 0.28);
        context.beginPath();
        context.arc(x, y, radius, 0, 2 * Math.PI);
        context.strokeStyle = css("--line");
        context.lineWidth = 1;
        context.stroke();
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(x + real * radius, y - imag * radius);
        context.strokeStyle = row === 0
          ? css("--amber") : row === 1 ? css("--mineral") : css("--oxide");
        context.lineWidth = 2;
        context.stroke();
      });
    });
  }

  function modes(result) {
    const { context, width, height } = fit("modes");
    const values = result.modePower;
    const maximum = Math.max(...values, 1);
    const base = height - 28;
    const cell = (width - 36) / values.length;
    context.clearRect(0, 0, width, height);
    context.strokeStyle = css("--line");
    context.beginPath();
    context.moveTo(28, base);
    context.lineTo(width - 8, base);
    context.stroke();
    values.forEach((value, mode) => {
      const barHeight = (height - 52) * value / maximum;
      context.fillStyle = mode ? css("--mineral") : css("--amber");
      context.globalAlpha = 0.8;
      context.fillRect(
        30 + cell * mode,
        base - barHeight,
        Math.max(3, cell - 4),
        barHeight
      );
      context.globalAlpha = 1;
      context.fillStyle = css("--muted");
      context.font = "9px " + css("--mono");
      context.textAlign = "center";
      context.fillText(String(mode), 30 + cell * (mode + 0.5) - 2, base + 14);
    });
    context.fillStyle = css("--muted");
    context.textAlign = "left";
    context.fillText("collective mode k", 10, 14);
  }

  return { harmonics, modes, topology };
});
