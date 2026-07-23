(function () {
  "use strict";

  const Core = globalThis.QutritCore;
  const Draw = globalThis.QutritDraw;
  const byId = (id) => document.getElementById(id);
  const controls = [
    "code", "site-a", "x-a", "z-a", "second",
    "site-b", "x-b", "z-b", "fault"
  ];
  let currentResult;

  function selectOptions(select, values, label) {
    const previous = Number(select.value);
    select.replaceChildren(...values.map((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = label(value);
      return option;
    }));
    select.value = String(Math.min(previous || 0, values.at(-1)));
  }

  function configure(code) {
    const sites = Array.from({ length: code.n }, (_, index) => index);
    ["site-a", "site-b"].forEach((id) =>
      selectOptions(byId(id), sites, (site) => "q" + site)
    );
    if (Number(byId("site-b").value) === 0 && code.n > 1) {
      byId("site-b").value = "1";
    }
    byId("code-note").textContent =
      code.stabilizers.length + " exact commuting checks · corrects every error through weight " + code.t;
  }

  function currentError(code) {
    const terms = [{
      site: Number(byId("site-a").value),
      x: Number(byId("x-a").value),
      z: Number(byId("z-a").value)
    }];
    if (byId("second").checked) {
      terms.push({
        site: Number(byId("site-b").value),
        x: Number(byId("x-b").value),
        z: Number(byId("z-b").value)
      });
    }
    return Core.fromTerms(code.n, terms);
  }

  function inspect(result) {
    byId("parameter").textContent =
      "[[" + result.code.n + "," + result.code.k + "," + result.code.d + "]]₃";
    byId("radius").textContent = "exact radius t = " + result.code.t;
    byId("weight").textContent = "w = " + result.errorWeight;
    byId("coverage").textContent =
      (Core.decoder(result.code).size - 1).toLocaleString() + " enumerated errors";
    byId("trust").textContent = result.observation.trusted ? "TRUSTED" : "REJECTED";
    byId("outcome").textContent = result.success
      ? "logical correction succeeded"
      : !result.observation.trusted
        ? "no correction applied"
        : result.accepted
          ? "logical miscorrection detected"
          : "syndrome outside exact table";
    byId("cycle-status").textContent = result.success
      ? "Correction closes to the stabilizer"
      : !result.observation.trusted
        ? "Harmonic receiver failed closed"
        : result.accepted
          ? "Accepted residual is outside the stabilizer"
          : "Exact decoder rejected the syndrome";
    byId("syndrome").replaceChildren(...result.exactSyndrome.map((value, index) => {
      const cell = document.createElement("span");
      cell.dataset.state = String(value);
      cell.textContent = value;
      cell.setAttribute("aria-label", "check " + index + ": " + value);
      return cell;
    }));
    const check = Math.max(0, result.exactSyndrome.findIndex((value) => value));
    const symbol = result.exactSyndrome[check] || 0;
    byId("etq").textContent =
      "(" + check + ", " + symbol + ") → " + (3 * check + symbol);
    byId("agreement").textContent = result.observation.agreement
      ? "exact agreement" : "discord → reject";
    byId("dark").textContent = result.observation.darkInvariant
      ? "state-dark baseline holds" : "distortion → reject";
    byId("residual").textContent = String(Core.weight(result.residual));
  }

  function render(reconfigure) {
    const code = Core.CODES[byId("code").value];
    if (reconfigure) configure(code);
    byId("second-row").hidden = !byId("second").checked;
    currentResult = Core.run(code, currentError(code), byId("fault").value);
    inspect(currentResult);
    Draw.topology(currentResult);
    Draw.harmonics(currentResult);
    Draw.modes(currentResult);
  }

  function hear() {
    if (!currentResult) return;
    const AudioContext = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AudioContext) return;
    const audio = new AudioContext();
    const buffer = audio.createBuffer(1, audio.sampleRate * 0.7, audio.sampleRate);
    const output = buffer.getChannelData(0);
    currentResult.exactSyndrome.forEach((symbol, index) => {
      const frequency = 170 * 2 ** (index / 12);
      const phase = 2 * Math.PI * symbol / 3;
      for (let sample = 0; sample < output.length; sample++) {
        const time = sample / audio.sampleRate;
        output[sample] += (
          Math.sin(2 * Math.PI * frequency * time + phase) +
          0.45 * Math.sin(4 * Math.PI * frequency * time + 2 * phase) +
          0.15 * Math.sin(6 * Math.PI * frequency * time)
        ) / Math.max(1, currentResult.exactSyndrome.length);
      }
    });
    const source = audio.createBufferSource();
    source.buffer = buffer;
    source.connect(audio.destination);
    source.start();
    source.onended = () => audio.close();
  }

  ["x-a", "z-a", "x-b", "z-b"].forEach((id) =>
    selectOptions(byId(id), [0, 1, 2], String)
  );
  byId("x-a").value = "1";
  byId("z-a").value = "1";
  byId("x-b").value = "2";
  byId("z-b").value = "1";
  controls.forEach((id) =>
    byId(id).addEventListener("change", () => render(id === "code"))
  );
  byId("hear").addEventListener("click", hear);
  globalThis.addEventListener("resize", () => render(false));
  render(true);
})();
