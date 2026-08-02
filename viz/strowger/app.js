// SPDX-License-Identifier: MPL-2.0
"use strict";

const $ = (selector) => document.querySelector(selector);
const rack = $("#rack");
const eventList = $("#events");
let lastReceipt = null;
let lastTones = null;
let audioContext = null;
let operatorEvents = [];

function parseCsv(id) {
  return $(id).value.split(",").map((part) => Number.parseInt(part.trim(), 10));
}
function hash32(text) {
  let value = 2166136261;
  for (const char of text) {
    value ^= char.charCodeAt(0);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}
function pulseCount(digit, radix) {
  return digit === 0 ? radix : digit;
}
function firstFree(states) {
  return states.findIndex((state) => state === "free");
}
function deriveTones(identity) {
  const a = hash32(identity);
  const b = hash32(identity.split("").reverse().join(""));
  return {
    route_hz: 320 + (a % 1200),
    check_hz: 320 + (b % 1200),
    dark_reference_hz: 90
  };
}
function pseudoSha(text) {
  const parts = [];
  let seed = text;
  for (let i = 0; i < 8; i += 1) {
    const value = hash32(`${seed}|${i}`).toString(16).padStart(8, "0");
    parts.push(value);
    seed = value;
  }
  return parts.join("");
}
function appendEvent(events, device, action, details = {}) {
  const previous = events.length ? events[events.length - 1].event_sha256 : null;
  const event = {
    sequence: events.length,
    tick: events.length,
    device,
    action,
    details,
    previous_event_sha256: previous
  };
  event.event_sha256 = pseudoSha(JSON.stringify(event));
  events.push(event);
}
function selectorCard(name, radix, trunkCount, selected, level, states) {
  const card = document.createElement("article");
  card.className = "selector";
  card.innerHTML = `<h3>${name.toUpperCase()}</h3><div class="shaft"></div><div class="wiper"></div><div class="contacts"></div><div class="reading">level ${level} · contact ${selected < 0 ? "—" : selected}</div>`;
  const contacts = card.querySelector(".contacts");
  for (let i = 0; i < trunkCount; i += 1) {
    const dot = document.createElement("i");
    const angle = (Math.PI * 2 * i / trunkCount) - Math.PI / 2;
    dot.className = `contact ${states[i] || "free"} ${i === selected ? "active" : ""}`;
    dot.style.left = `${50 + Math.cos(angle) * 42}%`;
    dot.style.top = `${43 + Math.sin(angle) * 38}%`;
    contacts.append(dot);
  }
  const wiper = card.querySelector(".wiper");
  const angle = selected < 0 ? -90 : (360 * selected / trunkCount) - 90;
  wiper.style.transform = `rotate(${angle}deg)`;
  const shaft = card.querySelector(".shaft");
  shaft.style.transform = `translateX(-50%) translateY(${Math.min(level, radix - 1) * 4}px)`;
  return card;
}
function renderRack(radices, selected = [], states = []) {
  rack.replaceChildren();
  const labels = radices.slice(0, -2).map((_, i) => `Selector ${i + 1}`);
  labels.push("Connector");
  labels.forEach((name, i) => {
    const radix = i < radices.length - 2 ? radices[i] : radices.at(-1);
    const level = i < radices.length - 2 ? (lastReceipt?.digits[i] ?? 0) : (lastReceipt?.digits.at(-2) ?? 0);
    rack.append(selectorCard(name, radix, i === labels.length - 1 ? radices.at(-1) : 6, selected[i] ?? -1, level, states[i] || []));
  });
}
function renderEvents(events) {
  eventList.replaceChildren();
  for (const event of events) {
    const li = document.createElement("li");
    li.innerHTML = `<b>${String(event.sequence).padStart(3, "0")} · ${event.device}</b><br>${event.action} ${JSON.stringify(event.details)}`;
    eventList.append(li);
  }
  eventList.scrollTop = eventList.scrollHeight;
}
function drawScope(tones) {
  const canvas = $("#scope");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#2d332b";
  ctx.lineWidth = 1;
  for (let y = 30; y < height; y += 30) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  if (!tones) return;
  ctx.strokeStyle = "#c39a4a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const t = x / width;
    const y = height / 2
      + Math.sin(t * tones.route_hz * .08) * 26
      + Math.sin(t * tones.check_hz * .08) * 15
      + Math.sin(t * tones.dark_reference_hz * .08) * 8;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
function operatorEnabled() {
  return $("#mode").value !== "automatic";
}
function updateDesk() {
  const mode = $("#mode").value;
  $("#desk-state").textContent = mode === "automatic" ? "DISABLED" : mode.toUpperCase();
  $("#desk-state").style.color = mode === "automatic" ? "#b75b4f" : "#8ea86d";
  document.querySelectorAll("[data-op]").forEach((button) => {
    const manualOnly = ["step", "seize"].includes(button.dataset.op);
    button.disabled = mode === "automatic" || (manualOnly && mode !== "manual");
  });
}
function operatorAction(action) {
  if (!operatorEnabled()) return;
  const event = {
    action,
    operator_id: "local-console",
    target: action === "quarantine" ? "selector-1:0" : "exchange",
    reason: "operator-desk action"
  };
  operatorEvents.push(event);
  if (lastReceipt) {
    appendEvent(lastReceipt.events, "operator-desk", `operator_${action}`, event);
    lastReceipt.operator_commands.push(event);
    lastReceipt.sha256 = pseudoSha(JSON.stringify(lastReceipt));
    renderEvents(lastReceipt.events);
    $("#receipt-hash").textContent = lastReceipt.sha256;
  }
}
function route() {
  const digits = parseCsv("#digits");
  const radices = parseCsv("#radices");
  if (digits.length !== radices.length || radices.length < 3 || digits.some((d, i) => !Number.isInteger(d) || d < 0 || d >= radices[i])) {
    $("#state").textContent = "INVALID REQUEST";
    return;
  }
  const events = [];
  appendEvent(events, "linefinder-0", "seize_first_free_linefinder", {request_id: "browser-call"});
  const selected = [];
  const states = [];
  let outcome = "committed";
  const fault = $("#fault").value;
  for (let stage = 0; stage < radices.length - 2; stage += 1) {
    const state = Array(6).fill("free");
    if (fault === "busy" && stage === 0) {
      state[0] = "busy"; state[1] = "busy";
    }
    if (operatorEvents.some((event) => event.action === "quarantine") && stage === 0) state[0] = "quarantined";
    states.push(state);
    let observed = pulseCount(digits[stage], radices[stage]);
    if (fault === "missed" && stage === 0) observed -= 1;
    if (fault === "duplicate" && stage === 0) observed += 1;
    appendEvent(events, `selector-${stage}`, "receive_digit", {digit: digits[stage], expected_pulses: pulseCount(digits[stage], radices[stage]), observed_pulses: observed});
    if (observed !== pulseCount(digits[stage], radices[stage]) || (fault === "stuck" && stage === 1)) {
      appendEvent(events, `selector-${stage}`, "selector_fault", {reason: fault});
      outcome = "selector_fault";
      break;
    }
    const trunk = firstFree(state);
    if (trunk < 0) {
      appendEvent(events, `selector-${stage}`, "all_trunks_busy");
      outcome = "all_trunks_busy";
      break;
    }
    selected.push(trunk);
    appendEvent(events, `selector-${stage}`, "first_free_trunk_selected", {level: digits[stage], contact: trunk});
  }
  let tones = null;
  let verified = false;
  if (outcome === "committed") {
    appendEvent(events, "connector", "two_axis_destination_selected", {vertical: digits.at(-2), rotary: digits.at(-1), destination: $("#destination").value});
    tones = deriveTones(`${digits}|${selected}|${$("#destination").value}`);
    const observed = {...tones};
    if (fault === "tone") observed.route_hz += 7;
    verified = JSON.stringify(tones) === JSON.stringify(observed);
    appendEvent(events, "tone-verifier", "verify_route_tones", {expected: tones, observed, verified});
    if (!verified) outcome = "tone_mismatch";
    appendEvent(events, "exchange", verified ? "commit_verified_route" : "reject_tone_mismatch", {destination: $("#destination").value});
  }
  lastReceipt = {
    schema: "qec.strowger-route-receipt.browser-demo.v1",
    qec_version: "170.3.0",
    mode: $("#mode").value,
    digits,
    radices,
    destination: $("#destination").value,
    selected,
    outcome,
    events,
    operator_commands: [...operatorEvents],
    claim_boundary: {
      classical_routing_only: true,
      decoder_replacement: false,
      quantum_hardware_claim: false,
      operator_may_force_accept: false
    }
  };
  lastReceipt.sha256 = pseudoSha(JSON.stringify(lastReceipt));
  lastTones = tones;
  $("#state").textContent = outcome.toUpperCase();
  $("#linefinder").textContent = "0";
  $("#route-path").textContent = selected.length ? selected.join(" → ") : "—";
  $("#verify").textContent = tones ? (verified ? "MATCH" : "REJECT") : "—";
  $("#route-tone").textContent = tones ? `${tones.route_hz} Hz` : "—";
  $("#check-tone").textContent = tones ? `${tones.check_hz} Hz` : "—";
  $("#dark-tone").textContent = tones ? `${tones.dark_reference_hz} Hz` : "—";
  $("#receipt-hash").textContent = lastReceipt.sha256;
  renderRack(radices, [...selected, digits.at(-1)], states);
  renderEvents(events);
  drawScope(tones);
}
function playTones() {
  if (!lastTones) return;
  audioContext ||= new AudioContext();
  const start = audioContext.currentTime;
  [lastTones.route_hz, lastTones.check_hz, lastTones.dark_reference_hz].forEach((frequency, index) => {
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = index === 2 ? "sine" : "triangle";
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(index === 2 ? 0.035 : 0.08, start + .02);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + .8);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(start);
    oscillator.stop(start + .82);
  });
}
function wavBytes(tones) {
  const rate = 48000, seconds = 1, samples = rate * seconds, bytes = 44 + samples * 2;
  const buffer = new ArrayBuffer(bytes), view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, bytes - 8, true); write(8, "WAVE"); write(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i += 1) {
    const t = i / rate;
    const envelope = Math.min(1, i / 600) * Math.min(1, (samples - i) / 1200);
    const value = envelope * (
      Math.sin(2 * Math.PI * tones.route_hz * t) * .35
      + Math.sin(2 * Math.PI * tones.check_hz * t) * .25
      + Math.sin(2 * Math.PI * tones.dark_reference_hz * t) * .12
    );
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, value)) * 32767, true);
  }
  return new Blob([buffer], {type: "audio/wav"});
}
function saveBlob(blob, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 500);
}
$("#route").addEventListener("click", route);
$("#reset").addEventListener("click", () => {
  lastReceipt = null; lastTones = null; operatorEvents = [];
  $("#state").textContent = "HOME"; $("#linefinder").textContent = "—";
  $("#route-path").textContent = "—"; $("#verify").textContent = "—";
  $("#receipt-hash").textContent = "no receipt"; eventList.replaceChildren();
  renderRack(parseCsv("#radices")); drawScope(null);
});
$("#mode").addEventListener("change", updateDesk);
document.querySelectorAll("[data-op]").forEach((button) => button.addEventListener("click", () => operatorAction(button.dataset.op)));
$("#hear").addEventListener("click", playTones);
$("#wav").addEventListener("click", () => lastTones && saveBlob(wavBytes(lastTones), "qec-strowger-tones.wav"));
$("#download").addEventListener("click", () => lastReceipt && saveBlob(new Blob([JSON.stringify(lastReceipt, null, 2)], {type: "application/json"}), "qec-strowger-receipt.json"));
updateDesk();
renderRack(parseCsv("#radices"));
drawScope(null);
