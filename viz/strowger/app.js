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
  return $(id).value.split(",").map((part) => Number(part.trim()));
}

function validRadices(radices) {
  if (radices.length < 3) return false;
  return radices.every((radix, index) => (
    Number.isInteger(radix)
    && radix >= 2
    && radix <= (index === radices.length - 1 ? 4096 : 256)
  ));
}

function hash32(text) {
  let value = 2166136261;
  for (const char of text) {
    value ^= char.charCodeAt(0);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function demoDigest(text) {
  const parts = [];
  let seed = text;
  for (let index = 0; index < 8; index += 1) {
    const value = hash32(`${seed}|${index}`).toString(16).padStart(8, "0");
    parts.push(value);
    seed = value;
  }
  return parts.join("");
}

function pulseCount(digit, radix) {
  return digit === 0 ? radix : digit;
}

function firstFree(states) {
  return states.findIndex((state) => state === "free");
}

function deriveTones(identity) {
  const primary = hash32(identity);
  const check = hash32(identity.split("").reverse().join(""));
  return {
    route_hz: 320 + (primary % 1200),
    check_hz: 320 + (check % 1200),
    dark_reference_hz: 90
  };
}

function appendEvent(events, device, action, details = {}) {
  const previous = events.length ? events[events.length - 1].event_digest : null;
  const event = {
    sequence: events.length,
    tick: events.length,
    device,
    action,
    details,
    previous_event_digest: previous
  };
  event.event_digest = demoDigest(JSON.stringify(event));
  events.push(event);
}

function selectorCard(name, radix, trunkCount, selected, level, states) {
  const card = document.createElement("article");
  card.className = "selector";
  card.innerHTML = `<h3>${name.toUpperCase()}</h3><div class="shaft"></div><div class="wiper"></div><div class="contacts"></div><div class="reading">level ${level} · contact ${selected < 0 ? "—" : selected}</div>`;
  const contacts = card.querySelector(".contacts");
  for (let index = 0; index < trunkCount; index += 1) {
    const dot = document.createElement("i");
    const angle = (Math.PI * 2 * index / trunkCount) - Math.PI / 2;
    dot.className = `contact ${states[index] || "free"} ${index === selected ? "active" : ""}`;
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
  const labels = radices.slice(0, -2).map((_, index) => `Selector ${index + 1}`);
  labels.push("Connector");
  labels.forEach((name, index) => {
    const radix = index < radices.length - 2 ? radices[index] : radices.at(-1);
    const level = index < radices.length - 2
      ? (lastReceipt?.digits[index] ?? 0)
      : (lastReceipt?.digits.at(-2) ?? 0);
    rack.append(selectorCard(
      name,
      radix,
      index === labels.length - 1 ? radices.at(-1) : 6,
      selected[index] ?? -1,
      level,
      states[index] || []
    ));
  });
}

function renderEvents(events) {
  eventList.replaceChildren();
  for (const event of events) {
    const item = document.createElement("li");
    const summary = document.createElement("b");
    summary.textContent = `${String(event.sequence).padStart(3, "0")} · ${event.device}`;
    const details = document.createTextNode(
      `${event.action} ${JSON.stringify(event.details)}`
    );
    item.append(summary, document.createElement("br"), details);
    eventList.append(item);
  }
  eventList.scrollTop = eventList.scrollHeight;
}

function drawScope(tones) {
  const canvas = $("#scope");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#2d332b";
  context.lineWidth = 1;
  for (let y = 30; y < height; y += 30) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  if (!tones) return;
  context.strokeStyle = "#c39a4a";
  context.lineWidth = 2;
  context.beginPath();
  for (let x = 0; x < width; x += 1) {
    const time = x / width;
    const y = height / 2
      + Math.sin(time * tones.route_hz * 0.08) * 26
      + Math.sin(time * tones.check_hz * 0.08) * 15
      + Math.sin(time * tones.dark_reference_hz * 0.08) * 8;
    if (x === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }
  context.stroke();
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

function refreshDemoDigest() {
  if (!lastReceipt) return;
  const unsigned = {...lastReceipt};
  delete unsigned.digest;
  lastReceipt.digest = demoDigest(JSON.stringify(unsigned));
  $("#receipt-hash").textContent = `demo:${lastReceipt.digest}`;
}

function operatorAction(action) {
  if (!operatorEnabled()) return;
  const command = {
    action,
    operator_id: "local-console",
    target: action === "quarantine" ? "selector-0:0" : "exchange",
    reason: "operator-desk action"
  };
  operatorEvents.push(command);
  if (lastReceipt) {
    appendEvent(lastReceipt.events, "operator-desk", `operator_${action}`, command);
    lastReceipt.operator_commands.push(command);
    refreshDemoDigest();
    renderEvents(lastReceipt.events);
  }
}

function route() {
  const digits = parseCsv("#digits");
  const radices = parseCsv("#radices");
  if (
    digits.length !== radices.length
    || !validRadices(radices)
    || digits.some((digit, index) => (
      !Number.isInteger(digit) || digit < 0 || digit >= radices[index]
    ))
  ) {
    $("#state").textContent = "INVALID REQUEST";
    return;
  }

  const events = [];
  appendEvent(events, "linefinder-0", "seize_first_free_linefinder", {
    request_id: "browser-call"
  });
  const selected = [];
  const states = [];
  let outcome = "committed";
  const fault = $("#fault").value;

  for (let stage = 0; stage < radices.length - 2; stage += 1) {
    const state = Array(6).fill("free");
    if (fault === "busy" && stage === 0) {
      state[0] = "busy";
      state[1] = "busy";
    }
    if (
      operatorEvents.some((event) => event.action === "quarantine")
      && stage === 0
    ) {
      state[0] = "quarantined";
    }
    states.push(state);
    let observed = pulseCount(digits[stage], radices[stage]);
    if (fault === "missed" && stage === 0) observed -= 1;
    if (fault === "duplicate" && stage === 0) observed += 1;
    appendEvent(events, `selector-${stage}`, "receive_digit", {
      digit: digits[stage],
      expected_pulses: pulseCount(digits[stage], radices[stage]),
      observed_pulses: observed
    });
    if (
      observed !== pulseCount(digits[stage], radices[stage])
      || (fault === "stuck" && stage === 1)
    ) {
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
    appendEvent(events, `selector-${stage}`, "first_free_trunk_selected", {
      level: digits[stage],
      contact: trunk
    });
  }

  let tones = null;
  let verified = false;
  if (outcome === "committed") {
    appendEvent(events, "connector", "two_axis_destination_selected", {
      vertical: digits.at(-2),
      rotary: digits.at(-1),
      destination: $("#destination").value
    });
    tones = deriveTones(`${digits}|${selected}|${$("#destination").value}`);
    const observed = {...tones};
    if (fault === "tone") observed.route_hz += 7;
    verified = JSON.stringify(tones) === JSON.stringify(observed);
    appendEvent(events, "tone-verifier", "verify_route_tones", {
      expected: tones,
      observed,
      verified
    });
    if (!verified) outcome = "tone_mismatch";
    appendEvent(
      events,
      "exchange",
      verified ? "commit_verified_route" : "reject_tone_mismatch",
      {destination: $("#destination").value}
    );
  }

  lastReceipt = {
    schema: "qec.strowger-browser-demonstration.v1",
    qec_version: "170.3.0",
    browser_demo_only: true,
    canonical_receipt: false,
    digest_algorithm: "fnv1a-derived-demo-digest-v1",
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
      operator_may_force_accept: false,
      browser_demo_only: true,
      canonical_receipt: false
    }
  };
  refreshDemoDigest();
  lastTones = tones;
  $("#state").textContent = outcome.toUpperCase();
  $("#linefinder").textContent = "0";
  $("#route-path").textContent = selected.length ? selected.join(" → ") : "—";
  $("#verify").textContent = tones ? (verified ? "MATCH" : "REJECT") : "—";
  $("#route-tone").textContent = tones ? `${tones.route_hz} Hz` : "—";
  $("#check-tone").textContent = tones ? `${tones.check_hz} Hz` : "—";
  $("#dark-tone").textContent = tones ? `${tones.dark_reference_hz} Hz` : "—";
  renderRack(radices, [...selected, digits.at(-1)], states);
  renderEvents(events);
  drawScope(tones);
}

function playTones() {
  if (!lastTones) return;
  audioContext ||= new AudioContext();
  const start = audioContext.currentTime;
  [
    lastTones.route_hz,
    lastTones.check_hz,
    lastTones.dark_reference_hz
  ].forEach((frequency, index) => {
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = index === 2 ? "sine" : "triangle";
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(index === 2 ? 0.035 : 0.08, start + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.8);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(start);
    oscillator.stop(start + 0.82);
  });
}

function wavBytes(tones) {
  const rate = 48000;
  const seconds = 1;
  const samples = rate * seconds;
  const bytes = 44 + samples * 2;
  const buffer = new ArrayBuffer(bytes);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((character, index) => {
    view.setUint8(offset + index, character.charCodeAt(0));
  });
  write(0, "RIFF");
  view.setUint32(4, bytes - 8, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples * 2, true);
  for (let index = 0; index < samples; index += 1) {
    const time = index / rate;
    const envelope = Math.min(1, index / 600) * Math.min(1, (samples - index) / 1200);
    const value = envelope * (
      Math.sin(2 * Math.PI * tones.route_hz * time) * 0.35
      + Math.sin(2 * Math.PI * tones.check_hz * time) * 0.25
      + Math.sin(2 * Math.PI * tones.dark_reference_hz * time) * 0.12
    );
    view.setInt16(
      44 + index * 2,
      Math.max(-1, Math.min(1, value)) * 32767,
      true
    );
  }
  return new Blob([buffer], {type: "audio/wav"});
}

function saveBlob(blob, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 500);
}

$("#route").addEventListener("click", route);
$("#reset").addEventListener("click", () => {
  lastReceipt = null;
  lastTones = null;
  operatorEvents = [];
  $("#state").textContent = "HOME";
  $("#linefinder").textContent = "—";
  $("#route-path").textContent = "—";
  $("#verify").textContent = "—";
  $("#receipt-hash").textContent = "no receipt";
  eventList.replaceChildren();
  renderRack(parseCsv("#radices"));
  drawScope(null);
});
$("#mode").addEventListener("change", updateDesk);
document.querySelectorAll("[data-op]").forEach((button) => {
  button.addEventListener("click", () => operatorAction(button.dataset.op));
});
$("#hear").addEventListener("click", playTones);
$("#wav").addEventListener("click", () => {
  if (lastTones) saveBlob(wavBytes(lastTones), "qec-strowger-tones.wav");
});
$("#download").addEventListener("click", () => {
  if (lastReceipt) {
    saveBlob(
      new Blob([JSON.stringify(lastReceipt, null, 2)], {type: "application/json"}),
      "qec-strowger-browser-demonstration.json"
    );
  }
});

updateDesk();
renderRack(parseCsv("#radices"));
drawScope(null);