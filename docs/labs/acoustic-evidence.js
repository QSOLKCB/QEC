// SPDX-License-Identifier: MPL-2.0
'use strict';
// Read-only presentation. Never produces or validates canonical QEC evidence.
const MAX_JSON_BYTES = 8 * 1024 * 1024;
const MAX_STAGE_BYTES = 1024 * 1024;
const MAX_SAMPLES = 73728;
const MAX_TRIALS = 38;
const STAGES = ['original', 'observed', 'filtered'];
const NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);
let entries = [], entryIndex = 0, methodIndex = 0, attachments = {};
let loadGeneration = 0, stageGeneration = {original: 0, observed: 0, filtered: 0};
let audioGeneration = 0, audio = null;

function requireShape(condition, message = 'Unsupported or malformed acoustic structure.') {
  if (!condition) throw new Error(message);
}
function object(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function integer(value, low, high) { return Number.isSafeInteger(value) && value >= low && value <= high; }
function bool(value) { return typeof value === 'boolean'; }
function label(value, maximum = 240) { return typeof value === 'string' && value.length <= maximum; }
function digest(value) { return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value); }
function array(value, maximum) { return Array.isArray(value) && value.length <= maximum; }
function symbols(value, maximum = 32) {
  return array(value, maximum) && value.every(x => x === null || integer(x, 0, 3));
}
function reasons(value) { return array(value, 32) && value.every(x => label(x)); }
function seal(value, name) {
  requireShape(object(value) && value.schema === 'qec.acoustic-' + name + '.v1' &&
    ['177.0', '177.2', '177.3', '177.5'].includes(value.contract_version) && digest(value.sha256));
  return value;
}
function parseJSON(text, maximum = MAX_JSON_BYTES) {
  requireShape(typeof text === 'string' && text.length <= maximum, 'JSON exceeds the bounded file size.');
  requireShape(new TextEncoder().encode(text).length <= maximum, 'JSON exceeds the bounded file size.');
  // Bound nesting before parsing as well as after it. Quoted brackets are data.
  let depth = 0, quoted = false, escaped = false;
  for (const char of text) {
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '[' || char === '{') {
      requireShape(++depth <= 32, 'JSON nesting exceeds 32 levels.');
    } else if (char === ']' || char === '}') depth--;
  }
  let value;
  try { value = JSON.parse(text); } catch (_) { throw new Error('Could not parse JSON.'); }
  let nodes = 0;
  function visit(item, nesting) {
    requireShape(++nodes <= 200000 && nesting <= 32, 'JSON structure exceeds inspection bounds.');
    if (typeof item === 'string') requireShape(item.length <= 4 * MAX_SAMPLES, 'JSON string exceeds inspection bounds.');
    else if (typeof item === 'number') requireShape(Number.isFinite(item), 'Non-finite JSON number.');
    else if (Array.isArray(item)) {
      requireShape(item.length <= 128, 'JSON array exceeds inspection bounds.');
      item.forEach(x => visit(x, nesting + 1));
    } else if (object(item)) {
      const keys = Object.keys(item);
      requireShape(keys.length <= 128, 'JSON object exceeds inspection bounds.');
      for (const key of keys) {
        requireShape(key.length <= 128 && !['__proto__', 'constructor', 'prototype'].includes(key), 'Unsupported JSON key.');
        visit(item[key], nesting + 1);
      }
    }
  }
  visit(value, 0);
  requireShape(object(value));
  return value;
}
function parsePCM(value) {
  seal(value, 'pcm');
  requireShape([24000, 48000].includes(value.sample_rate) && integer(value.sample_count, 0, MAX_SAMPLES) &&
    value.layout === 'mono_signed_16bit_little_endian' && digest(value.pcm_sha256) &&
    typeof value.pcm_hex === 'string' && value.pcm_hex.length === value.sample_count * 4 &&
    /^[0-9a-f]*$/.test(value.pcm_hex), 'Expected bounded mono PCM16 JSON at 24 or 48 kHz.');
  const samples = new Int16Array(value.sample_count);
  for (let n = 0; n < samples.length; n++) {
    const at = 4 * n;
    samples[n] = parseInt(value.pcm_hex.slice(at, at + 2), 16) + 256 * parseInt(value.pcm_hex.slice(at + 2, at + 4), 16);
  }
  return {samples, sampleRate: value.sample_rate, claimedHash: value.pcm_sha256, format: 'PCM JSON'};
}
function parseWAV(buffer) {
  requireShape(buffer instanceof ArrayBuffer && buffer.byteLength >= 44 && buffer.byteLength <= 44 + 2 * MAX_SAMPLES,
    'WAV exceeds the bounded PCM16 size or is truncated.');
  const view = new DataView(buffer), bytes = new Uint8Array(buffer);
  const tag = at => String.fromCharCode(...bytes.slice(at, at + 4));
  requireShape(tag(0) === 'RIFF' && tag(8) === 'WAVE' && view.getUint32(4, true) === bytes.length - 8,
    'Expected a complete RIFF/WAVE file.');
  let position = 12, format = null, data = null, chunks = 0;
  while (position < bytes.length) {
    requireShape(++chunks <= 64 && position + 8 <= bytes.length, 'Malformed or excessive WAV chunks.');
    const kind = tag(position), size = view.getUint32(position + 4, true), start = position + 8;
    requireShape(start + size + (size % 2) <= bytes.length, 'Truncated WAV chunk.');
    if (kind === 'fmt ') {
      requireShape(format === null && (size === 16 || size === 18), 'Unsupported or duplicate WAV format.');
      requireShape(view.getUint16(start, true) === 1 && view.getUint16(start + 2, true) === 1 &&
        view.getUint16(start + 14, true) === 16 && view.getUint16(start + 12, true) === 2,
        'WAV must be uncompressed mono PCM16.');
      const rate = view.getUint32(start + 4, true);
      requireShape([24000, 48000].includes(rate) && view.getUint32(start + 8, true) === rate * 2 &&
        (size === 16 || view.getUint16(start + 16, true) === 0), 'Unsupported WAV sample rate or byte rate.');
      format = rate;
    } else if (kind === 'data') {
      requireShape(data === null && size % 2 === 0 && size / 2 <= MAX_SAMPLES, 'Invalid or duplicate WAV data.');
      data = {start, size};
    }
    position = start + size + (size % 2);
  }
  requireShape(format !== null && data !== null && position === bytes.length, 'Missing WAV format or samples.');
  const samples = new Int16Array(data.size / 2);
  for (let n = 0; n < samples.length; n++) samples[n] = view.getInt16(data.start + 2 * n, true);
  return {samples, sampleRate: format, claimedHash: null, format: 'WAV PCM16'};
}
function observation(value) {
  seal(value, 'harmonic-observation');
  requireShape(bool(value.accepted) && integer(value.symbol_count, 1, 32) && reasons(value.reasons) &&
    (value.estimated_syndrome === null || symbols(value.estimated_syndrome)) && array(value.per_symbol, 32));
  requireShape(value.accepted ? symbols(value.estimated_syndrome) && value.estimated_syndrome.length === value.symbol_count : value.estimated_syndrome === null);
  for (const row of value.per_symbol) {
    requireShape(object(row) && integer(row.index, 0, 31) && bool(row.accepted) &&
      (row.estimated_symbol === null || integer(row.estimated_symbol, 0, 3)) && array(row.blocks, 3));
    for (const block of row.blocks) requireShape(object(block) && bool(block.accepted) && reasons(block.reasons) &&
      (block.candidate_symbol === null || integer(block.candidate_symbol, 0, 3)));
  }
  return value;
}
function decision(value) {
  requireShape(object(value) && bool(value.attempted) && bool(value.accepted) &&
    (value.reason === undefined || label(value.reason)));
  return value;
}
function bridge(value) {
  seal(value, 'syndrome-bridge');
  requireShape(bool(value.candidate_accepted) && bool(value.candidate_false_accept) && bool(value.verified_recovery) &&
    symbols(value.measured_syndrome, 4) && (value.estimated_syndrome === null || symbols(value.estimated_syndrome, 4)) &&
    digest(value.expected_reference_sha256));
  observation(value.receiver_observation);
  decision(value.decoder_decision);
  return value;
}
function source(value) {
  seal(value, 'comparative-source');
  requireShape(label(value.case_name) && [3, 4].includes(value.alphabet_size) && integer(value.seed, 0, 0xffffffff));
  seal(value.acoustic_source, 'source');
  seal(value.measured_reference, 'measured-syndrome-reference');
  requireShape(symbols(value.measured_reference.measured_syndrome, 4) && value.measured_reference.measured_syndrome.length === 4);
  parsePCM(value.acoustic_source.encoded_sound);
  return value;
}
function method(value) {
  seal(value, 'comparative-method');
  requireShape(['unfiltered', 'cancellation_only', 'fec_only', 'combined', 'robust_median_period',
    'direct_syndrome_reference', 'exact_clean_reference'].includes(value.method) && symbols(value.predicted_symbols, 4) &&
    object(value.metrics) && object(value.resources) && object(value.logical_outcome) && bool(value.logical_outcome.success));
  decision(value.decoder_decision);
  for (const name of ['raw_receiver_accepted_frames', 'raw_receiver_false_accept_frames',
    'independently_verified_frames', 'decoder_attempted_frames', 'decoder_accepted_frames', 'exact_model_success_frames']) {
    requireShape(integer(value.metrics[name], 0, 1));
  }
  requireShape(bool(value.metrics.raw_receiver_metrics_applicable));
  if (value.receiver_reasons !== undefined) requireShape(reasons(value.receiver_reasons));
  return value;
}
function baseEntry(value, name) {
  return {artifact: value, title: name, source: null, stages: {}, candidate: null, rejected: null, gate: null, methods: []};
}
function inspectEntry(value) {
  requireShape(object(value));
  let entry = baseEntry(value, value.schema);
  switch (value.schema) {
    case 'qec.acoustic-comparative-trial.v1':
      seal(value, 'comparative-trial');
      entry.source = source(value.source);
      requireShape(array(value.methods, 7) && value.methods.length > 0);
      entry.methods = value.methods.map(method);
      requireShape(new Set(entry.methods.map(x => x.method)).size === entry.methods.length);
      for (const role of STAGES) entry.stages[role] = parsePCM(value[role]);
      break;
    case 'qec.acoustic-lab-demonstration.v1':
      seal(value, 'lab-demonstration');
      entry.source = source(value.source);
      requireShape(object(value.roles));
      for (const role of STAGES) entry.stages[role] = parsePCM(value.roles[role]);
      entry.candidate = observation(value.roles.candidate);
      entry.rejected = observation(value.roles.rejected);
      requireShape(entry.rejected.accepted === false);
      entry.gate = bridge(value.roles.verified);
      break;
    case 'qec.acoustic-comparative-source.v1':
      entry.source = source(value);
      entry.stages.original = parsePCM(value.acoustic_source.encoded_sound);
      break;
    case 'qec.acoustic-source.v1':
      seal(value, 'source');
      requireShape(symbols(value.measured_symbols));
      entry.stages.original = parsePCM(value.encoded_sound);
      break;
    case 'qec.acoustic-noise-observation.v1':
      seal(value, 'noise-observation');
      entry.stages.original = parsePCM(value.original);
      entry.stages.observed = parsePCM(value.observed);
      break;
    case 'qec.acoustic-cancellation-receipt.v1':
      seal(value, 'cancellation-receipt');
      entry.stages.observed = parsePCM(value.observed);
      entry.stages.filtered = parsePCM(value.filtered);
      break;
    case 'qec.acoustic-harmonic-observation.v1':
      observation(value);
      entry[value.accepted ? 'candidate' : 'rejected'] = value;
      break;
    case 'qec.acoustic-syndrome-bridge.v1':
      entry.gate = bridge(value);
      entry[value.receiver_observation.accepted ? 'candidate' : 'rejected'] = value.receiver_observation;
      break;
    default: throw new Error('Unsupported acoustic schema. Attach standalone PCM JSON using a stage picker.');
  }
  if (entry.source) entry.title = 'q' + entry.source.alphabet_size + ' · ' + entry.source.case_name;
  return entry;
}
function inspectEvidence(value) {
  if (value.schema === 'qec.acoustic-comparative-battery.v1' || value.schema === 'qec.acoustic-comparative-battery-source.v1') {
    seal(value, value.schema.includes('battery-source') ? 'comparative-battery-source' : 'comparative-battery');
    requireShape(array(value.trials, MAX_TRIALS) && value.trials.length > 0, 'Battery must contain 1–38 bounded trials.');
    const expected = value.schema.includes('battery-source') ? 'qec.acoustic-comparative-source.v1' : 'qec.acoustic-comparative-trial.v1';
    requireShape(value.trials.every(x => object(x) && x.schema === expected));
    return value.trials.map(inspectEntry);
  }
  return [inspectEntry(value)];
}
function summary(value) { return JSON.stringify(value, null, 2); }
function observationSummary(value) {
  return {imported_receiver_accepted: value.accepted, estimated_syndrome: value.estimated_syndrome,
    reasons: value.reasons, per_symbol: value.per_symbol.map(row => ({index: row.index,
      accepted: row.accepted, estimated_symbol: row.estimated_symbol,
      blocks: row.blocks.map(block => ({candidate_symbol: block.candidate_symbol, accepted: block.accepted, reasons: block.reasons}))})),
    interpretation: 'Fallible receiver claim; no independent truth established by this viewer.'};
}
function selectedEntry() { return entries[entryIndex] || null; }
function stagePCM(role) { return attachments[role]?.pcm || selectedEntry()?.stages[role] || null; }
function equalPCM(a, b) {
  return a && b && a.sampleRate === b.sampleRate && a.samples.length === b.samples.length && a.samples.every((x, n) => x === b.samples[n]);
}
function drawWave(role, pcm) {
  const svg = $(role + '-wave');
  svg.replaceChildren();
  const axis = document.createElementNS(NS, 'line');
  for (const [key, val] of Object.entries({x1: 0, x2: 800, y1: 80, y2: 80})) axis.setAttribute(key, val);
  svg.append(axis);
  if (!pcm || !pcm.samples.length) return;
  const samples = pcm.samples, columns = Math.min(800, samples.length), segments = [];
  for (let i = 0; i < columns; i++) {
    const start = Math.floor(i * samples.length / columns), end = Math.floor((i + 1) * samples.length / columns);
    let minimum = 32767, maximum = -32768;
    for (let j = start; j < end; j++) { minimum = Math.min(minimum, samples[j]); maximum = Math.max(maximum, samples[j]); }
    const x = ((i + .5) * 800 / columns).toFixed(2);
    const y = sample => (80 - sample / 32768 * 72).toFixed(2);
    segments.push('M' + x + ',' + y(minimum) + 'L' + x + ',' + y(maximum));
    if (end - start === 1 && i + 1 < columns) segments.push('M' + x + ',' + y(samples[start]) + 'L' + (((i + 1.5) * 800 / columns).toFixed(2)) + ',' + y(samples[end]));
  }
  const path = document.createElementNS(NS, 'path');
  path.setAttribute('d', segments.join(' '));
  svg.append(path);
}
function populate(select, values, selected) {
  select.replaceChildren();
  for (const [index, text] of values.entries()) {
    const option = document.createElement('option'); option.value = String(index); option.textContent = text; select.append(option);
  }
  select.value = String(selected); select.disabled = values.length < 2;
}
function render() {
  const entry = selectedEntry(), chosen = entry?.methods[methodIndex];
  populate($('trial-select'), entries.map(x => x.title), entryIndex);
  populate($('method-select'), entry?.methods.map(x => x.method) || [], methodIndex);
  $('evidence-meta').textContent = entry ? 'Imported schema: ' + entry.artifact.schema + '\nClaimed artifact SHA-256 (not verified): ' + entry.artifact.sha256 +
    (entry.source ? '\nCaller-declared reference SHA-256: ' + entry.source.measured_reference.sha256 : '') : '';
  for (const role of STAGES) {
    const pcm = stagePCM(role), attachment = attachments[role];
    drawWave(role, pcm);
    $(role + '-play').disabled = !pcm || !pcm.samples.length || audio !== null;
    $(role + '-info').textContent = pcm ? pcm.samples.length + ' samples · ' + pcm.sampleRate + ' Hz · ' + (pcm.samples.length / pcm.sampleRate).toFixed(6) + ' s\n' +
      (attachment ? 'Manual attachment: ' + attachment.name : 'Embedded in imported evidence') + '\n' + pcm.format +
      (pcm.claimedHash ? '\nClaimed PCM hash (unverified): ' + pcm.claimedHash : '\nNo claimed JSON hash') : 'No supplied PCM. Nothing generated.';
  }
  const manual = STAGES.filter(role => attachments[role]);
  $('association').textContent = manual.length ? manual.map(role => role + ': ' +
    (equalPCM(attachments[role].pcm, entry?.stages[role]) ? 'exact sample match with embedded PCM; replay still required.' :
      'manual attachment has no matching embedded PCM. Imported receiver/gate claims do not establish its outcome.')).join('\n') :
    'Waveforms come only from the imported artifacts. Their relationship to receiver/gate claims requires independent Python replay.';
  $('candidate').textContent = entry?.candidate ? summary(observationSummary(entry.candidate)) : 'No receiver candidate supplied.';
  $('rejected').textContent = entry?.rejected ? summary(observationSummary(entry.rejected)) : 'No rejected observation supplied.';
  $('verified').textContent = entry?.gate ? summary({imported_candidate_accepted: entry.gate.candidate_accepted,
    imported_candidate_false_accept: entry.gate.candidate_false_accept, imported_verified_recovery: entry.gate.verified_recovery,
    measured_syndrome: entry.gate.measured_syndrome, estimated_syndrome: entry.gate.estimated_syndrome,
    expected_reference_sha256: entry.gate.expected_reference_sha256, decoder_decision: entry.gate.decoder_decision,
    interpretation: 'Caller-declared fixture authority. All gate/correction claims remain unverified here.'}) : 'No reference-gated decision supplied.';
  $('metrics').textContent = 'No comparative method supplied.';
  if (chosen) {
    const m = chosen.metrics;
    $('candidate').textContent = summary({method: chosen.method, imported_raw_receiver_applicable: m.raw_receiver_metrics_applicable,
      imported_raw_receiver_accepted: m.raw_receiver_metrics_applicable ? Boolean(m.raw_receiver_accepted_frames) : null,
      predicted_symbols: chosen.predicted_symbols,
      interpretation: chosen.method === 'direct_syndrome_reference' ? 'Privileged direct-syndrome control; no acoustic receiver.' :
        chosen.method === 'exact_clean_reference' ? 'Privileged clean-PCM control; not observed-channel recovery.' : 'Receiver claim before independent-reference gate.'});
    $('rejected').textContent = summary({receiver_reasons: chosen.receiver_reasons || [],
      imported_raw_false_accept_frames: m.raw_receiver_false_accept_frames,
      decoder_reason: chosen.decoder_decision.reason || 'Not supplied by this control',
      interpretation: 'A raw false accept is a wrong receiver word even when the later reference gate vetoes it.'});
    $('verified').textContent = summary({imported_independently_verified_frames: m.independently_verified_frames,
      measured_reference: entry.source?.measured_reference.measured_syndrome,
      decoder_decision: chosen.decoder_decision, logical_outcome: chosen.logical_outcome,
      interpretation: 'Imported caller-declared fixture gate and finite-model claims; Python replay required.'});
    $('metrics').textContent = summary({metrics: chosen.metrics, resources: chosen.resources});
  } else if (entry?.gate && !entry.rejected) {
    $('rejected').textContent = summary({imported_candidate_false_accept: entry.gate.candidate_false_accept,
      receiver_reasons: entry.gate.receiver_observation.reasons, decoder_reason: entry.gate.decoder_decision.reason});
  }
}
function closeAudio(context) {
  if (!context) return;
  try { Promise.resolve(context.close()).catch(() => {}); } catch (_) { /* best effort release */ }
}
function stopPlayback(message = 'Audio stopped.') {
  audioGeneration++;
  const previous = audio; audio = null;
  if (previous) { try { previous.source?.stop(); } catch (_) { /* already stopped */ } closeAudio(previous.context); }
  $('stop').disabled = true;
  $('audio-status').textContent = message;
  render();
}
async function play(role) {
  const pcm = stagePCM(role);
  if (!pcm || !pcm.samples.length) return;
  stopPlayback('Preparing actual PCM playback…');
  const token = audioGeneration;
  let context;
  try {
    const Audio = window.AudioContext || window.webkitAudioContext;
    requireShape(typeof Audio === 'function', 'Web Audio is unavailable in this browser.');
    context = new Audio();
    audio = {context, source: null};
    $('stop').disabled = false; render();
    await context.resume();
    if (token !== audioGeneration) { closeAudio(context); return; }
    const buffer = context.createBuffer(1, pcm.samples.length, pcm.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let n = 0; n < pcm.samples.length; n++) channel[n] = pcm.samples[n] / 32768;
    const source = context.createBufferSource(), gain = context.createGain();
    source.buffer = buffer; gain.gain.value = .25;
    source.connect(gain); gain.connect(context.destination); audio.source = source;
    source.onended = () => { if (token === audioGeneration) stopPlayback('Playback finished.'); };
    source.start();
    $('audio-status').textContent = 'Playing actual ' + role + ' PCM at 25% gain.';
  } catch (error) {
    if (token !== audioGeneration) { closeAudio(context); return; }
    stopPlayback('Could not play PCM: ' + error.message);
  }
}
function clearAttachments() {
  attachments = {};
  for (const role of STAGES) { stageGeneration[role]++; $(role + '-file').value = ''; }
  $('stage-error').textContent = '';
}
async function loadEvidence(file) {
  if (!file) return;
  const token = ++loadGeneration;
  stopPlayback(); clearAttachments(); entries = []; entryIndex = 0; methodIndex = 0;
  render(); $('import-error').textContent = ''; $('import-status').textContent = 'Reading local JSON…';
  try {
    requireShape(integer(file.size, 1, MAX_JSON_BYTES), 'Evidence file must be 1 byte–8 MiB.');
    const text = await file.text();
    if (token !== loadGeneration) return;
    const parsed = inspectEvidence(parseJSON(text));
    if (token !== loadGeneration) return;
    entries = parsed; render();
    $('import-status').textContent = 'Loaded ' + entries.length + ' bounded item(s). Imported claims remain unverified.';
  } catch (error) {
    if (token !== loadGeneration) return;
    $('import-error').textContent = 'Could not load evidence: ' + error.message;
    $('import-status').textContent = 'No evidence loaded.'; render();
  }
}
async function loadStage(role, file) {
  if (!file || !STAGES.includes(role)) return;
  const token = ++stageGeneration[role], evidenceToken = loadGeneration;
  stopPlayback(); delete attachments[role];
  // A failed replacement must not silently fall back to an older attachment.
  render(); $('stage-error').textContent = '';
  try {
    requireShape(integer(file.size, 1, MAX_STAGE_BYTES), 'Stage file must be 1 byte–1 MiB.');
    requireShape(label(file.name, 256), 'Stage filename is too long.');
    const buffer = await file.arrayBuffer();
    if (token !== stageGeneration[role] || evidenceToken !== loadGeneration) return;
    requireShape(buffer instanceof ArrayBuffer && buffer.byteLength <= MAX_STAGE_BYTES && buffer.byteLength === file.size, 'Stage file size changed or exceeds bounds.');
    const bytes = new Uint8Array(buffer);
    const isWAV = bytes.length >= 4 && bytes[0] === 82 && bytes[1] === 73 && bytes[2] === 70 && bytes[3] === 70;
    const pcm = isWAV ? parseWAV(buffer) : parsePCM(parseJSON(new TextDecoder('utf-8', {fatal: true}).decode(buffer), MAX_STAGE_BYTES));
    attachments[role] = {pcm, name: file.name}; render();
  } catch (error) {
    if (token !== stageGeneration[role] || evidenceToken !== loadGeneration) return;
    $('stage-error').textContent = 'Could not load ' + role + ': ' + error.message; render();
  }
}
function reset() {
  loadGeneration++; clearAttachments(); entries = []; entryIndex = 0; methodIndex = 0;
  $('evidence-file').value = ''; $('import-error').textContent = ''; $('import-status').textContent = 'No evidence loaded.';
  stopPlayback('Audio is off. Play starts only when clicked.');
}
$('evidence-file').addEventListener('change', event => loadEvidence(event.target.files[0]));
$('reset').addEventListener('click', reset);
$('stop').addEventListener('click', () => stopPlayback());
$('trial-select').addEventListener('change', () => {
  const index = Number($('trial-select').value);
  if (!integer(index, 0, entries.length - 1)) return;
  loadGeneration++; clearAttachments(); entryIndex = index; methodIndex = 0; stopPlayback();
});
$('method-select').addEventListener('change', () => {
  const index = Number($('method-select').value);
  if (!integer(index, 0, (selectedEntry()?.methods.length || 0) - 1)) return;
  methodIndex = index; stopPlayback();
});
for (const role of STAGES) {
  $(role + '-file').addEventListener('change', event => loadStage(role, event.target.files[0]));
  $(role + '-play').addEventListener('click', () => play(role));
}
window.addEventListener('pagehide', reset);
reset();
