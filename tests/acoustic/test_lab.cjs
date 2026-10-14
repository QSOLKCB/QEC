// SPDX-License-Identifier: MPL-2.0
'use strict';
// Dependency-free DOM/audio stubs. This does not claim rendered-browser/audio QA.
// Python test_lab.py supplies current, actual native artifacts in a temp file.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'docs/labs/acoustic-evidence.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'docs/labs/acoustic-evidence.js'), 'utf8');
assert.ok(process.argv[2], 'Run pytest tests/acoustic/test_lab.py to supply native artifacts.');
const fixtures = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.listeners = {}; this.attrs = {};
    this.value = ''; this._text = ''; this.disabled = false; this.files = [];
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(x => x.textContent).join(''); }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  click() { return this.listeners.click?.({target: this}); }
  change() { return this.listeners.change?.({target: this}); }
}
const ids = {};
for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) {
  assert.ok(!ids[id], 'Duplicate id'); ids[id] = new Element();
}
let audioCount = 0;
const windowEvents = {};
const sandbox = vm.createContext({
  document: {getElementById: id => { assert.ok(ids[id], id); return ids[id]; },
    createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)},
  window: {AudioContext: class {constructor() { audioCount++; throw new Error('Unexpected audio'); }},
    addEventListener: (name, fn) => { windowEvents[name] = fn; }},
  TextEncoder, TextDecoder, ArrayBuffer, Uint8Array, Int16Array, DataView, console,
});
vm.runInContext(script + '\nglobalThis.api = {parseJSON,parsePCM,parseWAV,inspectEvidence,loadEvidence,loadStage,play,reset,stagePCM,' +
  'getEntry: selectedEntry, getAttachments: () => attachments, getAudio: () => audio};', sandbox);
const api = sandbox.api;
const clone = value => JSON.parse(JSON.stringify(value));
const stringFile = (value, name = 'evidence.json') => {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  const bytes = new TextEncoder().encode(text);
  return {name, size: bytes.length, text: async () => text, arrayBuffer: async () => bytes.buffer};
};
const bytesFile = (bytes, name = 'stage.wav') => {
  const buffer = Uint8Array.from(bytes).buffer;
  return {name, size: buffer.byteLength, arrayBuffer: async () => buffer};
};
const wavFile = role => bytesFile(Buffer.from(fixtures.wav_hex[role], 'hex'));
const defer = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
const assertSamples = (actual, expected) => assert.deepEqual(Array.from(actual.samples), Array.from(expected.samples));
async function main() {
  assert.equal(audioCount, 0);
  for (const role of ['original', 'observed', 'filtered']) {
    assert.equal(ids[role + '-play'].disabled, true);
    assert.match(ids[role + '-info'].textContent, /Nothing generated/);
  }
  assert.match(ids.candidate.textContent, /No receiver/);
  const golden = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/clean_q3.json'), 'utf8'));
  api.inspectEvidence(golden.source);
  const goldenWav = fs.readFileSync(path.join(__dirname, 'fixtures/clean_q3.wav'));
  assertSamples(api.parseWAV(Uint8Array.from(goldenWav).buffer), api.parsePCM(golden.source.encoded_sound));

  // Real native demo, byte-exact WAV/JSON agreement, and unchanged signed samples.
  await api.loadEvidence(stringFile(fixtures.demo));
  assert.equal(ids['import-error'].textContent, '');
  assert.match(ids.candidate.textContent, /"imported_receiver_accepted": true/);
  assert.match(ids.rejected.textContent, /"imported_receiver_accepted": false/);
  assert.match(ids.verified.textContent, /"imported_verified_recovery": true/);
  assert.match(ids.verified.textContent, /unverified/);
  for (const role of ['original', 'observed', 'filtered']) {
    const pcm = api.stagePCM(role);
    assertSamples(api.parseWAV(await wavFile(role).arrayBuffer()), pcm);
    const hex = Buffer.alloc(2 * pcm.samples.length);
    pcm.samples.forEach((sample, n) => hex.writeInt16LE(sample, n * 2));
    assert.equal(hex.toString('hex'), fixtures.demo.roles[role].pcm_hex);
    assert.equal(ids[role + '-play'].disabled, false);
    assert.ok(ids[role + '-wave'].children[1].attrs.d.length > 0);
    await api.loadStage(role, wavFile(role));
    assert.match(ids.association.textContent, /exact sample match/);
  }
  assert.equal(audioCount, 0);

  // Source-only input never synthesizes observed/filtered data or decisions.
  await api.loadEvidence(stringFile(fixtures.source));
  assert.ok(api.stagePCM('original'));
  assert.equal(api.stagePCM('observed'), null);
  assert.equal(api.stagePCM('filtered'), null);
  assert.equal(Object.keys(api.getAttachments()).length, 0);
  assert.match(ids.verified.textContent, /No reference-gated/);

  // Coherent wrong-word acceptance remains visible before independent veto.
  await api.loadEvidence(stringFile(fixtures.trial));
  const immutableTrial = JSON.stringify(api.getEntry().artifact);
  assert.equal(ids['import-error'].textContent, '');
  assert.match(ids.candidate.textContent, /"imported_raw_receiver_accepted": true/);
  assert.match(ids.rejected.textContent, /"imported_raw_false_accept_frames": 1/);
  assert.match(ids.verified.textContent, /"imported_independently_verified_frames": 0/);
  assert.match(ids.verified.textContent, /reference_mismatch/);
  ids['method-select'].value = '5'; ids['method-select'].change();
  assert.match(ids.candidate.textContent, /no acoustic receiver/);
  assert.match(ids.candidate.textContent, /"imported_raw_receiver_accepted": null/);
  ids['method-select'].value = '6'; ids['method-select'].change();
  assert.match(ids.candidate.textContent, /Privileged clean-PCM/);
  await api.loadStage('observed', wavFile('filtered'));
  assert.match(ids.association.textContent, /no matching embedded PCM/);
  assert.equal(JSON.stringify(api.getEntry().artifact), immutableTrial, 'UI and stage replacements must not mutate imported receipts');

  await api.loadEvidence(stringFile(fixtures.battery));
  assert.equal(ids['import-error'].textContent, '');
  assert.equal(ids['trial-select'].children.length, 2);
  ids['trial-select'].value = '1'; ids['trial-select'].change();
  assert.match(ids['evidence-meta'].textContent, /comparative-trial/);
  assert.notEqual(api.stagePCM('original').samples.length, api.stagePCM('observed').samples.length);
  assert.match(ids.rejected.textContent, /sample_count_mismatch/);
  await api.loadEvidence(stringFile(fixtures.battery_source));
  assert.equal(ids['import-error'].textContent, '');
  assert.equal(api.stagePCM('observed'), null);

  // Standalone supplied native observations and gate receipts, no fabricated PCM.
  for (const value of [fixtures.demo.roles.candidate, fixtures.demo.roles.rejected,
    fixtures.demo.roles.verified, fixtures.demo.rejected_bridge, fixtures.demo.noise_observation, fixtures.demo.cancellation]) {
    await api.loadEvidence(stringFile(value));
    assert.equal(ids['import-error'].textContent, '', value.schema);
  }
  await api.loadEvidence(stringFile(fixtures.demo.roles.verified));
  assert.equal(api.stagePCM('filtered'), null);
  await api.loadStage('filtered', stringFile(fixtures.demo.roles.filtered));
  assert.match(ids.association.textContent, /no matching embedded PCM/);

  // Plain text only, including imported labels/reasons and hostile filenames.
  const hostile = clone(fixtures.demo);
  hostile.roles.rejected.reasons = ['<img src=x onerror=alert(1)>'];
  hostile.source.case_name = '<script>boom</script>';
  await api.loadEvidence(stringFile(hostile));
  assert.equal(ids['import-error'].textContent, '');
  assert.match(ids.rejected.textContent, /<img/);
  assert.equal(ids.rejected.children.length, 0);
  assert.equal(ids['trial-select'].children[0].children.length, 0);
  await api.loadStage('original', bytesFile(Buffer.from(fixtures.wav_hex.original, 'hex'), '<img>.wav'));
  assert.match(ids['original-info'].textContent, /<img>/);
  assert.equal(ids['original-info'].children.length, 0);

  // Strict malformed/unsupported/bounded imports clear old evidence and audio.
  const invalidPCM = [];
  for (const edit of [x => { x.sample_count++; }, x => { x.sample_rate = 44100; }, x => { x.pcm_hex += 'zz'; },
    x => { x.sample_count = true; }, x => { x.pcm_sha256 = 'bad'; }, x => { x.layout = 'stereo'; }]) {
    const value = clone(fixtures.demo.roles.original); edit(value); invalidPCM.push(value);
  }
  invalidPCM.forEach(value => assert.throws(() => api.parsePCM(value)));
  for (const value of [null, [], {}, true, '{', 'x'.repeat(8 * 1024 * 1024 + 1),
    '{"a":' + '['.repeat(40) + '0' + ']'.repeat(40) + '}', '{"a":1e400}',
    '{"__proto__":{"polluted":true}}', '{"a":[' + '0,'.repeat(128) + '0]}',
    {...fixtures.demo.roles.original}, {...fixtures.demo, roles: {}}, {...fixtures.demo.roles.candidate, accepted: 'true'}]) {
    await api.loadEvidence(stringFile(value));
    assert.notEqual(ids['import-error'].textContent, '');
    assert.equal(api.getEntry(), null);
  }
  const tooMany = clone(fixtures.battery); tooMany.trials = Array(39).fill(fixtures.trial);
  assert.throws(() => api.inspectEvidence(tooMany), /1–38/);
  assert.throws(() => api.parseJSON('"' + 'é'.repeat(600000) + '"', 1024 * 1024), /bounded/);
  let read = false;
  await api.loadEvidence({size: 8 * 1024 * 1024 + 1, text: async () => { read = true; return '{}'; }});
  assert.equal(read, false);
  await api.loadStage('original', {size: 1024 * 1024 + 1, name: 'big.wav', arrayBuffer: async () => { read = true; }});
  assert.equal(read, false);

  // WAV format/layout/chunk validation, including bounded non-audio metadata.
  const valid = Buffer.from(fixtures.wav_hex.original, 'hex');
  for (const change of [b => b.writeUInt16LE(2, 22), b => b.writeUInt16LE(32, 34), b => b.writeUInt16LE(3, 20),
    b => b.writeUInt32LE(44100, 24), b => b.writeUInt32LE(0xffffffff, 40), b => b.writeUInt32LE(1, 4),
    b => b.writeUInt32LE(1, 28), b => b.writeUInt16LE(4, 32), b => b.write('RIFF', 8)]) {
    const buffer = Buffer.from(valid); change(buffer);
    assert.throws(() => api.parseWAV(Uint8Array.from(buffer).buffer));
  }
  assert.throws(() => api.parseWAV(new ArrayBuffer(43)));
  assert.throws(() => api.parseWAV(new ArrayBuffer(44 + 2 * 73728 + 1)));
  const metadata = Buffer.concat([valid.subarray(0, 36), Buffer.from('JUNK'), Buffer.from([1, 0, 0, 0, 99, 0]), valid.subarray(36)]);
  metadata.writeUInt32LE(metadata.length - 8, 4);
  assertSamples(api.parseWAV(Uint8Array.from(metadata).buffer), api.parseWAV(Uint8Array.from(valid).buffer));
  const duplicate = Buffer.concat([valid, valid.subarray(36)]); duplicate.writeUInt32LE(duplicate.length - 8, 4);
  assert.throws(() => api.parseWAV(Uint8Array.from(duplicate).buffer), /duplicate WAV data/);

  // Pending older JSON reads cannot replace later evidence or undo reset.
  const old = defer();
  const pending = api.loadEvidence({size: 10, text: () => old.promise});
  await api.loadEvidence(stringFile(fixtures.source));
  old.resolve(JSON.stringify(fixtures.demo)); await pending;
  assert.equal(api.getEntry().artifact.schema, fixtures.source.schema);
  const cleared = defer(); const pendingReset = api.loadEvidence({size: 10, text: () => cleared.promise});
  ids.reset.click(); cleared.resolve(JSON.stringify(fixtures.demo)); await pendingReset;
  assert.equal(api.getEntry(), null);
  const lateError = defer(); const failure = api.loadEvidence({size: 10, text: () => lateError.promise});
  await api.loadEvidence(stringFile(fixtures.demo)); lateError.reject(new Error('obsolete')); await failure;
  assert.equal(ids['import-error'].textContent, '');

  // Per-stage newer imports, trial switches, evidence imports, and reset cancel old reads.
  const slow = defer();
  const staleStage = api.loadStage('observed', {size: valid.length, name: 'old.wav', arrayBuffer: () => slow.promise});
  await api.loadStage('observed', wavFile('observed'));
  slow.resolve(Uint8Array.from(valid).buffer); await staleStage;
  assertSamples(api.stagePCM('observed'), api.parsePCM(fixtures.demo.roles.observed));
  for (const action of [() => api.reset(), () => api.loadEvidence(stringFile(fixtures.source)),
    () => { ids['trial-select'].value = '1'; ids['trial-select'].change(); }]) {
    await api.loadEvidence(stringFile(fixtures.battery));
    const wait = defer();
    const stale = api.loadStage('original', {size: valid.length, name: 'old.wav', arrayBuffer: () => wait.promise});
    await action(); wait.resolve(Uint8Array.from(valid).buffer); await stale;
    assert.equal(Object.keys(api.getAttachments()).length, 0);
  }

  // No fallback generation: attach empty/silence PCM literally, and never play empty.
  const empty = clone(fixtures.demo.roles.original); empty.pcm_hex = ''; empty.sample_count = 0;
  await api.loadStage('original', stringFile(empty));
  assert.equal(api.stagePCM('original').samples.length, 0);
  assert.equal(ids['original-play'].disabled, true);
  await ids['original-play'].click(); assert.equal(audioCount, 0);

  // Real buffer samples and rate, fixed attenuation, user-only start and stop.
  const contexts = [];
  class FakeAudio {
    constructor() { this.state = 'running'; this.destination = {}; this.buffers = []; this.sources = []; this.gains = []; contexts.push(this); }
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createBuffer(channels, count, rate) {
      const data = new Float32Array(count); const b = {channels, rate, data, getChannelData: () => data}; this.buffers.push(b); return b;
    }
    createBufferSource() {
      const s = {connect() {}, start() { this.started = true; }, stop() { this.stopped = true; this.onended?.(); }};
      this.sources.push(s); return s;
    }
    createGain() { const gain = {gain: {value: 0}, connect() {}}; this.gains.push(gain); return gain; }
  }
  sandbox.window.AudioContext = FakeAudio;
  await api.loadEvidence(stringFile(fixtures.demo)); assert.equal(contexts.length, 0);
  await ids['observed-play'].click();
  const firstAudio = contexts.at(-1), expected = api.stagePCM('observed');
  assert.equal(firstAudio.buffers[0].rate, expected.sampleRate);
  assert.equal(firstAudio.buffers[0].channels, 1);
  assert.deepEqual(Array.from(firstAudio.buffers[0].data), Array.from(expected.samples, n => n / 32768));
  assert.equal(firstAudio.gains[0].gain.value, .25);
  assert.equal(firstAudio.sources[0].started, true);
  ids.stop.click(); assert.equal(firstAudio.state, 'closed'); assert.equal(api.getAudio(), null);
  assert.equal(ids.stop.disabled, true);
  await ids['filtered-play'].click();
  const secondAudio = contexts.at(-1); secondAudio.sources[0].onended();
  assert.equal(secondAudio.state, 'closed'); assert.match(ids['audio-status'].textContent, /finished/);
  assert.equal(ids['filtered-play'].disabled, false);

  // Late resume/rejection after stop/reset cannot start or stop a newer playback.
  for (const reject of [false, true]) {
    const wait = defer();
    sandbox.window.AudioContext = class extends FakeAudio { resume() { return wait.promise; } };
    const oldPlay = ids['observed-play'].click(), older = contexts.at(-1);
    ids.stop.click(); sandbox.window.AudioContext = FakeAudio;
    await ids['filtered-play'].click(); const newer = contexts.at(-1);
    if (reject) wait.reject(new Error('old resume failed')); else wait.resolve();
    await oldPlay;
    assert.equal(older.sources.length, 0); assert.equal(older.state, 'closed');
    assert.equal(newer.state, 'running'); assert.equal(api.getAudio().context, newer);
    ids.reset.click(); assert.equal(newer.state, 'closed');
    await api.loadEvidence(stringFile(fixtures.demo));
  }
  await ids['original-play'].click(); const closing = contexts.at(-1);
  windowEvents.pagehide(); assert.equal(closing.state, 'closed'); assert.equal(api.getEntry(), null);
  assert.equal(ids['original-play'].disabled, true);

  // No network, synthesized oscillator, unsafe DOM, storage or external assets.
  assert.equal(/\b(fetch|WebSocket|XMLHttpRequest|EventSource|createOscillator)\s*\(/.test(script), false);
  assert.equal(/innerHTML|outerHTML|insertAdjacentHTML|localStorage|sessionStorage/.test(script), false);
  assert.equal(/(?:Math\.random|Date\.now|new Date)\s*\(/.test(script), false);
  assert.match(html, /connect-src 'none'/);
  assert.equal(/(?:src|href)="https?:/.test(html), false);
  console.log('PASS: actual PCM/WAV, imported claim separation, bounded hostile inputs; stale loads, reset and click-only playback.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
