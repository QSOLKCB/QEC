// SPDX-License-Identifier: MPL-2.0
'use strict';
// Dependency-free behavioral coverage. This DOM/audio stub is not browser layout
// or acoustic QA; native receipt acceptance is never established by this test.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'docs/labs/digital/index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.options = this.children;
    this.value = '';
    this._text = '';
    this.attributes = {};
    this.listeners = {};
    this.disabled = false;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this.children.splice(0, this.children.length, ...children);
    this._text = '';
  }
  set textContent(value) { this._text = String(value); this.children.splice(0); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(key, listener) { this.listeners[key] = listener; }
  remove() {}
  click() { return this.listeners.click?.({target: this}); }
}

const elements = {};
for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) {
  assert.ok(!elements[id], 'Duplicate HTML ID: ' + id);
  elements[id] = new Element('div');
}
for (const [id, value] of Object.entries({
  payload: 'SAME INPUT. SAME OUTPUT.', encoding: 'text', 'slot-count': '4',
  'fault-frame': '3', fault: 'none',
})) elements[id].value = value;
let audioCreated = 0;
const context = vm.createContext({
  document: {
    getElementById: id => elements[id],
    createElement: tag => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    body: new Element('body'),
  },
  window: {
    addEventListener() {},
    AudioContext: function () { audioCreated++; throw new Error('Audio started before click'); },
  },
  TextEncoder, Blob, URL, setTimeout, console,
});
vm.runInContext(script + '\nglobalThis.api = {' +
  'simulate, serialize, readInput, execute, inspectNative, getRun: () => run, renderFrame};', context);
const {api} = context;
assert.equal(elements['result-pill'].textContent, 'DEMO ACCEPT');
assert.equal(audioCreated, 0);

let cases = 0;
const payloads = [
  '', '00', 'ff', '414141', '0051ff0051', '000102030405060708090a0b0c0d0e0f',
  Array.from({length: 64}, (_, i) => i.toString(16).padStart(2, '0')).join(''),
];
const faults = ['none', 'bit', 'loss', 'duplicate', 'reorder', 'slip', 'collision'];
for (const slotCount of [4, 8]) {
  for (let inputSlot = 0; inputSlot < slotCount; inputSlot++) {
    for (let outputSlot = 0; outputSlot < slotCount; outputSlot++) {
      for (const payload of payloads) {
        for (const target of [0, 63]) {
          for (const fault of faults) {
            const input = {
              payload_hex: payload, slot_count: slotCount, input_slot: inputSlot,
              output_slot: outputSlot, fault, requested_fault_frame: target,
            };
            const first = api.simulate(input);
            const replay = api.simulate(JSON.parse(JSON.stringify(input)));
            assert.equal(api.serialize(first.artifact), api.serialize(replay.artifact));
            assert.equal(first.accepted, fault === 'none', JSON.stringify(input));
            assert.equal(first.artifact.canonical_evidence, false);
            assert.equal(first.artifact.reassembled_payload_hex, fault === 'none' ? payload : null);
            assert.equal(first.count, Math.max(1, payload.length / 2));
            first.original.forEach(frame => assert.equal(frame.wire.length, 5 + 2 * slotCount));
            first.artifact.events.forEach((event, i) => assert.equal(event.logical_tick, i));
            assert.ok(first.display.length > 0);
            cases++;
          }
        }
      }
    }
  }
}

// Input rejection retains the prior run, and DATA 00 is distinct from IDLE.
const initial = api.serialize(api.getRun().artifact);
elements.payload.value = 'f';
elements.encoding.value = 'hex';
api.execute();
assert.match(elements['input-error'].textContent, /complete byte pairs/);
assert.equal(api.serialize(api.getRun().artifact), initial);
elements.payload.value = '00 ff 41';
api.execute();
assert.equal(api.getRun().artifact.reassembled_payload_hex, '00ff41');
assert.equal(api.getRun().display[0].decoded.slots[0].kind, 'DATA');
elements.payload.value = '';
api.execute();
assert.equal(api.getRun().accepted, true);
assert.equal(api.getRun().display[0].decoded.slots[0].kind, 'IDLE');
elements.payload.value = 'ff'.repeat(65);
api.execute();
assert.match(elements['input-error'].textContent, /64-byte/);
elements.payload.value = 'é'.repeat(33);
elements.encoding.value = 'text';
api.execute();
assert.match(elements['input-error'].textContent, /64-byte/);
elements.payload.value = 'é😀';
api.execute();
assert.equal(api.getRun().artifact.reassembled_payload_hex, 'c3a9f09f9880');

// Actual native golden fixture is recognized only for read-only inspection.
const native = fs.readFileSync(path.join(__dirname, 'fixtures/digital-receipt-v1.json'), 'utf8');
api.inspectNative(native);
assert.equal(elements['import-error'].textContent, '');
assert.match(elements['native-result'].textContent, /STRUCTURE RECOGNIZED/);
assert.match(elements['native-result'].textContent, /S1:IDLE/);
assert.match(elements['native-result'].textContent, /CONTENT UNVERIFIED/);
const rejected = JSON.parse(native);
rejected.frames = [];
rejected.sync_receipt = null;
rejected.interchange_receipt = null;
api.inspectNative(JSON.stringify(rejected));
assert.equal(elements['import-error'].textContent, '');
api.inspectNative(api.serialize(api.getRun().artifact));
assert.match(elements['import-error'].textContent, /Demo exports cannot/);
api.inspectNative('{');
assert.match(elements['import-error'].textContent, /Could not parse/);
api.inspectNative('x'.repeat(1048577));
assert.match(elements['import-error'].textContent, /1 MiB/);
for (const invalid of [null, [], {}, 'hi']) {
  api.inspectNative(JSON.stringify(invalid));
  assert.notEqual(elements['import-error'].textContent, '');
}
const oversized = JSON.parse(native);
oversized.input.frame_schema.slot_count = 17;
api.inspectNative(JSON.stringify(oversized));
assert.match(elements['import-error'].textContent, /Unsupported native shape/);
const adversarial = JSON.parse(native);
adversarial.reason = '<img src=x onerror=alert(1)>';
api.inspectNative(JSON.stringify(adversarial));
assert.match(elements['native-result'].textContent, /<img/);
assert.equal(elements['native-result'].children.some(el => el.tagName === 'img'), false);
elements['clear-import'].click();
assert.match(elements['native-result'].textContent, /No receipt loaded/);
assert.equal(audioCreated, 0);

// No external code, asset requests, random input, or wall-clock model inputs.
assert.equal(/\b(fetch|WebSocket|XMLHttpRequest|EventSource)\s*\(/.test(script), false);
assert.equal(/(?:Math\.random|Date\.now|new Date)\s*\(/.test(script), false);
assert.equal(/<script[^>]+src=|<link[^>]+(?:stylesheet|preload)/i.test(html), false);
assert.match(html, /connect-src 'none'/);
console.log(`PASS: ${cases} deterministic scenario pairs, input and native-import boundaries.`);

async function testInteractions() {
  const contexts = [];
  class FakeAudio {
    constructor() {
      this.state = 'running';
      this.currentTime = 5;
      this.oscillators = [];
      this.gains = [];
      this.destination = {};
      contexts.push(this);
    }
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createOscillator() {
      const oscillator = {
        frequency: {value: 0}, connect() {},
        start(at) { this.startAt = at; }, stop(at) { this.stopAt = at; },
      };
      this.oscillators.push(oscillator);
      return oscillator;
    }
    createGain() {
      const values = [];
      this.gains.push(values);
      return {
        gain: {
          setValueAtTime: (...args) => values.push(args),
          linearRampToValueAtTime: (...args) => values.push(args),
        },
        connect() {},
      };
    }
  }
  context.window.AudioContext = FakeAudio;
  await elements.sound.click();
  assert.equal(contexts.length, 1);
  assert.equal(contexts[0].oscillators.length, api.getRun().display.length);
  assert.equal(elements.stop.disabled, false);
  assert.ok(contexts[0].gains.flat().every(([value]) => value <= .035));
  elements.stop.click();
  assert.equal(contexts[0].state, 'closed');
  assert.equal(elements.sound.disabled, false);

  // A late rejection from stopped playback cannot stop a newer playback.
  let rejectFirst;
  context.window.AudioContext = class extends FakeAudio {
    resume() { return new Promise((_, reject) => { rejectFirst = reject; }); }
  };
  const first = elements.sound.click();
  elements.stop.click();
  context.window.AudioContext = FakeAudio;
  await elements.sound.click();
  const latest = contexts.at(-1);
  rejectFirst(new Error('closed'));
  await first;
  assert.equal(latest.state, 'running');
  elements.stop.click();

  // Pending file reads cannot replace a later Clear or pasted inspection.
  let resolveFile;
  const loading = elements['native-file'].listeners.change({
    target: {files: [{size: native.length, text: () => new Promise(resolve => { resolveFile = resolve; })}]},
  });
  elements['clear-import'].click();
  resolveFile(native);
  await loading;
  assert.match(elements['native-result'].textContent, /No receipt loaded/);
  let resolveAnother;
  const loadingAgain = elements['native-file'].listeners.change({
    target: {files: [{size: native.length, text: () => new Promise(resolve => { resolveAnother = resolve; })}]},
  });
  elements['native-json'].value = '{';
  elements.inspect.click();
  resolveAnother(native);
  await loadingAgain;
  assert.match(elements['import-error'].textContent, /Could not parse/);

  const beforeExport = api.serialize(api.getRun().artifact);
  elements.next.click();
  elements.export.click();
  assert.equal(api.serialize(api.getRun().artifact), beforeExport);
  console.log('PASS: click-only audio scheduling, stop/restart races, file races, and export isolation.');
}
testInteractions().catch(error => { console.error(error); process.exitCode = 1; });
