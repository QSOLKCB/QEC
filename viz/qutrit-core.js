// Exact browser-side GF(3) model for the qutrit QEC visual lab.
(function (global, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.QutritCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const mod3 = (value) => ((value % 3) + 3) % 3;
  const zero = (size) => Array(size).fill(0);
  const key = (values) => values.join("");
  const add = (left, right) => left.map((value, i) => mod3(value + right[i]));
  const inverse = (error) => ({
    x: error.x.map((value) => mod3(-value)),
    z: error.z.map((value) => mod3(-value))
  });
  const compose = (left, right) => ({
    x: add(left.x, right.x),
    z: add(left.z, right.z)
  });
  const weight = (error) => error.x.reduce(
    (sum, value, i) => sum + Number(value !== 0 || error.z[i] !== 0),
    0
  );

  function rank(rows) {
    const work = rows.map((row) => row.map(mod3));
    if (!work.length) return 0;
    let pivot = 0;
    for (let column = 0; column < work[0].length && pivot < work.length; column++) {
      const found = work.findIndex((row, index) => index >= pivot && row[column]);
      if (found < 0) continue;
      [work[pivot], work[found]] = [work[found], work[pivot]];
      const factor = work[pivot][column] === 1 ? 1 : 2;
      work[pivot] = work[pivot].map((value) => mod3(value * factor));
      work.forEach((row, index) => {
        if (index === pivot || !row[column]) return;
        const scale = row[column];
        work[index] = row.map((value, j) => mod3(value - scale * work[pivot][j]));
      });
      pivot += 1;
    }
    return pivot;
  }

  function symplectic(left, right) {
    const n = left.length / 2;
    let value = 0;
    for (let i = 0; i < n; i++) {
      value += left[i] * right[n + i] - left[n + i] * right[i];
    }
    return mod3(value);
  }

  function syndrome(code, error) {
    const vector = error.x.concat(error.z);
    return code.stabilizers.map((check) => symplectic(check, vector));
  }

  function isStabilizer(code, error) {
    const vector = error.x.concat(error.z);
    return rank(code.stabilizers.concat([vector])) === rank(code.stabilizers);
  }

  function fiveQutrit() {
    const stabilizers = [];
    for (let shift = 0; shift < 4; shift++) {
      const x = zero(5);
      const z = zero(5);
      [0, 3].forEach((site) => { x[(site + shift) % 5] = 1; });
      [1, 2].forEach((site) => { z[(site + shift) % 5] = 1; });
      stabilizers.push(x.concat(z));
    }
    return { id: "five", name: "Cyclic", n: 5, k: 1, d: 3, t: 1, stabilizers };
  }

  function shorQutrit() {
    const stabilizers = [];
    for (let block = 0; block < 3; block++) {
      [[0, 1], [1, 2]].forEach(([left, right]) => {
        const x = zero(9);
        const z = zero(9);
        z[3 * block + left] = 1;
        z[3 * block + right] = 2;
        stabilizers.push(x.concat(z));
      });
    }
    [[0, 1], [1, 2]].forEach(([left, right]) => {
      const x = zero(9);
      const z = zero(9);
      for (let offset = 0; offset < 3; offset++) {
        x[3 * left + offset] = 1;
        x[3 * right + offset] = 2;
      }
      stabilizers.push(x.concat(z));
    });
    return { id: "shor", name: "Shor", n: 9, k: 1, d: 3, t: 1, stabilizers };
  }

  function golayQutrit() {
    const checks = [
      [2, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0],
      [2, 2, 1, 2, 0, 1, 0, 1, 0, 0, 0],
      [2, 1, 2, 0, 2, 1, 0, 0, 1, 0, 0],
      [2, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0],
      [2, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1]
    ];
    const blank = zero(11);
    const stabilizers = checks.map((row) => row.concat(blank))
      .concat(checks.map((row) => blank.concat(row)));
    return { id: "golay", name: "Ternary Golay", n: 11, k: 1, d: 5, t: 2, stabilizers };
  }

  const CODES = Object.freeze({
    five: fiveQutrit(),
    shor: shorQutrit(),
    golay: golayQutrit()
  });
  const localPowers = [];
  for (let x = 0; x < 3; x++) {
    for (let z = 0; z < 3; z++) {
      if (x || z) localPowers.push([x, z]);
    }
  }

  function enumerate(code, maximumWeight) {
    const errors = [];
    for (let site = 0; site < code.n; site++) {
      localPowers.forEach(([xPower, zPower]) => {
        errors.push(fromTerms(code.n, [{ site, x: xPower, z: zPower }]));
      });
    }
    if (maximumWeight < 2) return errors;
    for (let left = 0; left < code.n; left++) {
      for (let right = left + 1; right < code.n; right++) {
        localPowers.forEach(([x1, z1]) => {
          localPowers.forEach(([x2, z2]) => {
            errors.push(fromTerms(code.n, [
              { site: left, x: x1, z: z1 },
              { site: right, x: x2, z: z2 }
            ]));
          });
        });
      }
    }
    return errors;
  }

  function fromTerms(n, terms) {
    const error = { x: zero(n), z: zero(n) };
    terms.forEach((term) => {
      error.x[term.site] = mod3(error.x[term.site] + Number(term.x));
      error.z[term.site] = mod3(error.z[term.site] + Number(term.z));
    });
    return error;
  }

  const decoders = {};
  function decoder(code) {
    if (decoders[code.id]) return decoders[code.id];
    const identity = { x: zero(code.n), z: zero(code.n) };
    const table = new Map([[key(syndrome(code, identity)), identity]]);
    enumerate(code, code.t).forEach((candidate) => {
      const syndromeKey = key(syndrome(code, candidate));
      const previous = table.get(syndromeKey);
      if (!previous) {
        table.set(syndromeKey, candidate);
      } else if (!isStabilizer(code, compose(candidate, inverse(previous)))) {
        throw new Error("Exact correction condition failed for " + code.name);
      }
    });
    decoders[code.id] = table;
    return table;
  }

  function phasor(symbol, harmonic) {
    const angle = 2 * Math.PI * mod3(symbol * harmonic) / 3;
    return [Math.cos(angle), Math.sin(angle)];
  }

  function observe(values, fault) {
    const h1 = values.map((value) => phasor(value, 1));
    const h2 = values.map((value) => phasor(value, 2));
    const h3 = values.map((value) => phasor(value, 3));
    if (fault === "h2" && values.length) h2[0] = phasor(mod3(values[0] + 1), 2);
    if (fault === "h3" && values.length) h3[0] = [0, 0];
    const nearest = (samples, order) => samples.map((sample) => {
      const scores = [0, 1, 2].map((symbol) => {
        const ideal = phasor(symbol, order);
        return (sample[0] - ideal[0]) ** 2 + (sample[1] - ideal[1]) ** 2;
      });
      return scores.indexOf(Math.min(...scores));
    });
    const first = nearest(h1, 1);
    const second = nearest(h2, 2);
    const agreement = first.every((value, i) => value === second[i]);
    const darkInvariant = h3.every(([real, imag]) =>
      Math.abs(real - 1) < 1e-12 && Math.abs(imag) < 1e-12
    );
    return { syndrome: first, h1, h2, h3, agreement, darkInvariant,
      trusted: agreement && darkInvariant };
  }

  function collectivePower(values) {
    const field = values.map((value) => phasor(value, 1));
    return values.map((_, mode) => {
      let real = 0;
      let imag = 0;
      field.forEach(([a, b], site) => {
        const angle = -2 * Math.PI * mode * site / values.length;
        real += a * Math.cos(angle) - b * Math.sin(angle);
        imag += a * Math.sin(angle) + b * Math.cos(angle);
      });
      return (real * real + imag * imag) / values.length;
    });
  }

  function run(code, error, fault) {
    const exactSyndrome = syndrome(code, error);
    const observation = observe(exactSyndrome, fault);
    const representative = decoder(code).get(key(observation.syndrome));
    const accepted = Boolean(observation.trusted && representative);
    const correction = accepted
      ? inverse(representative) : null;
    const residual = correction ? compose(error, correction) : error;
    const success = Boolean(correction && isStabilizer(code, residual));
    return { code, error, exactSyndrome, observation, accepted, correction, residual,
      success, errorWeight: weight(error), withinRadius: weight(error) <= code.t,
      modePower: collectivePower(exactSyndrome) };
  }

  return { CODES, collectivePower, decoder, fromTerms, phasor, run, syndrome, weight };
});
