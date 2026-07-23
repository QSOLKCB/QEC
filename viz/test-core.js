"use strict";

const assert = require("node:assert/strict");
const Core = require("./qutrit-core.js");

assert.equal(Core.decoder(Core.CODES.five).size, 41);
assert.equal(Core.decoder(Core.CODES.shor).size, 61);
assert.equal(Core.decoder(Core.CODES.golay).size, 3609);

const goldens = {
  five: [2, 0, 2, 2],
  shor: [2, 0, 0, 0, 0, 0, 2, 0],
  golay: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
};
Object.values(Core.CODES).forEach((code) => {
  const goldenError = Core.fromTerms(code.n, [{ site: 0, x: 1, z: 2 }]);
  assert.deepEqual(Core.syndrome(code, goldenError), goldens[code.id]);
  for (let site = 0; site < code.n; site++) {
    for (let x = 0; x < 3; x++) {
      for (let z = 0; z < 3; z++) {
        if (!x && !z) continue;
        const error = Core.fromTerms(code.n, [{ site, x, z }]);
        assert.equal(Core.run(code, error, "none").success, true);
      }
    }
  }
});

const golay = Core.CODES.golay;
let weightTwo = 0;
for (let left = 0; left < golay.n; left++) {
  for (let right = left + 1; right < golay.n; right++) {
    for (let x1 = 0; x1 < 3; x1++) {
      for (let z1 = 0; z1 < 3; z1++) {
        if (!x1 && !z1) continue;
        for (let x2 = 0; x2 < 3; x2++) {
          for (let z2 = 0; z2 < 3; z2++) {
            if (!x2 && !z2) continue;
            const error = Core.fromTerms(golay.n, [
              { site: left, x: x1, z: z1 },
              { site: right, x: x2, z: z2 }
            ]);
            assert.equal(Core.run(golay, error, "none").success, true);
            assert.equal(Core.run(golay, error, "h2").observation.trusted, false);
            assert.equal(Core.run(golay, error, "h3").observation.trusted, false);
            weightTwo += 1;
          }
        }
      }
    }
  }
}
assert.equal(weightTwo, 3520);

const error = Core.fromTerms(golay.n, [{ site: 0, x: 1, z: 2 }]);
assert.equal(Core.run(golay, error, "h2").observation.trusted, false);
assert.equal(Core.run(golay, error, "h2").accepted, false);
assert.equal(Core.run(golay, error, "h3").observation.trusted, false);

const five = Core.CODES.five;
let exposedMiscorrection = false;
for (let left = 0; left < five.n && !exposedMiscorrection; left++) {
  for (let right = left + 1; right < five.n && !exposedMiscorrection; right++) {
    for (const [x1, z1] of [[1, 0], [0, 1], [1, 1]]) {
      for (const [x2, z2] of [[1, 0], [0, 1], [1, 1]]) {
        const candidate = Core.fromTerms(five.n, [
          { site: left, x: x1, z: z1 },
          { site: right, x: x2, z: z2 }
        ]);
        const result = Core.run(five, candidate, "none");
        if (result.accepted && !result.success) exposedMiscorrection = true;
      }
    }
  }
}
assert.equal(exposedMiscorrection, true);
console.log("qutrit viz core: exact checks passed");
