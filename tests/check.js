// Sanity checks for the regression math. Run with: node tests/check.js
'use strict';
require('../js/regression.js');
const { Reg } = globalThis;

let failures = 0;
function check(cond, msg) {
  if (!cond) {
    failures++;
    console.error('FAIL:', msg);
  }
}
const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

const rng = Reg.mulberry32(7);
const sine = Reg.presets.sine;
const pts = Reg.sample(sine.f, 30, sine.noise, rng);
const xs = pts.map((p) => p.x);
const ys = pts.map((p) => p.y);

// 1. Degree-1 OLS matches the textbook simple linear regression.
{
  const f = Reg.fit(xs, ys, { basis: 'poly', size: 1, penalty: 'none' });
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b) / n;
  const my = ys.reduce((a, b) => a + b) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  check(close(f.predict(0), my - slope * mx, 1e-6), `OLS intercept ${f.predict(0)}`);
  check(close(f.predict(1) - f.predict(0), slope, 1e-6), `OLS slope ${f.predict(1) - f.predict(0)}`);
}

// 2. n = degree + 1 points are interpolated exactly by OLS.
{
  const few = pts.slice(0, 6);
  const f = Reg.fit(few.map((p) => p.x), few.map((p) => p.y), { basis: 'poly', size: 5, penalty: 'none' });
  check(f.trainMSE < 1e-6, `interpolation train MSE ${f.trainMSE}`);
}

// 3. KKT conditions hold for ridge, lasso and elastic net on every basis.
for (const basis of Object.keys(Reg.bases)) {
  for (const penalty of ['ridge', 'lasso', 'elastic']) {
    for (const logLambda of [-3, -1.5, 0]) {
      const m = Reg.normalizeModel({ basis, size: Reg.bases[basis].initial, penalty, logLambda, alpha: 0.5 });
      const { lambda, alpha } = Reg.lambdaAlpha(m);
      const f = Reg.fit(xs, ys, m);
      const { G, c, p } = f.prep;
      const tag = `${basis}/${penalty}/λ=${lambda}`;
      for (let j = 0; j < p; j++) {
        let gw = 0;
        for (let k = 0; k < p; k++) gw += G[j][k] * f.w[k];
        // gradient of the smooth part
        const g = c[j] - gw - lambda * (1 - alpha) * f.w[j];
        const l1 = lambda * alpha;
        if (Math.abs(f.w[j]) > 1e-10) check(close(g, l1 * Math.sign(f.w[j]), 1e-4), `${tag}: KKT active j=${j} g=${g}`);
        else check(Math.abs(g) <= l1 + 1e-4, `${tag}: KKT inactive j=${j} g=${g}`);
      }
    }
  }
}

// 4. Larger lambda never increases the ridge coefficient norm; huge lambda gives the mean.
{
  let prev = Infinity;
  for (const logLambda of [-4, -2, 0, 2]) {
    const f = Reg.fit(xs, ys, { basis: 'poly', size: 8, penalty: 'ridge', logLambda });
    check(f.l2 <= prev + 1e-9, `ridge norm not monotone at ${logLambda}`);
    prev = f.l2;
  }
  const flat = Reg.fit(xs, ys, { basis: 'poly', size: 8, penalty: 'lasso', logLambda: 2 });
  const my = ys.reduce((a, b) => a + b) / ys.length;
  check(flat.nonzero === 0 && close(flat.predict(3), my, 1e-9), 'lasso at huge λ should be the mean');
}

// 5. Degenerate data does not blow up.
{
  for (const sub of [[], [{ x: 2, y: 3 }], [{ x: 2, y: 3 }, { x: 2, y: 5 }]]) {
    const f = Reg.fit(sub.map((p) => p.x), sub.map((p) => p.y), { basis: 'rbf', size: 10, penalty: 'lasso' });
    const v = f.predict(4);
    check(sub.length === 0 || isFinite(v), `degenerate n=${sub.length}: ${v}`);
  }
}

// 6. Path, CV and size curves are finite; CV picks a sensible degree for the sine.
{
  const t0 = Date.now();
  for (const penalty of ['ridge', 'lasso', 'elastic']) {
    const pth = Reg.path(xs, ys, { basis: 'poly', size: 12, penalty, alpha: 0.5 });
    check(pth.cv && pth.cv.mean.every(isFinite), `${penalty} CV curve not finite`);
    check(pth.best >= 0, `${penalty} no best λ`);
  }
  const tPath = Date.now() - t0;
  const t1 = Date.now();
  const sc = Reg.sizeCurve(xs, ys, { basis: 'poly', penalty: 'none' });
  const tSize = Date.now() - t1;
  check(sc.best >= 2 && sc.best <= 9, `sine best degree index ${sc.best}`);
  console.log(`timing: 3 paths ${tPath} ms, size curve ${tSize} ms`);
}

// 7. Oracle beats (or equals) any single setting on the test set.
{
  const r = Reg.mulberry32(99);
  const t = Reg.randomTarget(r);
  const train = Reg.sample(t.f, 20, 0.6, r);
  const test = Reg.sample(t.f, 200, 0.6, r);
  const t0 = Date.now();
  const o = Reg.oracle(train, test);
  console.log(`timing: oracle ${Date.now() - t0} ms (${t.name})`);
  const f = Reg.fit(train.map((p) => p.x), train.map((p) => p.y), { basis: 'poly', size: 3, penalty: 'none' });
  const e = Reg.mse(f, test.map((p) => p.x), test.map((p) => p.y));
  check(o.mse <= e + 1e-9, `oracle ${o.mse} worse than cubic ${e}`);
  check(o.mse > 0.6 ** 2 * 0.5, `oracle suspiciously low ${o.mse}`);
}

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('All checks passed');
