// Regression math: basis expansions, ridge / lasso / elastic-net solvers,
// cross-validation, regularization paths and data generators.
// Works in the browser (window.Reg) and in Node (globalThis.Reg).
(function (root) {
  'use strict';

  const DOMAIN = { x: [0, 10], y: [0, 10] };

  // ---------------------------------------------------------------------------
  // Random numbers
  // ---------------------------------------------------------------------------

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rng) {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  // ---------------------------------------------------------------------------
  // Basis expansions. Each maps x in [0, 10] to a feature vector (no intercept;
  // the intercept is always fitted and never penalized).
  // ---------------------------------------------------------------------------

  const SUP = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
  const sup = (k) => String(k).split('').map((c) => SUP[+c]).join('');
  const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
  const sub = (k) => String(k).split('').map((c) => SUB[+c]).join('');

  const bases = {
    poly: {
      id: 'poly',
      name: 'Polynomial',
      sizeLabel: 'Degree',
      min: 1,
      max: 15,
      initial: 3,
      count: (k) => k,
      // Powers of u = (x - 5) / 5 in [-1, 1] keep the columns well scaled.
      fill(x, k, width, out) {
        const u = (x - 5) / 5;
        let v = 1;
        for (let j = 0; j < k; j++) {
          v *= u;
          out[j] = v;
        }
      },
      labels: (k) => Array.from({ length: k }, (_, j) => (j === 0 ? 'x' : `x${sup(j + 1)}`)),
      tex: (k) => `\\phi_j(x) = u^j,\\; u = \\tfrac{x-5}{5},\\; j = 1\\ldots ${k}`,
    },
    rbf: {
      id: 'rbf',
      name: 'Gaussian RBF',
      sizeLabel: 'Centers',
      min: 1,
      max: 20,
      initial: 8,
      count: (k) => k,
      fill(x, k, width, out) {
        const spacing = 10 / k;
        const s = spacing * width;
        for (let j = 0; j < k; j++) {
          const c = (j + 0.5) * spacing;
          const d = (x - c) / s;
          out[j] = Math.exp(-0.5 * d * d);
        }
      },
      labels: (k) => Array.from({ length: k }, (_, j) => `φ${sub(j + 1)}`),
      tex: (k) => `\\phi_j(x) = e^{-(x-c_j)^2 / 2s^2},\\; j = 1\\ldots ${k}`,
    },
    fourier: {
      id: 'fourier',
      name: 'Fourier',
      sizeLabel: 'Frequencies',
      min: 1,
      max: 10,
      initial: 3,
      count: (k) => 2 * k,
      // Half-wave base frequency so the fit is not forced to be periodic on [0, 10].
      fill(x, k, width, out) {
        const t = (Math.PI * x) / 10;
        for (let j = 0; j < k; j++) {
          out[2 * j] = Math.sin((j + 1) * t);
          out[2 * j + 1] = Math.cos((j + 1) * t);
        }
      },
      labels: (k) => {
        const out = [];
        for (let j = 1; j <= k; j++) out.push(`sin${sub(j)}`, `cos${sub(j)}`);
        return out;
      },
      tex: (k) => `\\phi(x) = \\{\\sin\\tfrac{j\\pi x}{10}, \\cos\\tfrac{j\\pi x}{10}\\}_{j=1}^{${k}}`,
    },
  };

  const penalties = {
    none: { id: 'none', name: 'None', alpha: 0, desc: 'Ordinary least squares. Nothing stops the weights from growing to chase the noise.' },
    ridge: { id: 'ridge', name: 'Ridge', alpha: 0, desc: 'L2 penalty. Shrinks all weights smoothly towards zero but rarely makes any exactly zero.' },
    lasso: { id: 'lasso', name: 'Lasso', alpha: 1, desc: 'L1 penalty. Pushes weights all the way to zero, so it also selects features.' },
    elastic: { id: 'elastic', name: 'Elastic net', alpha: null, desc: 'A mix of L1 and L2: sparse like the lasso, but more stable with correlated features.' },
  };

  const penaltyTex = {
    none: '\\min_{b,\\,w}\\; \\tfrac{1}{2n}\\lVert y - b - \\Phi w \\rVert_2^2',
    ridge: '\\min_{b,\\,w}\\; \\tfrac{1}{2n}\\lVert y - b - \\Phi w \\rVert_2^2 + \\tfrac{\\lambda}{2}\\lVert w \\rVert_2^2',
    lasso: '\\min_{b,\\,w}\\; \\tfrac{1}{2n}\\lVert y - b - \\Phi w \\rVert_2^2 + \\lambda \\lVert w \\rVert_1',
    elastic:
      '\\min_{b,\\,w}\\; \\tfrac{1}{2n}\\lVert y - b - \\Phi w \\rVert_2^2 + \\lambda\\big(\\alpha\\lVert w \\rVert_1 + \\tfrac{1-\\alpha}{2}\\lVert w \\rVert_2^2\\big)',
  };

  const LOG_LAMBDA = { min: -5, max: 2 };

  function defaultModel() {
    return { basis: 'poly', size: 3, width: 1, penalty: 'ridge', logLambda: -2, alpha: 0.5 };
  }

  function normalizeModel(m) {
    const d = defaultModel();
    const out = Object.assign({}, d, m || {});
    if (!bases[out.basis]) out.basis = d.basis;
    if (!penalties[out.penalty]) out.penalty = d.penalty;
    const b = bases[out.basis];
    out.size = Math.round(clamp(Number(out.size) || b.initial, b.min, b.max));
    out.width = clamp(Number(out.width) || 1, 0.25, 3);
    out.logLambda = clamp(isFinite(out.logLambda) ? Number(out.logLambda) : d.logLambda, LOG_LAMBDA.min, LOG_LAMBDA.max);
    out.alpha = clamp(isFinite(out.alpha) ? Number(out.alpha) : d.alpha, 0.05, 0.95);
    return out;
  }

  // Effective (lambda, alpha) for a model; 'none' is lambda = 0.
  function lambdaAlpha(m) {
    const pen = penalties[m.penalty];
    const lambda = m.penalty === 'none' ? 0 : Math.pow(10, m.logLambda);
    const alpha = pen.alpha == null ? m.alpha : pen.alpha;
    return { lambda, alpha };
  }

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  function featureCount(m) {
    return bases[m.basis].count(m.size);
  }

  // Column-major design matrix: cols[j][i] = phi_j(x_i).
  function design(xs, m) {
    const b = bases[m.basis];
    const p = b.count(m.size);
    const n = xs.length;
    const cols = Array.from({ length: p }, () => new Float64Array(n));
    const row = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      b.fill(xs[i], m.size, m.width, row);
      for (let j = 0; j < p; j++) cols[j][i] = row[j];
    }
    return cols;
  }

  // ---------------------------------------------------------------------------
  // Fitting. Features and y are standardized; the objective is
  //   (1/2n) ||yc - Z w||^2 + lambda * (alpha ||w||_1 + (1 - alpha)/2 ||w||^2)
  // which is the glmnet convention. alpha = 0 is ridge and has a closed form.
  // ---------------------------------------------------------------------------

  // Standardize and precompute the Gram matrix G = Z'Z/n and c = Z'yc/n.
  function prepare(xs, ys, m, cols) {
    const n = xs.length;
    cols = cols || design(xs, m);
    const p = cols.length;
    const mean = new Float64Array(p);
    const sd = new Float64Array(p);
    let ym = 0;
    for (let i = 0; i < n; i++) ym += ys[i];
    ym = n ? ym / n : 0;
    let sy = 0;
    for (let i = 0; i < n; i++) sy += (ys[i] - ym) ** 2;
    sy = n ? Math.sqrt(sy / n) : 0;
    if (!(sy > 1e-12)) sy = 1;

    const Z = [];
    for (let j = 0; j < p; j++) {
      const c = cols[j];
      let mu = 0;
      for (let i = 0; i < n; i++) mu += c[i];
      mu = n ? mu / n : 0;
      let v = 0;
      for (let i = 0; i < n; i++) v += (c[i] - mu) ** 2;
      let s = n ? Math.sqrt(v / n) : 0;
      const z = new Float64Array(n);
      if (s > 1e-10) {
        for (let i = 0; i < n; i++) z[i] = (c[i] - mu) / s;
      } else {
        s = 1; // constant column: carries no information beyond the intercept
      }
      mean[j] = mu;
      sd[j] = s;
      Z.push(z);
    }
    const yc = new Float64Array(n);
    for (let i = 0; i < n; i++) yc[i] = (ys[i] - ym) / sy;

    const G = Array.from({ length: p }, () => new Float64Array(p));
    const c = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const zj = Z[j];
      let s = 0;
      for (let i = 0; i < n; i++) s += zj[i] * yc[i];
      c[j] = n ? s / n : 0;
      for (let k = j; k < p; k++) {
        const zk = Z[k];
        let t = 0;
        for (let i = 0; i < n; i++) t += zj[i] * zk[i];
        G[j][k] = G[k][j] = n ? t / n : 0;
      }
    }
    return { n, p, mean, sd, ym, sy, G, c };
  }

  // Cholesky solve of (A + jitter I) x = b, increasing jitter if needed.
  function cholSolve(A, b, jitter) {
    const p = b.length;
    let eps = jitter || 0;
    for (let attempt = 0; attempt < 12; attempt++) {
      const L = Array.from({ length: p }, () => new Float64Array(p));
      let ok = true;
      for (let i = 0; i < p && ok; i++) {
        for (let j = 0; j <= i; j++) {
          let s = A[i][j] + (i === j ? eps : 0);
          for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
          if (i === j) {
            if (!(s > 0)) {
              ok = false;
              break;
            }
            L[i][i] = Math.sqrt(s);
          } else L[i][j] = s / L[j][j];
        }
      }
      if (ok) {
        const z = new Float64Array(p);
        for (let i = 0; i < p; i++) {
          let s = b[i];
          for (let k = 0; k < i; k++) s -= L[i][k] * z[k];
          z[i] = s / L[i][i];
        }
        const x = new Float64Array(p);
        for (let i = p - 1; i >= 0; i--) {
          let s = z[i];
          for (let k = i + 1; k < p; k++) s -= L[k][i] * x[k];
          x[i] = s / L[i][i];
        }
        return x;
      }
      eps = eps ? eps * 10 : 1e-12;
    }
    return new Float64Array(p);
  }

  const OLS_JITTER = 1e-12;

  function ridgeSolve(prep, lambda) {
    return cholSolve(prep.G, prep.c, lambda + OLS_JITTER);
  }

  function softThreshold(v, t) {
    return v > t ? v - t : v < -t ? v + t : 0;
  }

  // Coordinate descent on the Gram matrix (cheap: p is small, n can be larger).
  function cdSolve(prep, lambda, alpha, warm, opts = {}) {
    const { p, G, c } = prep;
    const tol = opts.tol || 1e-7;
    const maxSweeps = opts.maxSweeps || 20000;
    const w = warm ? Float64Array.from(warm) : new Float64Array(p);
    const q = new Float64Array(p); // q = G w
    for (let j = 0; j < p; j++) {
      if (w[j] === 0) continue;
      for (let k = 0; k < p; k++) q[k] += G[k][j] * w[j];
    }
    const l1 = lambda * alpha;
    const l2 = lambda * (1 - alpha);
    let sweeps = 0;
    for (; sweeps < maxSweeps; sweeps++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const gjj = G[j][j];
        const denom = gjj + l2;
        const rho = c[j] - q[j] + gjj * w[j];
        const wn = denom > 1e-12 ? softThreshold(rho, l1) / denom : 0;
        const d = wn - w[j];
        if (d !== 0) {
          const Gj = G[j];
          for (let k = 0; k < p; k++) q[k] += Gj[k] * d;
          w[j] = wn;
          const ad = Math.abs(d);
          if (ad > maxDelta) maxDelta = ad;
        }
      }
      if (maxDelta < tol) break;
    }
    w.sweeps = sweeps;
    return w;
  }

  // Coordinate descent is slow to converge with strongly correlated columns
  // (high-degree polynomials), but finds the right support quickly. Given a
  // support, the solution is a linear solve; a small active-set loop then
  // fixes up the support until the KKT conditions hold exactly.
  function activeSetPolish(prep, w0, l1, l2) {
    const { p, G, c } = prep;
    const sign = new Float64Array(p);
    for (let j = 0; j < p; j++) sign[j] = Math.sign(w0[j]);
    const w = new Float64Array(p);
    for (let iter = 0; iter < 4 * p + 10; iter++) {
      const A = [];
      for (let j = 0; j < p; j++) if (sign[j] !== 0) A.push(j);
      w.fill(0);
      if (A.length) {
        const GA = A.map((i) => Float64Array.from(A, (j) => G[i][j]));
        const rhs = Float64Array.from(A, (j) => c[j] - l1 * sign[j]);
        const x = cholSolve(GA, rhs, l2 + OLS_JITTER);
        let flipped = false;
        A.forEach((j, i) => {
          if (Math.sign(x[i]) !== sign[j]) {
            sign[j] = 0;
            flipped = true;
          } else w[j] = x[i];
        });
        if (flipped) continue;
      }
      let worst = -1;
      let worstG = l1 * (1 + 1e-9) + 1e-12;
      for (let j = 0; j < p; j++) {
        if (sign[j] !== 0) continue;
        let g = c[j];
        for (let k = 0; k < p; k++) g -= G[j][k] * w[k];
        if (Math.abs(g) > worstG) {
          worstG = Math.abs(g);
          worst = j;
        }
      }
      if (worst < 0) return w;
      let g = c[worst];
      for (let k = 0; k < p; k++) g -= G[worst][k] * w[k];
      sign[worst] = Math.sign(g);
    }
    return null;
  }

  function solve(prep, lambda, alpha, warm, opts = {}) {
    if (prep.p === 0) return new Float64Array(0);
    if (alpha === 0 || lambda === 0) return ridgeSolve(prep, lambda);
    const cdOpts = { tol: opts.tol || 1e-6, maxSweeps: opts.maxSweeps || 300 };
    const w = cdSolve(prep, lambda, alpha, warm, cdOpts);
    const exact = activeSetPolish(prep, w, lambda * alpha, lambda * (1 - alpha));
    return exact || cdSolve(prep, lambda, alpha, w, { tol: 1e-8, maxSweeps: 20000 });
  }

  // Raw-scale coefficients so predict(x) = b0 + sum beta_j phi_j(x).
  function toModel(prep, w, m) {
    const p = prep.p;
    const beta = new Float64Array(p);
    let b0 = prep.ym;
    for (let j = 0; j < p; j++) {
      beta[j] = (prep.sy * w[j]) / prep.sd[j];
      b0 -= beta[j] * prep.mean[j];
    }
    const b = bases[m.basis];
    const row = new Float64Array(p);
    function predict(x) {
      b.fill(x, m.size, m.width, row);
      let s = b0;
      for (let j = 0; j < p; j++) s += beta[j] * row[j];
      return s;
    }
    return { w, beta, b0, predict };
  }

  // Effective degrees of freedom tr(H): on the active set A,
  // tr(G_AA (G_AA + lambda (1 - alpha) I)^-1), plus one for the intercept.
  function effectiveDf(prep, w, lambda, alpha) {
    const A = [];
    for (let j = 0; j < prep.p; j++) if (alpha === 0 || lambda === 0 || w[j] !== 0) A.push(j);
    if (!A.length) return 1;
    const GA = A.map((i) => Float64Array.from(A, (j) => prep.G[i][j]));
    const shift = lambda * (1 - alpha) + OLS_JITTER;
    let tr = 0;
    for (let k = 0; k < A.length; k++) {
      // column k of (G_AA + shift I)^-1 G_AA
      const col = Float64Array.from(A, (_, i) => GA[i][k]);
      const x = cholSolve(GA, col, shift);
      tr += x[k];
    }
    return 1 + Math.min(tr, prep.n - 1);
  }

  function mse(model, xs, ys) {
    if (!xs.length) return NaN;
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += (ys[i] - model.predict(xs[i])) ** 2;
    return s / xs.length;
  }

  function variance(ys) {
    if (!ys.length) return NaN;
    let m = 0;
    for (const y of ys) m += y;
    m /= ys.length;
    let s = 0;
    for (const y of ys) s += (y - m) ** 2;
    return s / ys.length;
  }

  // Fit one model to (xs, ys).
  function fit(xs, ys, m, opts) {
    m = normalizeModel(m);
    const { lambda, alpha } = lambdaAlpha(m);
    const prep = prepare(xs, ys, m);
    const w = solve(prep, lambda, alpha, null, opts);
    const model = toModel(prep, w, m);
    model.prep = prep;
    model.lambda = lambda;
    model.alpha = alpha;
    model.trainMSE = mse(model, xs, ys);
    const v = variance(ys);
    model.r2 = v > 0 ? 1 - model.trainMSE / v : NaN;
    model.df = xs.length ? effectiveDf(prep, w, lambda, alpha) : 0;
    let nz = 0;
    let l1 = 0;
    let l2 = 0;
    for (const v of w) {
      if (Math.abs(v) > 1e-10) nz++;
      l1 += Math.abs(v);
      l2 += v * v;
    }
    model.nonzero = nz;
    model.l1 = l1;
    model.l2 = Math.sqrt(l2);
    return model;
  }

  // ---------------------------------------------------------------------------
  // Cross-validation
  // ---------------------------------------------------------------------------

  // K-fold assignment (leave-one-out for small n). Deterministic in the index so
  // folds don't jump around while you drag points.
  function folds(n, kMax = 5) {
    if (n < 3) return null;
    const k = n < 2 * kMax ? n : kMax;
    const out = [];
    for (let f = 0; f < k; f++) {
      const test = [];
      const train = [];
      for (let i = 0; i < n; i++) (i % k === f ? test : train).push(i);
      out.push({ train, test });
    }
    return out;
  }

  function pick(arr, idx) {
    return idx.map((i) => arr[i]);
  }

  // Mean and standard error of the fold MSEs at one or more lambdas.
  // Returns { mean: Float64Array(L), se: Float64Array(L) } or null.
  function cvCurve(xs, ys, m, lambdas, alpha, opts = {}) {
    const fs = folds(xs.length, opts.k);
    if (!fs) return null;
    const L = lambdas.length;
    const foldErr = fs.map(() => new Float64Array(L));
    const allCols = design(xs, m);
    fs.forEach((f, fi) => {
      const tx = pick(xs, f.train);
      const ty = pick(ys, f.train);
      const cols = allCols.map((c) => Float64Array.from(f.train, (i) => c[i]));
      const prep = prepare(tx, ty, m, cols);
      const vx = pick(xs, f.test);
      const vy = pick(ys, f.test);
      let warm = null;
      for (const li of order(lambdas)) {
        const w = solve(prep, lambdas[li], alpha, warm, opts);
        warm = w;
        foldErr[fi][li] = mse(toModel(prep, w, m), vx, vy);
      }
    });
    const mean = new Float64Array(L);
    const se = new Float64Array(L);
    // Weight folds by size (LOO and uneven folds).
    const sizes = fs.map((f) => f.test.length);
    const total = sizes.reduce((a, b) => a + b, 0);
    for (let l = 0; l < L; l++) {
      let s = 0;
      for (let f = 0; f < fs.length; f++) s += foldErr[f][l] * sizes[f];
      const mu = s / total;
      let v = 0;
      for (let f = 0; f < fs.length; f++) v += (foldErr[f][l] - mu) ** 2;
      mean[l] = mu;
      se[l] = Math.sqrt(v / Math.max(1, fs.length - 1) / fs.length);
    }
    return { mean, se, k: fs.length };
  }

  // Indices sorted by decreasing lambda (warm starts work best from the sparse end).
  function order(lambdas) {
    return lambdas.map((_, i) => i).sort((a, b) => lambdas[b] - lambdas[a]);
  }

  function lambdaGrid(count = 36) {
    const out = [];
    for (let i = 0; i < count; i++) {
      out.push(Math.pow(10, LOG_LAMBDA.min + (i / (count - 1)) * (LOG_LAMBDA.max - LOG_LAMBDA.min)));
    }
    return out;
  }

  // Full regularization path: standardized coefficients, train and CV error per lambda.
  function path(xs, ys, m, opts = {}) {
    m = normalizeModel(m);
    const { alpha } = lambdaAlpha(m);
    const lambdas = opts.lambdas || lambdaGrid(opts.count);
    const prep = prepare(xs, ys, m);
    const coefs = new Array(lambdas.length);
    const train = new Float64Array(lambdas.length);
    let warm = null;
    for (const li of order(lambdas)) {
      const w = solve(prep, lambdas[li], alpha, warm, opts);
      warm = w;
      coefs[li] = w;
      train[li] = mse(toModel(prep, w, m), xs, ys);
    }
    const cv = cvCurve(xs, ys, m, lambdas, alpha, opts);
    let best = -1;
    if (cv) {
      for (let i = 0; i < lambdas.length; i++) if (best < 0 || cv.mean[i] < cv.mean[best]) best = i;
    }
    return { lambdas, logLambdas: lambdas.map(Math.log10), coefs, train, cv, best };
  }

  // Train and CV error as a function of the basis size, at the current penalty.
  function sizeCurve(xs, ys, m, opts = {}) {
    m = normalizeModel(m);
    const b = bases[m.basis];
    const { lambda, alpha } = lambdaAlpha(m);
    const sizes = [];
    const train = [];
    const cv = [];
    const se = [];
    for (let k = b.min; k <= b.max; k++) {
      const mk = Object.assign({}, m, { size: k });
      const f = fit(xs, ys, mk, opts);
      const c = cvCurve(xs, ys, mk, [lambda], alpha, opts);
      sizes.push(k);
      train.push(f.trainMSE);
      cv.push(c ? c.mean[0] : NaN);
      se.push(c ? c.se[0] : NaN);
    }
    let best = -1;
    for (let i = 0; i < sizes.length; i++) if (isFinite(cv[i]) && (best < 0 || cv[i] < cv[best])) best = i;
    return { sizes, train, cv, se, best };
  }

  function cvAt(xs, ys, m, opts) {
    m = normalizeModel(m);
    const { lambda, alpha } = lambdaAlpha(m);
    const c = cvCurve(xs, ys, m, [lambda], alpha, opts);
    return c ? { mean: c.mean[0], se: c.se[0], k: c.k } : null;
  }

  // ---------------------------------------------------------------------------
  // Data
  // ---------------------------------------------------------------------------

  const presets = {
    sine: { name: 'Sine wave', f: (x) => 5 + 3 * Math.sin(0.9 * x - 1), n: 25, noise: 0.8 },
    linear: { name: 'Straight line', f: (x) => 2 + 0.6 * x, n: 20, noise: 0.8 },
    cubic: { name: 'Cubic', f: (x) => 5 + 0.04 * (x - 5) ** 3 - 0.35 * (x - 5), n: 25, noise: 0.7 },
    step: { name: 'Step', f: (x) => (x < 5 ? 3 : 7), n: 30, noise: 0.5 },
    bump: { name: 'Narrow bump', f: (x) => 3 + 4.5 * Math.exp(-((x - 6) ** 2) / 0.8), n: 30, noise: 0.4 },
    outliers: { name: 'Line + outliers', f: (x) => 2.5 + 0.5 * x, n: 20, noise: 0.4, outliers: 3 },
  };

  function sample(f, n, noise, rng, outliers = 0) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = 0.3 + rng() * 9.4;
      pts.push({ x, y: clamp(f(x) + noise * gauss(rng), 0.1, 9.9) });
    }
    for (let i = 0; i < outliers && i < n; i++) {
      const p = pts[Math.floor(rng() * n)];
      p.y = clamp(p.y + (rng() < 0.5 ? -1 : 1) * (3 + 2 * rng()), 0.2, 9.8);
    }
    return pts.sort((a, b) => a.x - b.x);
  }

  // Random hidden function for the challenge, roughly within [1.5, 8.5].
  function randomTarget(rng) {
    const r = (lo, hi) => lo + rng() * (hi - lo);
    const kinds = ['wave', 'poly', 'step', 'bumps', 'kink', 'logistic', 'chirp'];
    const kind = kinds[Math.floor(rng() * kinds.length)];
    let f;
    let name;
    if (kind === 'wave') {
      const a = r(1.5, 3);
      const w = r(0.5, 1.4);
      const ph = r(0, 2 * Math.PI);
      const c = r(4, 6);
      f = (x) => c + a * Math.sin(w * x + ph);
      name = 'A sine wave';
    } else if (kind === 'poly') {
      // Lagrange interpolation through random knots
      const kx = [0.5, 3.5, 6.5, 9.5];
      const ky = kx.map(() => r(2, 8));
      f = (x) => {
        let s = 0;
        for (let i = 0; i < 4; i++) {
          let t = ky[i];
          for (let j = 0; j < 4; j++) if (j !== i) t *= (x - kx[j]) / (kx[i] - kx[j]);
          s += t;
        }
        return s;
      };
      name = 'A cubic polynomial';
    } else if (kind === 'step') {
      const at = r(3, 7);
      const lo = r(2, 4);
      const hi = r(6, 8);
      const up = rng() < 0.5;
      f = (x) => ((x < at) === up ? lo : hi);
      name = 'A step';
    } else if (kind === 'bumps') {
      const b = [r(1.5, 4.5), r(5.5, 8.5)];
      const h = [r(2, 4), r(-3, 3)];
      const s = [r(0.6, 1.5), r(0.6, 1.5)];
      f = (x) => 4 + h[0] * Math.exp(-((x - b[0]) ** 2) / (2 * s[0] ** 2)) + h[1] * Math.exp(-((x - b[1]) ** 2) / (2 * s[1] ** 2));
      name = 'Two bumps';
    } else if (kind === 'kink') {
      const at = r(3, 7);
      const s1 = r(-0.8, 0.8);
      const s2 = r(-0.8, 0.8);
      const c = r(4, 6);
      f = (x) => clamp(c + (x < at ? s1 * (x - at) : s2 * (x - at)), 1.5, 8.5);
      name = 'A kinked line';
    } else if (kind === 'logistic') {
      const at = r(3, 7);
      const k = r(0.8, 3);
      const lo = r(2, 3.5);
      const hi = r(6.5, 8);
      f = (x) => lo + (hi - lo) / (1 + Math.exp(-k * (x - at)));
      name = 'An S-curve';
    } else {
      const a = r(1.5, 2.5);
      const w = r(0.1, 0.2);
      f = (x) => 5 + a * Math.sin(w * x * x + 0.5);
      name = 'A chirp';
    }
    return { kind, name, f };
  }

  // Best test MSE over a grid of settings (the "oracle" reference for the challenge).
  function oracle(train, test, opts = {}) {
    const xs = train.map((p) => p.x);
    const ys = train.map((p) => p.y);
    const tx = test.map((p) => p.x);
    const ty = test.map((p) => p.y);
    const lambdas = lambdaGrid(opts.count || 18);
    let best = { mse: Infinity, model: null };
    for (const b of Object.values(bases)) {
      for (let k = b.min; k <= b.max; k++) {
        for (const pen of ['none', 'ridge', 'lasso']) {
          const m = normalizeModel({ basis: b.id, size: k, width: 1, penalty: pen });
          const prep = prepare(xs, ys, m);
          const { alpha } = lambdaAlpha(m);
          const grid = pen === 'none' ? [0] : lambdas;
          let warm = null;
          for (const li of order(grid)) {
            const w = solve(prep, grid[li], alpha, warm, { tol: 1e-5, maxSweeps: 60 });
            warm = w;
            const e = mse(toModel(prep, w, m), tx, ty);
            if (e < best.mse) {
              best = { mse: e, model: Object.assign({}, m, { logLambda: pen === 'none' ? m.logLambda : Math.log10(grid[li]) }) };
            }
          }
        }
      }
    }
    return best;
  }

  root.Reg = {
    DOMAIN,
    LOG_LAMBDA,
    bases,
    penalties,
    penaltyTex,
    presets,
    mulberry32,
    gauss,
    defaultModel,
    normalizeModel,
    lambdaAlpha,
    featureCount,
    design,
    prepare,
    solve,
    toModel,
    fit,
    mse,
    folds,
    cvCurve,
    cvAt,
    lambdaGrid,
    path,
    sizeCurve,
    sample,
    randomTarget,
    oracle,
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
