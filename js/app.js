// UI wiring: Explore mode (click-to-add scatter plot) and the Challenge.
(function () {
  'use strict';

  const R = window.Reg;
  const $ = (id) => document.getElementById(id);
  const [X0, X1] = R.DOMAIN.x;
  const [Y0, Y1] = R.DOMAIN.y;

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v ? JSON.parse(v) : fallback;
      } catch (e) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (e) {
        /* storage unavailable: ignore */
      }
    },
  };

  const cssColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function fmt(v, digits = 3) {
    if (v == null || Number.isNaN(v)) return '–';
    if (v === Infinity) return '∞';
    if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3)) return v.toExponential(1).replace('e', 'e');
    return String(Number(v.toPrecision(digits)));
  }

  function fmtLambda(logL) {
    const v = Math.pow(10, logL);
    if (v >= 100) return v.toFixed(0);
    if (v >= 1) return String(Number(v.toPrecision(3)));
    if (v >= 1e-3) return String(Number(v.toPrecision(2)));
    return v.toExponential(1);
  }

  function renderTex(el, tex) {
    if (window.katex) {
      try {
        window.katex.render(tex, el, { displayMode: true, throwOnError: false });
        el.classList.remove('raw');
        return;
      } catch (e) {
        /* fall through to raw text */
      }
    }
    el.textContent = tex;
    el.classList.add('raw');
  }

  function rafThrottle(fn) {
    let queued = false;
    return () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        fn();
      });
    };
  }

  function debounce(fn, ms) {
    let t = null;
    return () => {
      clearTimeout(t);
      t = setTimeout(fn, ms);
    };
  }

  function legendHtml(items) {
    return items
      .map(({ label, color, dashed, dot }) =>
        dot
          ? `<span><i class="dot" style="background:${color}"></i>${label}</span>`
          : `<span><i class="${dashed ? 'dashed' : ''}" style="background:${color};border-color:${color}"></i>${label}</span>`)
      .join('');
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function curve(fn, n = 400) {
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = X0 + (i / (n - 1)) * (X1 - X0);
      ys[i] = fn(xs[i]);
    }
    return { xs, ys };
  }

  function modelSummary(m) {
    const b = R.bases[m.basis];
    let s = `${b.name} · ${b.sizeLabel.toLowerCase()} ${m.size}`;
    if (m.basis === 'rbf') s += ` · width ${fmt(m.width, 2)}`;
    if (m.penalty === 'none') return `${s} · no penalty`;
    s += ` · ${R.penalties[m.penalty].name} λ = ${fmtLambda(m.logLambda)}`;
    if (m.penalty === 'elastic') s += `, α = ${fmt(m.alpha, 2)}`;
    return s;
  }

  // ---------------------------------------------------------------------------
  // Controls
  // ---------------------------------------------------------------------------

  // Slider + number input. `toInput`/`fromInput` convert between the slider's
  // value and what the number box shows (used for the log-scale λ).
  function slider(spec) {
    const wrap = el('div', 'param');
    const label = el('label', 'param-label', spec.label);
    const row = el('div', 'param-row');
    const range = document.createElement('input');
    range.type = 'range';
    range.min = spec.min;
    range.max = spec.max;
    range.step = spec.step;
    range.setAttribute('aria-label', spec.label);
    const num = document.createElement('input');
    num.type = 'number';
    num.step = spec.inputStep || spec.step;
    num.setAttribute('aria-label', `${spec.label} value`);
    const toInput = spec.toInput || ((v) => v);
    const fromInput = spec.fromInput || ((v) => v);
    range.addEventListener('input', () => {
      spec.set(Number(range.value));
      sync(false);
    });
    num.addEventListener('input', () => {
      const v = Number(num.value);
      if (num.value === '' || !isFinite(v)) return;
      const sv = fromInput(v);
      if (!isFinite(sv)) return;
      spec.set(clamp(sv, spec.min, spec.max));
      sync(true);
    });
    num.addEventListener('change', () => sync(false));
    row.append(range, num);
    wrap.append(label, row);
    function sync(skipNum) {
      const v = spec.get();
      range.value = v;
      if (!skipNum) num.value = toInput(v);
      if (spec.labelFn) label.textContent = spec.labelFn();
    }
    sync(false);
    return { el: wrap, sync, label };
  }

  function segmented(options, get, set, ariaLabel) {
    const g = el('div', 'segmented');
    g.setAttribute('role', 'radiogroup');
    g.setAttribute('aria-label', ariaLabel);
    const buttons = options.map(([value, text]) => {
      const b = el('button', null, text);
      b.type = 'button';
      b.dataset.value = value;
      b.addEventListener('click', () => {
        set(value);
        sync();
      });
      g.append(b);
      return b;
    });
    function sync() {
      for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.value === get()));
    }
    sync();
    return { el: g, sync };
  }

  // Features + penalty controls, shared by both modes. Mutates `model` in place.
  function buildModelControls(container, model, onChange, extras = {}) {
    container.innerHTML = '';
    const changed = (kind) => {
      Object.assign(model, R.normalizeModel(model));
      syncAll();
      onChange(kind);
    };

    container.append(el('h3', 'label', 'Features'));
    const basisSeg = segmented(
      Object.values(R.bases).map((b) => [b.id, b.id === 'rbf' ? 'RBF' : b.name]),
      () => model.basis,
      (v) => {
        if (v === model.basis) return;
        model.basis = v;
        model.size = R.bases[v].initial;
        changed('model');
      },
      'Basis functions'
    );
    container.append(basisSeg.el);
    const basisTex = el('div', 'formula small');
    container.append(basisTex);

    const params = el('div', 'params');
    const size = slider({
      label: 'Degree',
      labelFn: () => R.bases[model.basis].sizeLabel,
      min: 1,
      max: 15,
      step: 1,
      get: () => model.size,
      set: (v) => {
        model.size = Math.round(v);
        changed('model');
      },
    });
    const width = slider({
      label: 'Width (× spacing)',
      min: 0.25,
      max: 3,
      step: 0.05,
      get: () => model.width,
      set: (v) => {
        model.width = v;
        changed('model');
      },
    });
    params.append(size.el, width.el);
    container.append(params);
    let bestSizeBtn = null;
    if (extras.onBestSize) {
      bestSizeBtn = el('button', 'link-btn', 'Use CV-best size');
      bestSizeBtn.type = 'button';
      bestSizeBtn.addEventListener('click', extras.onBestSize);
      const r = el('div', 'link-row');
      r.append(bestSizeBtn);
      container.append(r);
    }

    container.append(el('h3', 'label', 'Penalty'));
    const penSeg = segmented(
      Object.values(R.penalties).map((p) => [p.id, p.name]),
      () => model.penalty,
      (v) => {
        model.penalty = v;
        changed('model');
      },
      'Penalty'
    );
    container.append(penSeg.el);
    const desc = el('p', 'desc');
    const tex = el('div', 'formula');
    container.append(desc, tex);

    const penParams = el('div', 'params');
    const lambda = slider({
      label: 'Strength λ',
      min: R.LOG_LAMBDA.min,
      max: R.LOG_LAMBDA.max,
      step: 0.05,
      inputStep: 'any',
      get: () => model.logLambda,
      set: (v) => {
        model.logLambda = v;
        changed('lambda');
      },
      toInput: (v) => fmtLambda(v),
      fromInput: (v) => (v > 0 ? Math.log10(v) : NaN),
    });
    const alpha = slider({
      label: 'L1 ratio α',
      min: 0.05,
      max: 0.95,
      step: 0.05,
      get: () => model.alpha,
      set: (v) => {
        model.alpha = v;
        changed('model');
      },
    });
    penParams.append(lambda.el, alpha.el);
    container.append(penParams);
    let bestLambdaBtn = null;
    if (extras.onBestLambda) {
      bestLambdaBtn = el('button', 'link-btn', 'Use CV-best λ');
      bestLambdaBtn.type = 'button';
      bestLambdaBtn.addEventListener('click', extras.onBestLambda);
      const r = el('div', 'link-row');
      r.append(bestLambdaBtn);
      container.append(r);
    }

    let lastBasis = null;
    let lastSize = null;
    let lastPenalty = null;
    function syncAll() {
      const b = R.bases[model.basis];
      basisSeg.sync();
      penSeg.sync();
      const range = size.el.querySelectorAll('input');
      range.forEach((i) => {
        i.min = b.min;
        i.max = b.max;
      });
      size.sync();
      width.sync();
      lambda.sync();
      alpha.sync();
      width.el.hidden = model.basis !== 'rbf';
      lambda.el.hidden = model.penalty === 'none';
      alpha.el.hidden = model.penalty !== 'elastic';
      if (bestLambdaBtn) bestLambdaBtn.parentElement.hidden = model.penalty === 'none';
      if (lastBasis !== model.basis || lastSize !== model.size) {
        renderTex(basisTex, b.tex(model.size));
        lastBasis = model.basis;
        lastSize = model.size;
      }
      if (lastPenalty !== model.penalty) {
        desc.textContent = R.penalties[model.penalty].desc;
        renderTex(tex, R.penaltyTex[model.penalty]);
        lastPenalty = model.penalty;
      }
    }
    syncAll();
    return {
      sync: syncAll,
      retex() {
        lastBasis = lastPenalty = null;
        syncAll();
      },
    };
  }

  function readoutHtml(items) {
    return items
      .map(({ label, value, title, accent }) => `<div${title ? ` title="${title}"` : ''}><dt>${label}</dt><dd${accent ? ' class="accent"' : ''}>${value}</dd></div>`)
      .join('');
  }

  // ---------------------------------------------------------------------------
  // Scatter rendering (shared)
  // ---------------------------------------------------------------------------

  function scatterSpec(o) {
    const c1 = cssColor('--series-1');
    const ink = cssColor('--series-2');
    const muted = cssColor('--series-3');
    const series = [];
    const legend = [];
    if (o.test) {
      series.push({ kind: 'points', xs: o.test.map((p) => p.x), ys: o.test.map((p) => p.y), color: muted, hollow: true, radius: 2.5 });
      legend.push({ label: 'Test points', color: muted, dot: true });
    }
    if (o.truth) {
      const t = curve(o.truth);
      series.push({ kind: 'line', xs: t.xs, ys: t.ys, color: muted, width: 1.5, dash: [2, 5] });
      legend.push({ label: 'True function', color: muted, dashed: true });
    }
    if (o.reference) {
      const t = curve(o.reference.predict);
      series.push({ kind: 'line', xs: t.xs, ys: t.ys, color: ink, width: 1.5, dash: [7, 5], alpha: 0.55 });
      legend.push({ label: o.referenceLabel, color: ink, dashed: true });
    }
    const pts = o.points;
    if (o.fit && o.residuals && pts.length) {
      series.push({
        kind: 'segments',
        x1: pts.map((p) => p.x),
        y1: pts.map((p) => p.y),
        y2: pts.map((p) => o.fit.predict(p.x)),
        color: c1,
        width: 1,
        alpha: 0.7,
      });
    }
    if (o.fit && pts.length) {
      const f = curve(o.fit.predict, 600);
      series.push({ kind: 'line', xs: f.xs, ys: f.ys, color: c1, width: 2.5 });
      legend.push({ label: 'Fit', color: c1 });
    }
    series.push({
      kind: 'points',
      xs: pts.map((p) => p.x),
      ys: pts.map((p) => p.y),
      color: ink,
      radius: 4.5,
      highlight: o.highlight,
      highlightColor: c1,
    });
    return {
      spec: {
        x: { min: X0, max: X1 },
        y: { min: Y0, max: Y1 },
        series,
        crosshair: false,
        message: pts.length ? null : o.emptyMessage || null,
      },
      legend,
    };
  }

  function nearestPoint(plot, points, px, py, radius) {
    let best = -1;
    let bestD = radius * radius;
    for (let i = 0; i < points.length; i++) {
      const dx = plot.xToPx(points[i].x) - px;
      const dy = plot.yToPx(points[i].y) - py;
      const d = dx * dx + dy * dy;
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  function scatterTooltip(hover, plot, points, fit, truth) {
    const ink = cssColor('--series-2');
    const c1 = cssColor('--series-1');
    const sw = (c) => `<span class="sw" style="background:${c}"></span>`;
    const i = nearestPoint(plot, points, hover.px, hover.py, 10);
    if (i >= 0) {
      const p = points[i];
      const lines = [`${sw(ink)}<b>point</b> (${p.x.toFixed(2)}, ${p.y.toFixed(2)})`];
      if (fit) {
        const yh = fit.predict(p.x);
        lines.push(`${sw(c1)}<b>fit</b> ${yh.toFixed(2)}`, `<b>residual</b> ${(p.y - yh).toFixed(2)}`);
      }
      return lines;
    }
    if (!fit) return null;
    const lines = [`<b>x</b> ${hover.x.toFixed(2)}`, `${sw(c1)}<b>fit</b> ${fit.predict(hover.x).toFixed(2)}`];
    if (truth) lines.push(`<b>true</b> ${truth(hover.x).toFixed(2)}`);
    return lines;
  }

  // ---------------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------------

  const plots = [];
  const themeListeners = [];
  function applyTheme(theme) {
    if (theme) document.documentElement.setAttribute('data-theme', theme);
    else document.documentElement.removeAttribute('data-theme');
    themeListeners.forEach((fn) => fn());
  }
  $('theme-toggle').addEventListener('click', () => {
    const current =
      document.documentElement.getAttribute('data-theme') ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    store.set('rp-theme', next);
    applyTheme(next);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => themeListeners.forEach((fn) => fn()));

  // ---------------------------------------------------------------------------
  // Explore mode
  // ---------------------------------------------------------------------------

  const saved = store.get('rp-explore', {});
  const ex = {
    preset: R.presets[saved.preset] || saved.preset === 'blank' ? saved.preset : 'sine',
    seed: saved.seed || 1,
    n: saved.n || R.presets.sine.n,
    noise: saved.noise != null ? saved.noise : R.presets.sine.noise,
    points: Array.isArray(saved.points) ? saved.points.filter((p) => isFinite(p.x) && isFinite(p.y)) : null,
    model: R.normalizeModel(saved.model || { basis: 'poly', size: 9, penalty: 'ridge', logLambda: -2.5 }),
    showTruth: saved.showTruth !== false,
    showOLS: !!saved.showOLS,
    showResid: !!saved.showResid,
    errView: saved.errView === 'size' ? 'size' : 'lambda',
    undo: [],
    fit: null,
    ols: null,
    cv: null,
    path: null,
    sizes: null,
    drag: null,
    highlight: -1,
  };

  function saveExplore() {
    store.set('rp-explore', {
      preset: ex.preset,
      seed: ex.seed,
      n: ex.n,
      noise: ex.noise,
      points: ex.points,
      model: ex.model,
      showTruth: ex.showTruth,
      showOLS: ex.showOLS,
      showResid: ex.showResid,
      errView: ex.errView,
    });
  }
  const saveExploreSoon = debounce(saveExplore, 300);

  const exPlot = new window.Plot($('ex-canvas'), {
    margin: { left: 40 },
    onHover: (h) => (ex.drag ? null : scatterTooltip(h, exPlot, ex.points, ex.fit, truthFn())),
  });
  const coefPlot = new window.Plot($('coef-canvas'), { margin: { left: 44, bottom: 28 }, onHover: coefHover });
  const pathPlot = new window.Plot($('path-canvas'), { margin: { left: 44, bottom: 28 }, onHover: pathHover });
  const errPlot = new window.Plot($('err-canvas'), { margin: { left: 44, bottom: 28 }, onHover: errHover });
  plots.push(exPlot, coefPlot, pathPlot, errPlot);

  const truthFn = () => (R.presets[ex.preset] ? R.presets[ex.preset].f : null);

  // Data controls
  const presetSel = $('ex-preset');
  for (const [id, p] of Object.entries(R.presets)) presetSel.append(new Option(p.name, id));
  presetSel.append(new Option('Blank canvas', 'blank'));
  presetSel.value = ex.preset;

  const dataParams = $('ex-data-params');
  const nSlider = slider({
    label: 'Points',
    min: 5,
    max: 100,
    step: 1,
    get: () => ex.n,
    set: (v) => {
      ex.n = Math.round(v);
      regenerate();
    },
  });
  const noiseSlider = slider({
    label: 'Noise σ',
    min: 0,
    max: 2,
    step: 0.05,
    get: () => ex.noise,
    set: (v) => {
      ex.noise = v;
      regenerate();
    },
  });
  dataParams.append(nSlider.el, noiseSlider.el);

  function syncDataControls() {
    const blank = ex.preset === 'blank';
    nSlider.el.hidden = blank;
    noiseSlider.el.hidden = blank;
    $('ex-generate').hidden = blank;
    nSlider.sync();
    noiseSlider.sync();
    $('ex-show-truth').closest('label').hidden = blank;
  }

  function pushUndo() {
    ex.undo.push(ex.points.map((p) => ({ x: p.x, y: p.y })));
    if (ex.undo.length > 100) ex.undo.shift();
    $('ex-undo').disabled = false;
  }

  function regenerate(record) {
    if (record) pushUndo();
    const p = R.presets[ex.preset];
    if (!p) {
      ex.points = [];
    } else {
      ex.points = R.sample(p.f, ex.n, ex.noise, R.mulberry32(ex.seed), p.outliers || 0);
    }
    dataChanged();
  }

  presetSel.addEventListener('change', () => {
    pushUndo();
    ex.preset = presetSel.value;
    const p = R.presets[ex.preset];
    if (p) {
      ex.n = p.n;
      ex.noise = p.noise;
    }
    syncDataControls();
    regenerate();
  });
  $('ex-generate').addEventListener('click', () => {
    ex.seed = (Math.random() * 2 ** 31) >>> 0;
    regenerate(true);
  });
  $('ex-clear').addEventListener('click', () => {
    if (!ex.points.length) return;
    pushUndo();
    ex.points = [];
    ex.preset = 'blank';
    presetSel.value = 'blank';
    syncDataControls();
    dataChanged();
  });
  function undo() {
    const prev = ex.undo.pop();
    if (!prev) return;
    ex.points = prev;
    $('ex-undo').disabled = !ex.undo.length;
    dataChanged();
  }
  $('ex-undo').addEventListener('click', undo);
  $('ex-undo').disabled = true;
  document.addEventListener('keydown', (e) => {
    if ($('explore').hidden) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select') return;
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      undo();
    }
  });

  for (const [id, key] of [['ex-show-truth', 'showTruth'], ['ex-show-ols', 'showOLS'], ['ex-show-resid', 'showResid']]) {
    $(id).checked = ex[key];
    $(id).addEventListener('change', () => {
      ex[key] = $(id).checked;
      saveExplore();
      if (key === 'showOLS') ex.ols = null;
      renderExploreSoon();
    });
  }

  // Model controls
  const exControls = buildModelControls(
    $('ex-model'),
    ex.model,
    (kind) => {
      ex.ols = null;
      saveExploreSoon();
      if (kind === 'lambda') {
        renderExploreSoon();
        // The path does not depend on λ; only the size curve does.
        if (ex.errView === 'size') curvesSoon();
        else renderCurvesSoon();
      } else dataChanged(true);
    },
    {
      onBestLambda: () => {
        computeCurves();
        if (ex.path && ex.path.best >= 0) {
          ex.model.logLambda = ex.path.logLambdas[ex.path.best];
          exControls.sync();
          saveExplore();
          renderExploreSoon();
          if (ex.errView === 'size') curvesSoon();
          else renderCurvesSoon();
        }
      },
      onBestSize: () => {
        const xs = ex.points.map((p) => p.x);
        const ys = ex.points.map((p) => p.y);
        const sc = R.sizeCurve(xs, ys, ex.model);
        if (sc.best >= 0) {
          ex.model.size = sc.sizes[sc.best];
          exControls.sync();
          dataChanged(true);
        }
      },
    }
  );

  // Error view toggle
  const errButtons = document.querySelectorAll('[data-err]');
  function syncErrButtons() {
    errButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.err === ex.errView)));
    $('err-size-btn').textContent = `vs ${R.bases[ex.model.basis].sizeLabel.toLowerCase()}`;
  }
  errButtons.forEach((b) =>
    b.addEventListener('click', () => {
      ex.errView = b.dataset.err;
      syncErrButtons();
      saveExplore();
      computeCurves();
      renderCurves();
    })
  );

  function dataChanged(modelOnly) {
    if (!modelOnly) saveExploreSoon();
    ex.ols = null;
    ex.path = null;
    ex.sizes = null;
    renderExploreSoon();
    curvesSoon();
  }

  // Fit, CV and readouts (cheap, every frame)
  function renderExplore() {
    const xs = ex.points.map((p) => p.x);
    const ys = ex.points.map((p) => p.y);
    const n = xs.length;
    ex.fit = n ? R.fit(xs, ys, ex.model) : null;
    ex.cv = n ? R.cvAt(xs, ys, ex.model) : null;
    if (ex.showOLS && n && ex.model.penalty !== 'none' && !ex.ols) {
      ex.ols = R.fit(xs, ys, Object.assign({}, ex.model, { penalty: 'none' }));
    }

    const { spec, legend } = scatterSpec({
      points: ex.points,
      fit: ex.fit,
      truth: ex.showTruth ? truthFn() : null,
      reference: ex.showOLS && ex.model.penalty !== 'none' ? ex.ols : null,
      referenceLabel: 'Unregularized',
      residuals: ex.showResid,
      highlight: ex.highlight,
      emptyMessage: 'Click anywhere to add a point',
    });
    exPlot.set(spec);
    $('ex-legend').innerHTML = legendHtml(legend);

    const f = ex.fit;
    const p = R.featureCount(ex.model);
    $('ex-readout').innerHTML = readoutHtml([
      { label: 'Points', value: String(n) },
      { label: 'Train MSE', value: f ? fmt(f.trainMSE) : '–' },
      {
        label: ex.cv ? (ex.cv.k === n ? 'LOO-CV MSE' : `${ex.cv.k}-fold CV MSE`) : 'CV MSE',
        value: ex.cv ? fmt(ex.cv.mean) : '–',
        title: ex.cv ? `± ${fmt(ex.cv.se)} (1 SE)` : 'needs at least 3 points',
        accent: true,
      },
      { label: 'R²', value: f && isFinite(f.r2) ? f.r2.toFixed(3) : '–' },
      { label: 'Eff. df', value: f ? f.df.toFixed(1) : '–', title: 'trace of the hat matrix, incl. intercept' },
      { label: 'Nonzero', value: f ? `${f.nonzero}/${p}` : '–' },
    ]);
    renderCoefs();
  }
  const renderExploreSoon = rafThrottle(renderExplore);

  // Regularization path and error curves (expensive, debounced)
  function computeCurves() {
    const xs = ex.points.map((p) => p.x);
    const ys = ex.points.map((p) => p.y);
    if (!ex.path && ex.model.penalty !== 'none' && xs.length >= 2) {
      ex.path = R.path(xs, ys, ex.model, { count: 36, tol: 1e-5, maxSweeps: 100 });
    }
    if (ex.errView === 'size' && xs.length >= 3) {
      ex.sizes = R.sizeCurve(xs, ys, ex.model, { tol: 1e-5, maxSweeps: 100 });
    }
  }
  const curvesSoon = debounce(() => {
    ex.sizes = null;
    computeCurves();
    renderCurves();
  }, 90);

  function renderCoefs() {
    const f = ex.fit;
    const labels = R.bases[ex.model.basis].labels(ex.model.size);
    const c1 = cssColor('--series-1');
    const muted = cssColor('--series-3');
    if (!f) {
      coefPlot.set({ x: { min: 0.5, max: labels.length + 0.5, labels }, y: { min: -1, max: 1, zero: true }, series: [], message: 'Add some points' });
      $('coef-meta').textContent = '';
      return;
    }
    const w = Array.from(f.w);
    const m = Math.max(0.1, ...w.map(Math.abs)) * 1.15;
    coefPlot.set({
      x: { min: 0.5, max: labels.length + 0.5, labels },
      y: { min: -m, max: m, zero: true },
      series: [{ kind: 'bars', ys: w, color: c1, zeroColor: muted }],
    });
    $('coef-meta').textContent = `L1 ${fmt(f.l1)} · L2 ${fmt(f.l2)}`;
  }

  function coefHover(h) {
    const f = ex.fit;
    if (!f) return null;
    const i = Math.round(h.x) - 1;
    if (i < 0 || i >= f.w.length) return null;
    const labels = R.bases[ex.model.basis].labels(ex.model.size);
    return [`<b>${labels[i]}</b> ${fmt(f.w[i], 4)}`];
  }

  const renderCurvesSoon = rafThrottle(() => renderCurves());

  function renderCurves() {
    syncErrButtons();
    const c1 = cssColor('--series-1');
    const ink = cssColor('--series-2');
    const muted = cssColor('--series-3');
    const lx = { min: R.LOG_LAMBDA.min, max: R.LOG_LAMBDA.max, log: true };
    const cur = ex.model.logLambda;
    const n = ex.points.length;
    const nonePenalty = ex.model.penalty === 'none';
    const noPenaltyMsg = 'No penalty selected.\nPick ridge, lasso or elastic net.';

    // Path
    if (nonePenalty || !ex.path) {
      pathPlot.set({ x: lx, y: { min: -1, max: 1, zero: true }, series: [], message: nonePenalty ? noPenaltyMsg : n < 2 ? 'Add at least 2 points' : '' });
    } else {
      const P = ex.path;
      const p = P.coefs[0].length;
      const series = [];
      const all = [];
      for (let j = 0; j < p; j++) {
        const ys = P.coefs.map((w) => w[j]);
        ys.forEach((v) => all.push(Math.abs(v)));
        series.push({ kind: 'line', xs: P.lambdas, ys, color: ink, width: 1.25, alpha: 0.45 });
      }
      all.sort((a, b) => a - b);
      const cw = ex.fit ? Array.from(ex.fit.w) : [];
      const top = Math.max(0.1, all[Math.floor(0.92 * (all.length - 1))] || 0, ...cw.map(Math.abs)) * 1.15;
      if (cw.length) series.push({ kind: 'markers', xs: cw.map(() => Math.pow(10, cur)), ys: cw, color: c1, radius: 3 });
      pathPlot.set({
        x: lx,
        y: { min: -top, max: top, zero: true },
        series,
        vlines: [{ x: Math.pow(10, cur), color: c1, width: 1.5 }],
        crosshair: true,
      });
    }

    // Error curve
    if (ex.errView === 'size') {
      const b = R.bases[ex.model.basis];
      const xr = { min: b.min - 0.5, max: b.max + 0.5, integer: true };
      const S = ex.sizes;
      if (!S || n < 3) {
        errPlot.set({ x: xr, y: { min: 0, max: 1 }, series: [], message: n < 3 ? 'Add at least 3 points' : '' });
        return;
      }
      const finiteCv = S.cv.filter(isFinite);
      const yMax = Math.max(...S.train, Math.min(Math.max(...finiteCv), 4 * Math.min(...finiteCv))) * 1.1 || 1;
      errPlot.set({
        x: xr,
        y: { min: 0, max: yMax },
        series: [
          { kind: 'band', xs: S.sizes, lo: S.cv.map((v, i) => v - S.se[i]), hi: S.cv.map((v, i) => v + S.se[i]), color: c1, fill: 0.12 },
          { kind: 'line', xs: S.sizes, ys: S.train, color: ink, width: 1.5, alpha: 0.7 },
          { kind: 'line', xs: S.sizes, ys: S.cv, color: c1, width: 2 },
          { kind: 'markers', xs: [S.sizes[S.best]], ys: [S.cv[S.best]], color: c1, radius: 4 },
        ],
        vlines: [{ x: ex.model.size, color: muted, dash: [3, 3], label: `${b.sizeLabel.toLowerCase()} ${ex.model.size}` }],
      });
      return;
    }
    if (nonePenalty || !ex.path || !ex.path.cv) {
      errPlot.set({ x: lx, y: { min: 0, max: 1 }, series: [], message: nonePenalty ? noPenaltyMsg : n < 3 ? 'Add at least 3 points' : '' });
      return;
    }
    const P = ex.path;
    const cv = Array.from(P.cv.mean);
    const se = Array.from(P.cv.se);
    const minCv = Math.min(...cv);
    const yMax = Math.max(Math.max(...P.train), Math.min(Math.max(...cv), 4 * minCv)) * 1.1 || 1;
    errPlot.set({
      x: lx,
      y: { min: 0, max: yMax },
      series: [
        { kind: 'band', xs: P.lambdas, lo: cv.map((v, i) => v - se[i]), hi: cv.map((v, i) => v + se[i]), color: c1, fill: 0.12 },
        { kind: 'line', xs: P.lambdas, ys: Array.from(P.train), color: ink, width: 1.5, alpha: 0.7 },
        { kind: 'line', xs: P.lambdas, ys: cv, color: c1, width: 2 },
        { kind: 'markers', xs: [P.lambdas[P.best]], ys: [cv[P.best]], color: c1, radius: 4 },
      ],
      vlines: [{ x: Math.pow(10, cur), color: muted, dash: [3, 3], label: `λ ${fmtLambda(cur)}` }],
      crosshair: true,
    });
  }

  function nearestIndex(arr, v) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (Math.abs(arr[i] - v) < Math.abs(arr[best] - v)) best = i;
    return best;
  }

  function pathHover(h) {
    if (!ex.path || ex.model.penalty === 'none') return null;
    return [`<b>λ</b> ${fmtLambda(Math.log10(h.x))}`, 'click to set λ'];
  }

  function errHover(h) {
    if (ex.errView === 'size') {
      const S = ex.sizes;
      if (!S) return null;
      const i = nearestIndex(S.sizes, h.x);
      return [
        `<b>${R.bases[ex.model.basis].sizeLabel.toLowerCase()}</b> ${S.sizes[i]}`,
        `<b>train</b> ${fmt(S.train[i])}`,
        `<b>CV</b> ${fmt(S.cv[i])} ± ${fmt(S.se[i])}`,
        'click to set',
      ];
    }
    const P = ex.path;
    if (!P || !P.cv || ex.model.penalty === 'none') return null;
    const i = nearestIndex(P.logLambdas, Math.log10(h.x));
    return [
      `<b>λ</b> ${fmtLambda(P.logLambdas[i])}`,
      `<b>train</b> ${fmt(P.train[i])}`,
      `<b>CV</b> ${fmt(P.cv.mean[i])} ± ${fmt(P.cv.se[i])}`,
      'click to set λ',
    ];
  }

  // Clicking the λ charts sets λ; clicking the size chart sets the size.
  function bindChartClick(plot, isSize) {
    plot.canvas.addEventListener('click', (e) => {
      if (!plot.spec || plot.spec.message) return;
      const { px, py } = plot.localPoint(e);
      if (!plot.inPlot(px, py)) return;
      const { x } = plot.pxToData(px, py);
      if (isSize()) {
        const b = R.bases[ex.model.basis];
        ex.model.size = clamp(Math.round(x), b.min, b.max);
        exControls.sync();
        dataChanged(true);
      } else {
        ex.model.logLambda = clamp(Math.round(Math.log10(x) * 20) / 20, R.LOG_LAMBDA.min, R.LOG_LAMBDA.max);
        exControls.sync();
        saveExploreSoon();
        renderExploreSoon();
        renderCurvesSoon();
      }
    });
  }
  bindChartClick(pathPlot, () => false);
  bindChartClick(errPlot, () => ex.errView === 'size');

  // Click to add, drag to move, click a point to remove.
  const canvas = $('ex-canvas');
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !exPlot.spec) return;
    const { px, py } = exPlot.localPoint(e);
    if (!exPlot.inPlot(px, py, 6)) return;
    e.preventDefault();
    const radius = e.pointerType === 'touch' ? 18 : 10;
    const hit = nearestPoint(exPlot, ex.points, px, py, radius);
    pushUndo();
    if (hit >= 0) {
      ex.drag = { i: hit, px, py, moved: false, existing: true };
    } else {
      const d = exPlot.pxToData(px, py);
      ex.points.push({ x: clamp(d.x, X0, X1), y: clamp(d.y, Y0, Y1) });
      ex.drag = { i: ex.points.length - 1, px, py, moved: true, existing: false };
      dataChanged();
    }
    ex.highlight = ex.drag.i;
    exPlot.tooltip.hidden = true;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    renderExploreSoon();
  });
  canvas.addEventListener('pointermove', (e) => {
    const { px, py } = exPlot.localPoint(e);
    if (!ex.drag) {
      const hit = nearestPoint(exPlot, ex.points, px, py, 10);
      canvas.style.cursor = hit >= 0 ? 'grab' : 'crosshair';
      if (hit !== ex.highlight) {
        ex.highlight = hit;
        renderExploreSoon();
      }
      return;
    }
    const d = ex.drag;
    if (!d.moved && Math.hypot(px - d.px, py - d.py) > 4) d.moved = true;
    if (!d.moved) return;
    const v = exPlot.pxToData(px, py);
    ex.points[d.i] = { x: clamp(v.x, X0, X1), y: clamp(v.y, Y0, Y1) };
    dataChanged();
  });
  function endDrag(e) {
    const d = ex.drag;
    if (!d) return;
    ex.drag = null;
    if (d.existing && !d.moved) {
      ex.points.splice(d.i, 1);
      ex.highlight = -1;
      dataChanged();
    }
    canvas.style.cursor = 'crosshair';
    if (e && canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    saveExplore();
  }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('pointerleave', () => {
    if (!ex.drag && ex.highlight >= 0) {
      ex.highlight = -1;
      renderExploreSoon();
    }
  });
  canvas.addEventListener('contextmenu', (e) => {
    // Right-click removes a point too.
    const { px, py } = exPlot.localPoint(e);
    const hit = nearestPoint(exPlot, ex.points, px, py, 10);
    if (hit < 0) return;
    e.preventDefault();
    pushUndo();
    ex.points.splice(hit, 1);
    ex.highlight = -1;
    dataChanged();
  });

  // ---------------------------------------------------------------------------
  // Challenge mode
  // ---------------------------------------------------------------------------

  const c = {
    model: R.normalizeModel(store.get('rp-challenge-model', { basis: 'poly', size: 3, penalty: 'none' })),
    round: null,
    submitted: false,
    oracle: null,
    oracleFit: null,
    fit: null,
    stats: Object.assign({ score: 0, won: 0, streak: 0, best: 0 }, store.get('rp-challenge-stats', {})),
  };
  const N_TEST = 200;

  const cPlot = new window.Plot($('c-canvas'), {
    margin: { left: 40 },
    onHover: (h) => (c.round ? scatterTooltip(h, cPlot, c.round.train, c.fit, c.submitted ? c.round.target.f : null) : null),
  });
  plots.push(cPlot);

  const cControls = buildModelControls($('c-model'), c.model, () => {
    store.set('rp-challenge-model', c.model);
    renderChallengeSoon();
  });

  for (const id of ['c-n', 'c-noise', 'c-threshold']) {
    const saved = store.get(`rp-${id}`, null);
    if (saved != null) $(id).value = saved;
    $(id).addEventListener('change', () => {
      store.set(`rp-${id}`, $(id).value);
      if (id !== 'c-threshold' && !c.submitted) newRound();
    });
  }

  function newRound() {
    const seed = (Math.random() * 2 ** 31) >>> 0;
    const rng = R.mulberry32(seed);
    const target = R.randomTarget(rng);
    const n = Number($('c-n').value);
    const noise = Number($('c-noise').value);
    c.round = { seed, target, noise, train: R.sample(target.f, n, noise, rng), test: R.sample(target.f, N_TEST, noise, rng) };
    c.submitted = false;
    c.oracle = null;
    c.oracleFit = null;
    $('c-result').hidden = true;
    $('c-submit').disabled = false;
    $('c-hint').textContent = 'Training data only · test points appear after you submit';
    renderChallenge();
    // Precompute the reference score in the background.
    const round = c.round;
    setTimeout(() => {
      if (c.round === round && !c.oracle) computeOracle();
    }, 400);
  }

  function computeOracle() {
    const r = c.round;
    c.oracle = R.oracle(r.train, r.test);
    c.oracleFit = R.fit(r.train.map((p) => p.x), r.train.map((p) => p.y), c.oracle.model);
  }

  function testMSE(fit) {
    const t = c.round.test;
    return R.mse(fit, t.map((p) => p.x), t.map((p) => p.y));
  }

  function renderChallenge() {
    if (!c.round) return;
    const r = c.round;
    const xs = r.train.map((p) => p.x);
    const ys = r.train.map((p) => p.y);
    c.fit = R.fit(xs, ys, c.model);
    const cv = R.cvAt(xs, ys, c.model);
    const { spec, legend } = scatterSpec({
      points: r.train,
      fit: c.fit,
      test: c.submitted ? r.test : null,
      truth: c.submitted ? r.target.f : null,
      reference: c.submitted ? c.oracleFit : null,
      referenceLabel: 'Best on test',
      residuals: false,
    });
    cPlot.set(spec);
    $('c-legend').innerHTML = legendHtml(legend);
    $('c-cv').textContent = cv ? fmt(cv.mean) : '–';

    const fb = $('c-feedback');
    if (cv && cv.mean > 2 * c.fit.trainMSE && cv.mean - c.fit.trainMSE > 0.2) {
      fb.textContent = `Training error is only ${fmt(c.fit.trainMSE)}. A gap this large usually means overfitting.`;
    } else {
      fb.textContent = "Lower is better. It's your only honest view of the hidden test set.";
    }

    const items = [
      { label: 'Train MSE', value: fmt(c.fit.trainMSE) },
      { label: 'CV MSE', value: cv ? fmt(cv.mean) : '–', accent: !c.submitted },
      { label: 'Eff. df', value: c.fit.df.toFixed(1) },
      { label: 'Test MSE', value: c.submitted ? fmt(testMSE(c.fit)) : 'hidden', accent: c.submitted },
    ];
    if (c.submitted) items.push({ label: 'Best on test', value: fmt(c.oracle.mse) });
    $('c-readout').innerHTML = readoutHtml(items);
  }
  const renderChallengeSoon = rafThrottle(renderChallenge);

  function submit() {
    if (!c.round || c.submitted) return;
    if (!c.oracle) computeOracle();
    c.submitted = true;
    const yours = testMSE(c.fit);
    const ratio = Math.min(1, c.oracle.mse / yours);
    const threshold = Number($('c-threshold').value);
    const win = ratio >= threshold;
    const points = Math.round(100 * ratio);
    const s = c.stats;
    s.score += win ? points : 0;
    if (win) {
      s.won += 1;
      s.streak += 1;
      s.best = Math.max(s.best, s.streak);
    } else s.streak = 0;
    store.set('rp-challenge-stats', s);
    renderStats();

    const r = c.round;
    const box = $('c-result');
    box.className = `result ${win ? 'win' : 'lose'}`;
    const headline = win
      ? ratio >= 0.99
        ? 'Spot <em>on</em>'
        : 'Nicely <em>generalized</em>'
      : yours > 3 * c.oracle.mse
        ? 'That one <em>wobbled</em>'
        : 'Close, but not <em>quite</em>';
    box.innerHTML = `
      <h3>${headline}</h3>
      <p>You reached <b>${points}%</b> of the best test score${win ? ` · +${points} points` : ''}.</p>
      <p>Your test MSE <b>${fmt(yours)}</b> · best found <b>${fmt(c.oracle.mse)}</b> · noise floor σ² <b>${fmt(r.noise ** 2)}</b></p>
      <p>Hidden function: <b>${r.target.name.toLowerCase()}</b>.</p>
      <p>Best setting: ${modelSummary(c.oracle.model)}. <button class="link-btn" id="c-try-best" type="button">Try it</button></p>
      <p class="meta">Keep tweaking: the test MSE now updates live.</p>`;
    box.hidden = false;
    $('c-try-best').addEventListener('click', () => {
      Object.assign(c.model, c.oracle.model);
      cControls.sync();
      renderChallenge();
    });
    $('c-submit').disabled = true;
    $('c-hint').textContent = 'Hollow dots are the 200 test points';
    renderChallenge();
  }

  function renderStats() {
    $('c-score').textContent = c.stats.score;
    $('c-won').textContent = c.stats.won;
    $('c-streak').textContent = c.stats.streak;
    $('c-best').textContent = c.stats.best;
  }

  $('c-submit').addEventListener('click', submit);
  $('c-new').addEventListener('click', () => {
    if (c.round && !c.submitted) {
      c.stats.streak = 0;
      store.set('rp-challenge-stats', c.stats);
      renderStats();
    }
    newRound();
  });

  // ---------------------------------------------------------------------------
  // Mode switching + routing
  // ---------------------------------------------------------------------------

  function setMode(mode) {
    $('explore').hidden = mode !== 'explore';
    $('challenge').hidden = mode !== 'challenge';
    document.querySelectorAll('.mode-switch button').forEach((b) => {
      b.setAttribute('aria-selected', String(b.dataset.mode === mode));
    });
    if (mode === 'challenge') {
      if (!c.round) newRound();
      else renderChallenge();
      cPlot.draw();
    } else {
      renderExplore();
      renderCurves();
      plots.forEach((p) => p.draw());
    }
  }

  document.querySelectorAll('.mode-switch button').forEach((b) =>
    b.addEventListener('click', () => {
      setMode(b.dataset.mode);
      history.replaceState(null, '', `#${b.dataset.mode}`);
    })
  );

  function route() {
    const hash = location.hash.replace(/^#\/?/, '');
    setMode(hash === 'challenge' ? 'challenge' : 'explore');
  }
  window.addEventListener('hashchange', route);

  themeListeners.push(() => {
    if (!$('explore').hidden) {
      renderExplore();
      renderCurves();
    } else renderChallenge();
  });

  // KaTeX loads asynchronously (defer); re-typeset once it is there.
  window.addEventListener('load', () => {
    exControls.retex();
    cControls.retex();
  });

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------

  const savedTheme = store.get('rp-theme', null);
  if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);
  syncDataControls();
  if (!ex.points) regenerate();
  renderStats();
  computeCurves();
  route();
})();
