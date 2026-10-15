// Small canvas plotter: lines, points, residual segments, bands, bars,
// vertical markers, optional log axes, and a hover tooltip.
(function (root) {
  'use strict';

  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / mag;
    const nice = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
    return nice * mag;
  }

  function linearTicks(lo, hi, target) {
    const step = niceStep(hi - lo, target);
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return { values: out, step };
  }

  function fmtTick(v, step) {
    if (v === 0) return '0';
    if (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3) return v.toExponential(0);
    const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    return v.toFixed(Math.min(dec, 6));
  }

  // Decades for log axes (values are already log10).
  function logTicks(lo, hi, target) {
    const span = hi - lo;
    const every = Math.max(1, Math.ceil(span / Math.max(1, target)));
    const out = [];
    for (let e = Math.ceil(lo - 1e-9); e <= hi + 1e-9; e++) if (e % every === 0) out.push(e);
    return out;
  }

  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  function fmtPow(e) {
    if (e === 0) return '1';
    if (e === 1) return '10';
    return '10' + String(e).split('').map((c) => SUP[c]).join('');
  }

  function cssVar(el, name) {
    return getComputedStyle(el).getPropertyValue(name).trim();
  }

  function withAlpha(color, alpha) {
    if (color.startsWith('#') && color.length === 7) {
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }
    return color;
  }

  class Plot {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.spec = null;
      this.hover = null; // { x, y, px, py } in data / pixel coords
      this.onHover = opts.onHover || null; // (hover) => [lines] | null
      this.margin = Object.assign({ top: 12, right: 12, bottom: 32, left: 48 }, opts.margin || {});
      this.tooltip = document.createElement('div');
      this.tooltip.className = 'plot-tooltip';
      this.tooltip.hidden = true;
      canvas.parentElement.appendChild(this.tooltip);

      new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
      canvas.addEventListener('pointermove', (e) => this._move(e));
      canvas.addEventListener('pointerleave', () => this.clearHover());
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.draw());
      if (document.fonts) document.fonts.ready.then(() => this.draw());
    }

    set(spec) {
      this.spec = spec;
      this.draw();
      if (this.hover) this._updateTooltip();
    }

    // Axis value -> plot coordinate (log axes take raw values and map through log10).
    _t(axis, v) {
      return axis.log ? Math.log10(v) : v;
    }

    xToPx(x) {
      const a = this.spec.x;
      return this.margin.left + ((this._t(a, x) - a.min) / (a.max - a.min)) * this.plotW;
    }

    yToPx(y) {
      const a = this.spec.y;
      return this.margin.top + this.plotH * (1 - (this._t(a, y) - a.min) / (a.max - a.min));
    }

    pxToData(px, py) {
      const ax = this.spec.x;
      const ay = this.spec.y;
      let x = ax.min + ((px - this.margin.left) / this.plotW) * (ax.max - ax.min);
      let y = ay.min + (1 - (py - this.margin.top) / this.plotH) * (ay.max - ay.min);
      if (ax.log) x = Math.pow(10, x);
      if (ay.log) y = Math.pow(10, y);
      return { x, y };
    }

    inPlot(px, py, pad = 0) {
      const m = this.margin;
      return px >= m.left - pad && px <= m.left + this.plotW + pad && py >= m.top - pad && py <= m.top + this.plotH + pad;
    }

    localPoint(e) {
      const rect = this.canvas.getBoundingClientRect();
      return { px: e.clientX - rect.left, py: e.clientY - rect.top };
    }

    clearHover() {
      this.hover = null;
      this.tooltip.hidden = true;
      this.draw();
    }

    _move(e) {
      if (!this.spec || !this.plotW) return;
      const { px, py } = this.localPoint(e);
      if (!this.inPlot(px, py)) {
        if (this.hover) this.clearHover();
        return;
      }
      this.hover = Object.assign({ px, py }, this.pxToData(px, py));
      this.draw();
      this._updateTooltip();
    }

    _updateTooltip() {
      const lines = this.onHover && this.hover ? this.onHover(this.hover) : null;
      if (!lines || !lines.length) {
        this.tooltip.hidden = true;
        return;
      }
      this.tooltip.innerHTML = lines.join('<br>');
      this.tooltip.hidden = false;
      const { px, py } = this.hover;
      const w = this.tooltip.offsetWidth;
      const h = this.tooltip.offsetHeight;
      const left = px + 16 + w > this.width ? px - 16 - w : px + 16;
      const top = Math.min(Math.max(0, py - h - 12), this.height - h);
      this.tooltip.style.left = `${Math.max(0, left)}px`;
      this.tooltip.style.top = `${top}px`;
    }

    draw() {
      const canvas = this.canvas;
      const parent = canvas.parentElement;
      const dpr = window.devicePixelRatio || 1;
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      if (!width || !height) return;
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
      this.width = width;
      this.height = height;
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (!this.spec) return;

      const m = this.margin;
      this.plotW = width - m.left - m.right;
      this.plotH = height - m.top - m.bottom;
      const spec = this.spec;
      const colors = {
        grid: cssVar(canvas, '--grid'),
        axis: cssVar(canvas, '--axis'),
        text: cssVar(canvas, '--text-muted'),
        surface: cssVar(canvas, '--surface'),
        ink: cssVar(canvas, '--series-2'),
      };
      ctx.font = "11px 'Martian Mono', ui-monospace, monospace";
      ctx.lineWidth = 1;

      // Y grid + labels
      ctx.fillStyle = colors.text;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const yTarget = Math.max(2, Math.floor(this.plotH / 48));
      const yVals = spec.y.log
        ? logTicks(spec.y.min, spec.y.max, yTarget).map((e) => [Math.pow(10, e), fmtPow(e)])
        : (() => {
            const t = linearTicks(spec.y.min, spec.y.max, yTarget);
            return t.values.map((v) => [v, fmtTick(v, t.step)]);
          })();
      for (const [v, label] of yVals) {
        const y = Math.round(this.yToPx(v)) + 0.5;
        ctx.strokeStyle = spec.y.zero && v === 0 ? colors.axis : colors.grid;
        ctx.beginPath();
        ctx.moveTo(m.left, y);
        ctx.lineTo(m.left + this.plotW, y);
        ctx.stroke();
        ctx.fillText(label, m.left - 8, y);
      }

      // X ticks + labels
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      let xVals;
      if (spec.x.labels) {
        const n = spec.x.labels.length;
        const every = Math.max(1, Math.ceil((n * 34) / this.plotW));
        xVals = spec.x.labels.map((l, i) => [i + 1, l]).filter((_, i) => i % every === 0);
      } else if (spec.x.log) {
        xVals = logTicks(spec.x.min, spec.x.max, Math.max(2, Math.floor(this.plotW / 56))).map((e) => [Math.pow(10, e), fmtPow(e)]);
      } else {
        const t = linearTicks(spec.x.min, spec.x.max, Math.max(2, Math.floor(this.plotW / 72)));
        xVals = t.values.map((v) => [v, spec.x.integer ? String(Math.round(v)) : fmtTick(v, t.step)]);
      }
      ctx.strokeStyle = colors.axis;
      for (const [v, label] of xVals) {
        const x = Math.round(this.xToPx(v)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, m.top + this.plotH);
        ctx.lineTo(x, m.top + this.plotH + 5);
        ctx.stroke();
        ctx.fillText(label, x, m.top + this.plotH + 8);
      }

      // Series, clipped to the plot area
      ctx.save();
      ctx.beginPath();
      ctx.rect(m.left, m.top, this.plotW, this.plotH);
      ctx.clip();
      for (const s of spec.series || []) this._drawSeries(s, colors);
      for (const v of spec.vlines || []) {
        const x = Math.round(this.xToPx(v.x)) + 0.5;
        ctx.strokeStyle = v.color;
        ctx.lineWidth = v.width || 1;
        ctx.setLineDash(v.dash || []);
        ctx.beginPath();
        ctx.moveTo(x, m.top);
        ctx.lineTo(x, m.top + this.plotH);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore();

      // Baseline axis
      ctx.strokeStyle = colors.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const by = Math.round(m.top + this.plotH) + 0.5;
      ctx.moveTo(m.left, by);
      ctx.lineTo(m.left + this.plotW, by);
      ctx.stroke();

      // Vertical marker labels (above the clip so they stay readable)
      for (const v of spec.vlines || []) {
        if (!v.label) continue;
        const x = this.xToPx(v.x);
        ctx.fillStyle = v.color;
        ctx.textBaseline = 'top';
        const w = ctx.measureText(v.label).width;
        const right = x + 6 + w <= m.left + this.plotW;
        ctx.textAlign = right ? 'left' : 'right';
        ctx.fillText(v.label, right ? x + 6 : x - 6, m.top + 4);
      }

      if (spec.message) {
        ctx.fillStyle = colors.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = "13px 'Onest', system-ui, sans-serif";
        const lines = spec.message.split('\n');
        lines.forEach((line, i) => {
          ctx.fillText(line, m.left + this.plotW / 2, m.top + this.plotH / 2 + (i - (lines.length - 1) / 2) * 20);
        });
      }

      if (spec.crosshair && this.hover) {
        const x = Math.round(this.hover.px) + 0.5;
        ctx.strokeStyle = colors.text;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, m.top);
        ctx.lineTo(x, m.top + this.plotH);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    _path(xs, ys) {
      const ctx = this.ctx;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        if (!isFinite(ys[i])) {
          pen = false;
          continue;
        }
        const px = this.xToPx(xs[i]);
        // Keep huge values from producing absurd pixel coordinates.
        const py = Math.max(-1e4, Math.min(1e4, this.yToPx(ys[i])));
        if (pen) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        pen = true;
      }
    }

    _drawSeries(s, colors) {
      const ctx = this.ctx;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.setLineDash(s.dash || []);
      ctx.globalAlpha = s.alpha == null ? 1 : s.alpha;

      if (s.kind === 'line') {
        this._path(s.xs, s.ys);
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width || 2;
        ctx.stroke();
      } else if (s.kind === 'band') {
        ctx.beginPath();
        for (let i = 0; i < s.xs.length; i++) {
          const px = this.xToPx(s.xs[i]);
          const py = this.yToPx(s.hi[i]);
          if (i) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
        }
        for (let i = s.xs.length - 1; i >= 0; i--) ctx.lineTo(this.xToPx(s.xs[i]), this.yToPx(Math.max(s.lo[i], 1e-300)));
        ctx.closePath();
        ctx.fillStyle = withAlpha(s.color, s.fill || 0.12);
        ctx.fill();
      } else if (s.kind === 'segments') {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width || 1;
        ctx.beginPath();
        for (let i = 0; i < s.x1.length; i++) {
          const x = Math.round(this.xToPx(s.x1[i])) + 0.5;
          ctx.moveTo(x, this.yToPx(s.y1[i]));
          ctx.lineTo(x, this.yToPx(s.y2[i]));
        }
        ctx.stroke();
      } else if (s.kind === 'points') {
        ctx.setLineDash([]);
        const r = s.radius || 4.5;
        for (let i = 0; i < s.xs.length; i++) {
          const big = s.highlight === i;
          const px = this.xToPx(s.xs[i]);
          const py = this.yToPx(s.ys[i]);
          ctx.beginPath();
          ctx.arc(px, py, big ? r + 2 : r, 0, 2 * Math.PI);
          if (s.hollow) {
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.25;
            ctx.stroke();
          } else {
            ctx.fillStyle = big && s.highlightColor ? s.highlightColor : s.color;
            ctx.fill();
            ctx.strokeStyle = colors.surface;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
      } else if (s.kind === 'bars') {
        ctx.setLineDash([]);
        const base = this.yToPx(0);
        const half = Math.min(0.36, s.half || 0.36);
        for (let i = 0; i < s.ys.length; i++) {
          const v = s.ys[i];
          const pl = this.xToPx(i + 1 - half);
          const pr = this.xToPx(i + 1 + half);
          const pt = Math.max(-1e4, Math.min(1e4, this.yToPx(v)));
          if (Math.abs(v) < 1e-10) {
            // A short tick marks an exact zero (the point of the lasso).
            ctx.strokeStyle = s.zeroColor || s.color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pl, Math.round(base) + 0.5);
            ctx.lineTo(pr, Math.round(base) + 0.5);
            ctx.stroke();
            continue;
          }
          const color = s.colors ? s.colors[i] : s.color;
          if (s.outline) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(pl + 0.5, Math.min(base, pt) + 0.5, pr - pl - 1, Math.abs(pt - base));
          } else {
            ctx.fillStyle = color;
            ctx.fillRect(pl, Math.min(base, pt), pr - pl, Math.max(1, Math.abs(pt - base)));
          }
        }
      } else if (s.kind === 'markers') {
        ctx.setLineDash([]);
        for (let i = 0; i < s.xs.length; i++) {
          ctx.beginPath();
          ctx.arc(this.xToPx(s.xs[i]), this.yToPx(s.ys[i]), s.radius || 3, 0, 2 * Math.PI);
          ctx.fillStyle = s.color;
          ctx.fill();
        }
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
  }

  root.Plot = Plot;
})(typeof globalThis !== 'undefined' ? globalThis : this);
