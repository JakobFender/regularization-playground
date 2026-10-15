# Regularization Playground

An interactive, dependency-free web app for building intuition about regression, overfitting and regularization.
A sibling of [Distribution Playground](https://github.com/jakobfender/distribution-playground).

**Live demo:** https://jakobfender.github.io/regularization-playground/ (after enabling Pages, see below)

## Features

### Explore (`#explore`)
The center is a scatter plot: **click to add a point, drag to move it, click (or right-click) a point to remove it**. Ctrl/Cmd+Z undoes.

- **Data:** sine, line, cubic, step, narrow bump and line-with-outliers generators with adjustable point count and noise, or a blank canvas.
- **Features:** polynomial (degree 1–15), Gaussian RBF (1–20 centers, adjustable width) or Fourier (1–10 frequencies).
- **Penalty:** none (OLS), ridge, lasso or elastic net, with a log-scale λ slider and the L1 ratio α. The objective is shown as a formula.
- **Overlays:** true function, unregularized fit, residuals; hover for values.
- **Readout:** train MSE, cross-validated MSE, R², effective degrees of freedom (trace of the hat matrix), nonzero weights.
- **Under the hood:** coefficient bars, the full regularization path, and training vs. CV error as a function of λ or of model size. Click either chart to set λ / size, or jump to the CV-best value.

### Challenge (`#challenge`)
Points are drawn from a hidden function plus noise. Tune the model using only the training data (and its CV error), then submit.
You are scored on 200 hidden test points relative to the best setting a grid search finds on that test set; win at 80 / 90 / 97 % of best.
After submitting, the true function, test points and best setting are revealed and test MSE updates live.

## Math notes

- Features and the target are standardized; the intercept is never penalized.
- Objective (glmnet convention): `(1/2n)‖y − b − Φw‖² + λ(α‖w‖₁ + (1−α)/2 ‖w‖²)`.
- Ridge / OLS use a Cholesky solve. Lasso / elastic net use coordinate descent on the Gram matrix followed by an
  active-set polish, so solutions satisfy the KKT conditions exactly even for ill-conditioned polynomial features.
- 5-fold CV (leave-one-out below 10 points); folds are fixed by point index so they don't jump while dragging.

## Running locally

No build step. Serve the folder with any static server:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

Run the math checks (OLS closed form, interpolation, KKT optimality, shrinkage, CV, oracle):

```sh
node tests/check.js
```

## Deploying to GitHub Pages

The workflow in `.github/workflows/pages.yml` runs the checks and deploys on every push to `main` (or manually via *Run workflow*).
One-time setup: in the repository go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

## Layout

| File | Purpose |
|---|---|
| `js/regression.js` | Bases, ridge / lasso / elastic-net solvers, CV, paths, data generators, challenge oracle |
| `js/plot.js` | Small canvas plotter (lines, points, bars, bands, log axes, hover) |
| `js/app.js` | Explore and challenge UI |
| `tests/check.js` | Node sanity checks |

Formulas are typeset with [KaTeX](https://katex.org/) from a CDN; if it can't load, the raw TeX is shown instead.
