# QML Playground – web app

A fully client-side rewrite of the QML Playground. The quantum state-vector simulator, training and all
visualizations run in the browser, so the app is a static site that can be hosted on GitHub Pages.
It supports one and two qubits, classification and regression, L1/L2 regularization and Monte Carlo
uncertainty, the same as the legacy Dash app in `../legacy`.

## Development

Requires Node.js 20+.

```bash
cd web
npm install
npm run dev        # http://localhost:5173/
npm test           # Vitest: PyTorch parity, gradients, trainer
npm run lint
npm run build      # static site in dist/
npm run preview    # serve dist/
```

To build for a sub-path, set `VITE_BASE`:
```bash
VITE_BASE=/qml-playground/ npm run build && VITE_BASE=/qml-playground/ npm run preview
# → http://localhost:4173/qml-playground/
```
In Git Bash on Windows, prefix with `MSYS_NO_PATHCONV=1`. Otherwise MSYS rewrites `/qml-playground/` into a Windows path.

## Deployment

`.github/workflows/pages.yml` builds, tests and deploys `web/` on every push to `main` that touches it.
The base path comes from the repository's own Pages configuration, so the same code deploys to
`pdebus.github.io/qml-playground/` and `fraunhofer-aisec.github.io/qml-playground/` without changes.
To enable deployment on a repository:

1. Go to *Settings → Pages → Build and deployment*, and set *Source* to **GitHub Actions**.
2. Optional: set the repository variable `PAGES_BASE_PATH` to override the base path, e.g. `/` for a custom
   domain. A `CNAME` file for a custom domain goes into `web/public/`.

## Architecture

```
src/sim/      complex 2×2 gates, Rot + analytic derivatives, the re-uploading circuit with adjoint backward pass
src/model/    losses (fidelity, NLL, MSE on ⟨Z⟩) with ∂ℓ/∂ψ, Adam (torch semantics), regularization, QuantumModel
src/data/     seeded RNG, dataset generators, single-qubit target states
src/worker/   Trainer (model + replay history, no DOM deps) and the Web Worker loop / message protocol
src/state/    Zustand store bridging UI and worker, settings model, URL hash encoding
src/viz/      Plotly partial bundle + figure builders (Bloch sphere, Q-simplex, metrics, …), unit-map rendering, marching squares, DFT
src/ui/       React panels and controls
tests/        Vitest suites; fixtures/reference.json is generated from the PyTorch app
```

- **Simulation.** States are interleaved `Float64Array`s. Every layer applies `Rot(φ,θ,ω) = RZ(ω)RY(θ)RZ(φ)` per
  qubit with angles `w·x + b`, then an entangler (default `CNOT(0→1)`) for two qubits. Qubit 0 is the most significant bit.
  Gradients are exact. The adjoint method walks the layers backwards, computing `∂L/∂a = Re⟨g|∂U/∂a|ψ⟩` and
  `g ← U†g`. An epoch of the largest configuration (2 qubits, 15 layers, 500 samples) takes about 10 ms.
- **Real-time loop.** The worker trains at the selected speed (1–60 epochs/s, or max). After each epoch it
  evaluates train and test sets and posts snapshots at most every ~33 ms, using transferable typed arrays.
  Heavy charts throttle their own redraws. Plotly's `react` diffs figures, and a constant `uirevision`
  keeps the camera you have rotated to.
- **Follow a sample, gate details.** Hovering a test point (Output, Results, Layer states) follows it; a click pins it.
  The main thread re-runs that one sample with the snapshot's θ (`src/viz/trajectory.ts`, every sub-rotation
  interpolated) to draw its path in the state plots, its angle on every dial and a crosshair on every unit map.
  Clicking a gate shows w, b, the full-batch gradient (sent with the snapshot at the unit-map cadence) and a
  circular histogram of its angle over the training data.
- **Output panel.** Classification shows the predicted class (faded by confidence) or, via the selector, one class
  score: the probability (2 qubits) or the fidelity with the class target (1 qubit). Regression adds the Fourier
  spectrum |c_k|, k = 0…10, of the model output and the target, a direct DFT on 512 points of t = πx
  (`src/viz/fourier.ts`).
- **Diagnostics.** Every epoch also records the mean gradient norm per layer over its batches (free: the
  gradients exist anyway; without regularization) and, for two qubits, the mean test-set tangle per layer. The
  Metrics panel shows both as epoch × layer heatmaps (histories over 300 epochs are averaged into bins). The
  circuit's *Gradient flow* toggle scales the taps with |∂L/∂w| and dims gates below 1% of the largest gradient.
- **Architecture options** (Settings → Architecture, structural, URL keys `ent`, `da`, `w`): the entangler
  (CNOT, CZ or none), which angles upload the data (all, or only RY's θ; φ and ω become pure biases) and the data
  weights (trainable, or fixed w = π for integer frequencies). Frozen weights are set in `QuantumModel` and get zero
  gradient, so Adam leaves them alone. The defaults are the original model, so old links and the parity fixtures
  stay valid.
- **Replay.** Parameters and Adam state are recorded per epoch, for the last 5000 epochs. Seeking shows a past
  epoch. *Resume from here* truncates the history and continues training from that point.
- **Sharing.** Settings are kept in the URL hash. *Share trained model* also encodes the displayed parameters
  (`p`, base64url float32) and the epoch (`ep`).

## Parity with the PyTorch reference

`legacy/tools/export_fixtures.py` runs the original `legacy/app/` modules on fixed inputs and parameters,
and records per-layer states, losses, gradients (with and without L1/L2), predictions, Adam steps, dataset
labels, targets and Fourier curves. Regenerate the fixtures after changing the Python reference:

```bash
pip install -r legacy/requirements.txt
python legacy/tools/export_fixtures.py
```

The tests compare at float32 precision, because the reference runs in complex64.

### Intentional differences (bug fixes)

| Reference behavior | Web app |
|---|---|
| A new Adam optimizer is created every epoch, which resets its moment estimates | Optimizer state persists; it is reset only when the model is reset |
| Regression noise uses unseeded `np.random`, so the data changes on every reload | Seeded and reproducible |
| A batch size larger than the training set (e.g. 128 on the ~80 regression samples) trains nothing | The batch is clamped to the data set size |
| The circuit diagram shows CZ symbols and no entangler after the last layer | Shows the CNOT that is actually applied after every layer |
| The live "predicted class" panel is colored by the true labels | Colored by the predictions |
| 15×15 decision grid with autoscaled colors | Configurable grid (default 40×40): predicted class faded by confidence, plus the decision boundary (argmax; p = 0.5 for two classes) |

Random draws (initialization, data points, shuffling) come from a seeded PRNG rather than torch's, so
individual runs differ numerically from the Dash app while following the same distributions.
