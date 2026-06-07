# Week 2 Summary — Dataset Generation (Status & Next Steps)

This file records what was completed through Phase 2 and provides a detailed, actionable plan for Phase 3 (synthetic dataset) and Phase 4–5 (literature dataset, PINN, validation). It also contains pragmatic suggestions to ensure reproducibility and fast iteration.

## Status (Completed)
- **PHASE 1: GitHub repository setup** — Completed
  - Repository structure created and committed.
  - Collaborator workflow and branch rules documented in `IMPLEMENTATION_PLAN.md`.
- **PHASE 2: Environment setup** — Completed locally
  - Virtual environment created and packages installed during testing.
  - `requirements.txt` placeholder exists; finalize after adding runtime deps.

Notes: the basic project skeleton (folders and `docs/` files) is in place under the workspace root.

---

## What remains from Week 2 (Phase 3 onward)
Phase 3 is the core Week 2 deliverable. Phase 4 and Phase 5 are partner and validation tasks respectively.

### PHASE 3 — Synthetic Dataset Generation (Detailed Implementation)
Goal: produce a reproducible, versioned synthetic dataset of THz curves and associated ground-truth parameters for Week 3 model training.

1. Time axis
   - Use: `t = np.linspace(0, 10, 200)` (0 → 10 ps, 200 samples).
   - Save `t.npy` in `data/synthetic/` so preprocessing uses the same grid.

2. Physics model (vectorized, NumPy + PyTorch variant)
   - Implement `src/signal_generator.py` with functions:
     - `generate_signal(t, tau_intra, tau_inter, k_transfer, A1, A2, f=None)` → returns signal (np.ndarray).
     - `sample_parameters(n, seed=None)` → returns DataFrame or dict of sampled params.
     - `generate_dataset(n_samples, out_dir, batch_size=1000, t=None, seed=42)` → writes `signals.npy` and `labels.csv` in streaming/chunked manner.
   - Choose `f(t)` as a small transient (e.g., `f(t) = t * np.exp(-t / (tau_inter + 1e-8))`) to reflect charge-transfer dynamics.

3. Parameter sampling
   - Ranges (recommended):
     - `tau_intra`: 0.1 → 5 (ps) — consider log-uniform for low values
     - `tau_inter`: 1 → 20 (ps)
     - `k_transfer`: 0 → 1
     - `A1`, `A2`: 0.5 → 2
     - `noise_std_rel`: 0 → 0.1 (fraction of signal std)
   - Save sampling policy in `data/synthetic/manifest.json` (ranges, seed, date, generator version).

4. Noise model
   - Implement `src/noise.py` with `add_gaussian_noise(signal, noise_std_rel, seed=None)`.
   - Use relative noise (std = noise_std_rel * signal.std()) to preserve SNR scaling.

5. Storage and format
   - `signals.npy`: shape `(N, 200)`, dtype `float32`.
   - `labels.csv`: columns `sample_id,tau_intra,tau_inter,k_transfer,A1,A2,noise_std_rel,structure_type,source`.
   - `manifest.json`: generator config and RNG seed.
   - For large N (>100k), use `h5py` or `np.memmap` to avoid memory spikes.

6. Generation strategy
   - First produce a **debug set**: N=1,000 to validate pipeline and plotting.
   - Then produce full set: N=20,000–50,000 (or as decided).
   - Write in chunks (e.g., 1k samples per iteration) to avoid peak memory usage.

7. Visual verification
   - Create scripts to plot:
     - Varying `tau_intra` (fix other params)
     - Varying `tau_inter`
     - Varying `k_transfer`
   - Save images to `results/figures/` with descriptive filenames (e.g., `vary_tau_intra.png`).

8. Reproducibility
   - Record RNG seed and manifest.
   - Commit `scripts/generate_synthetic.py` and small debug outputs for CI.

Quick commands to run generator (example)

```powershell
# activate venv
venv\Scripts\activate
# install minimal libs if needed
pip install numpy pandas matplotlib h5py
# generate a small debug set
python scripts/generate_synthetic.py --n 1000 --out data/synthetic --seed 42
```

---

### PHASE 4 — Literature Dataset (Partner)
Goal: collect, digitize, normalize, and annotate 5–20 real THz curves for validation.

1. Paper collection
   - Partner collects 5–20 high-quality THz/TDS papers and stores PDFs/screenshots under `docs/literature/` or `data/literature/raw/`.

2. Digitization process
   - Use WebPlotDigitizer (https://automeris.io/WebPlotDigitizer).
   - Standardize axes: resample to `t = np.linspace(0, 10, 200)`.

3. Storage
   - For each paper create `data/literature/<paper_id>/curve.csv` and `metadata.json` (paper, year, material, figure, axis range, notes).

4. Normalization
   - Apply consistent normalization policy (e.g., min→0, max→1) and record it in metadata.

5. Deliverable
   - `data/literature/` organized with `curve.csv` + `metadata.json` for each paper.

Suggestion: Provide a short how-to `docs/literature_digitization.md` for your partner outlining the exact calibration steps.

---

### PHASE 5 — Validation Strategy (PINN + Uncertainty)
Goal: Use literature curves for validation and implement physics-informed losses and uncertainty estimation on synthetic training.

1. PINN training (on synthetic data)
   - Implement `src/physics/thz_forward_model.py` (PyTorch) and `src/models/pinn.py`.
   - Train with hybrid loss: `L_total = L_data + lambda_recon * L_recon + lambda_phys * L_phys`.
   - Use validation holdouts to tune `lambda` values.

2. Uncertainty estimation
   - Monte Carlo Dropout: keep dropout active at inference and run T forward passes.
   - Ensembles: train K small models with different seeds and aggregate predictions.
   - Report mean ± std and 95% prediction intervals.

3. Literature validation
   - Preprocess literature curves to same grid and normalization used in training.
   - Run inference and save outputs with uncertainty to `results/predictions/` and a short report in `results/metrics/`.

4. Generalization experiments
   - Leave-one-structure-out: tag `structure_type` in synthetic labels, run LOO experiments.

---

## Practical Suggestions & Notes
- Use short venv paths (e.g., `C:\venv\thz`) to avoid long-path issues on Windows.
- Keep the initial physics model simple (sum of exponentials + small f(t)) to ensure gradient stability.
- Save small debug artifacts (first 100 samples) in Git for quick reproducibility and CI tests.
- Consider adding a small `requirements.txt` with exact versions for reproducibility. Add `torch` only when starting model development.
- Add `data/synthetic/manifest.json` to record exact parameter ranges and RNG seed — do not commit large binaries like `signals.npy` to Git; use Git LFS if necessary.
- Create basic unit tests in `tests/` for `generate_signal()` and `add_gaussian_noise()` to catch early regressions.

---

## Deliverables Checklist (Week 2)
- [x] Repo and structure
- [x] Environment setup
- [ ] Small debug synthetic dataset (1k) — recommended immediate next step
- [ ] Full synthetic dataset (20k–50k)
- [ ] Plots demonstrating parameter effects (3 plots)
- [ ] Literature curves (partner)
- [ ] Manifest and metadata files

If you want, I can now:
- Implement the starter scripts (`src/signal_generator.py`, `src/noise.py`, `scripts/generate_synthetic.py`) and a small unit test, or
- Add a pinned `requirements.txt` containing the minimal packages and version pins.

Which should I do next? ("implement scripts" or "write requirements")
