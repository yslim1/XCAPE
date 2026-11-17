# XCAPE Manual

**XCAPE** (**X**RD-**C**onstrained **A**ccelerated **P**rediction **E**ngine) is a crystal structure prediction (CSP) tool that searches for the atomic structure of a material by matching its predicted powder XRD (PXRD) pattern against a reference pattern, rather than searching purely on energy. It combines:

- **PyXtal** for symmetry-constrained random structure generation and evolutionary mutation,
- **SevenNet** (a machine-learning interatomic potential, MLIP) for fast structure relaxation and energy evaluation,
- **pymatgen** for XRD pattern simulation and structure comparison.

The core idea: given a known space group, composition, lattice, and an experimental (or reference) XRD pattern, generate many candidate atomic arrangements, relax them with an MLIP, score them by how well their simulated XRD pattern matches the reference, and evolve the population over generations (mutation + selection) until a good match (or energy minimum) is found.

---

## 1. Requirements & Installation

Python ≥ 3.12. Key dependencies (see `pyproject.toml`):

```
pymatgen>=2025.10.7
ase>=3.27.0
pyxtal==1.1.2
psutil>=7.2.2
pyyaml>=6.0.3
numpy>=2.4.2
scipy>=1.15
torch>=2.10.0
sevenn[torchsim]>=0.12.1
torch-sim-atomistic[symmetry]>=0.6.0
```

Install from the repo root (the directory containing `pyproject.toml`, one level above the `xcape` package):

```bash
pip install -e .
```

A CUDA-capable GPU is strongly recommended (SevenNet automatically uses `cuda` if `torch.cuda.is_available()`, otherwise falls back to CPU).

---

## 2. Quick Start

1. Obtain a reference XRD source — either an ICDD **PDF XML card** or a **CIF** file of a known/similar structure.
2. Copy `assets/config_example.yaml` (at the repo root) and edit `output_path` and `input_setting`.
3. Run:

```bash
xcape --config /path/to/your-config.yaml
# or equivalently
python -m xcape.main --config /path/to/your-config.yaml
```

(`--config` defaults to `<cwd>/xcape/config.yaml` if omitted.)

4. Monitor progress via the log files written into `output_path` (see [Section 7](#7-output-files)).

---

## 3. How the Algorithm Works

### 3.1 Setup phase (`main.py::main`)

1. **Load & validate config** — `load_config` reads the YAML; `validate_config` checks types/ranges/consistency (see [Section 6](#6-validation-rules)).
2. **Set up logging and runtime environment** (`runtime.setup_environment`):
   - Loads the SevenNet MLIP backend (`load_sevennet_calc`) — either a `SevenNetCalculator` (ASE-style, for `sequential` optimizer) or a `SevenNetModel` (batched, for `batch` optimizer).
   - Builds an `XRDCalculator` (pymatgen) using the configured wavelength.
   - Builds a `StructureMatcher` for deduplication.
   - Registers `random_generation_pyxtal` as the structure generator.
3. **Prepare the reference pattern** (`reference.prepare_reference`):
   - Parses the input XML (ICDD PDF card) or CIF into a `ReferenceInfo`: chemical formula, lattice parameters, space group, Z (formula units per cell), and the reference 2θ/intensity arrays.
   - Filters reference peaks to the configured `two_theta_range` if given.
   - Writes derived crystallographic data (lattice, space group, composition) back into `config` for downstream use.
4. **Determine distance constraints** (`reference.ensure_distance_constraint` → `modules/distance_constraint.py`):
   - If `distance_constraint` isn't already given in config, and `calc_dist_tol: True`, XCAPE runs a **melt–quench MD (MQA)** simulation with the MLIP to empirically estimate the minimum physically reasonable interatomic distance for every element pair (see [Section 3.5](#35-distance-constraints-mqa)).
   - If `calc_dist_tol: False`, a heuristic covalent-radius-based tolerance is used instead.
   - The resulting per-pair distances become a PyXtal `Tol_matrix`, used to reject physically implausible structures during generation, mutation, and relaxation.
5. Logging files are initialized, and the whole merged config is dumped to `total.yaml` in the output directory for reproducibility.

### 3.2 Generation loop (`main.py::_run_generation`, repeated for `structure.generation` generations)

Each generation:

1. **Build population** (`_build_generation_population`):
   - `random_ratio × population` structures are generated **from scratch** via PyXtal (`random_generation_pyxtal`), honoring the target space group, lattice, composition, and distance tolerance matrix.
   - The remaining `(1 - random_ratio) × population` structures are produced by **mutating** existing candidates (`perform_mutation_and_append`), unless this is generation 1 (no candidates yet), in which case the whole population is random.
2. **Relax** the full population with the MLIP (`_run_relaxation` → `serial_relax` or `batch_relax`, depending on `optimizer.type`).
2b. **Non-symmetry relaxation gate** (`_run_no_symmetry_relaxation`, optional — only when `structure.add_no_symmetry_relax: True`):
   - Rank the **converged** trials by energy and take the best `add_no_symmetry_relax_ratio × population` of them (rounded down, minimum 1). Non-converged trials are excluded, since their failure penalty is not applied until postprocessing.
   - Re-relax each one **from its originally generated structure** with `optimizer.fix_symmetry` forced to `False`.
   - These results are **reported only** — see §7. They never enter the candidate pool, so the structures carried into the next generation stay symmetry-constrained and remain valid inputs for the Wyckoff-based mutation operators. They *are* still tested for early stopping (step 4).
3. **Postprocess** (`postprocess_generation`):
   - Compute XRD **R-factor** and **Wasserstein distance** for every relaxed structure against the reference pattern.
   - Select candidates within threshold of the best energy/R-factor/Wasserstein (`_select_candidates`).
   - **Deduplicate** structurally- and metrically-similar candidates (`deduplicate_candidates`), then cap the pool size (`dedup.keep_max`) using a metric-mixed truncation strategy.
   - Write `POSCARs`/`CONTCARs` and append a row to `generation.log` for every trial.
   - Score the gate results from step 2b the same way and write them to `CONTCARs_no_sym` / `non-sym.log` (`process_no_symmetry_results`).
4. **Early stopping**: if any surviving candidate's R-factor is below `structure.early_stopping_threshold`, the loop stops early. Structures from the step-2b gate are checked too — they do not breed, but an R-factor below the threshold means the pattern has been matched regardless of whether symmetry was held fixed. When a gate structure is what triggers the stop, the `xcape.log` message says so explicitly, because that structure is in `CONTCARs_no_sym` rather than the candidate pool.

### 3.3 Structure generation (`modules/generation.py`)

`random_generation_pyxtal` builds a PyXtal `Lattice` from the reference lattice parameters (auto-detecting lattice type: cubic, tetragonal, hexagonal, trigonal, orthorhombic, monoclinic, triclinic) and repeatedly calls `pyxtal.from_random(...)` with the target space group, composition, and tolerance matrix until a valid structure is produced (up to 10 attempts per structure).

### 3.4 Mutation (`modules/mutation.py`, orchestrated by `modules/generation.py::perform_mutation_and_append`)

For each mutant to create:

1. A **parent** is chosen from the current candidate pool via Boltzmann-weighted sampling on a configurable metric (`energy`, `r_factor`, or `wasserstein`), with probability `weighted_parent_ratio` (otherwise a uniformly random parent is chosen). The Boltzmann **temperature** decays each generation (`temperature * temperature_decay^(gen-1)`, floored at `min_temperature`) — early generations explore more broadly, later generations exploit the best candidates more.
2. The parent structure is converted to a PyXtal object and re-symmetrized/converted to the initial target space group if it drifted.
3. One mutation **move** is sampled according to `mutation_ratios` and applied (retried up to `max_mutation_attempts` times if it fails); if all attempts fail, the parent is kept unchanged (`origin_type = max_attempts_reached`).

The available mutation moves (all composition-preserving, all validated against the distance tolerance matrix):

| Move | Ratio key | Description |
|---|---|---|
| Swap same multiplicity | `swap_same_mult_ratio` | Move one Wyckoff site to a different position of the same multiplicity. |
| Split site | `split_site_ratio` | Replace one site with two sites of the same species whose multiplicities sum to the original. |
| Merge sites | `merge_site_ratio` | Replace two same-species sites with one site of combined multiplicity. |
| Swap different elements | `swap_diff_elem_ratio` | Swap the species of two sites with equal multiplicity. |
| Site reconfiguration | `site_reconfig_ratio` | Reassign *all* Wyckoff sites of one randomly chosen species. |
| Resample position | `resample_pos_ratio` | Randomize the fractional coordinate of one movable (dof > 0) site. |
| Multi resample position | `multi_resample_pos_ratio` | Randomize coordinates of several movable sites simultaneously. |

All seven ratios must sum to 1.0.

### 3.5 Distance constraints (MQA)

When `calc_dist_tol: True` (default) and no explicit `distance_constraint` is given, `modules/distance_constraint.py::run_mqa` determines physically sensible minimum interatomic distances by:

1. Building a ~300-atom random structure with the target composition.
2. Melting it via Langevin MD starting at 4200 K, cooling in 800 K steps down to find the melting point (via mean-squared-displacement/diffusion coefficient threshold).
3. Running a production melt at the estimated melting temperature.
4. Scanning the melt trajectory for the shortest observed distance between every element pair (`check_shortest_distance`), clamped to `dist_tol_limit` and scaled by `dist_tol_scale`.

Results are cached to `<output_path>/mlip_mqa/distance_constraint.pkl` and reused on subsequent runs. Set `calc_dist_tol: False` to skip MQA and instead use covalent-radius-based heuristic tolerances (`Tol_matrix(prototype='atomic', factor=1.5)`), or provide your own `distance_constraint` map to bypass MQA entirely.

### 3.6 Relaxation (`modules/relaxation.py`)

Two backends, selected by `optimizer.type`:

- **`sequential`** (`serial_relax`): relaxes structures one at a time using ASE's `FIRE` optimizer with a `SevenNetCalculator`.
- **`batch`** (`batch_relax`): relaxes the whole population simultaneously on GPU using [torch-sim](https://github.com/Radical-AI/torch-sim)'s batched FIRE optimizer with a `SevenNetModel`. Much faster for large populations. Falls back to per-structure symmetry handling if batched `FixSymmetry` initialization fails for some structures.

Both use a **two-stage** relaxation:

1. **Short stage** (`max_short_step_per_atom` steps/atom): a cheap initial optimization used purely to screen out structures that immediately collapse into unphysically short interatomic distances (`status = distance_violation_initial`).
2. **Full stage** (`max_step_per_atom` steps/atom): full relaxation to force tolerance `ftol`. Final status is `converged`, `not_converged`, or `distance_violation_final` depending on convergence and a final distance check.

If `optimizer.fix_symmetry: True`, space-group symmetry is preserved during relaxation (ASE `FixSymmetry` / torch-sim `FixSymmetry`).

### 3.7 Scoring and selection (`modules/postprocess.py`)

For every relaxed structure:

- If relaxation did **not** converge cleanly, its energy/R-factor/Wasserstein are all penalized to `99.9`.
- Otherwise, a PXRD pattern is simulated (pymatgen `XRDCalculator`) and compared to the reference via:
  - **R-factor** (`utils/xrd_utils.py::r_factor_peak_pairing`): peaks within `two_theta_tol` of each other are clustered/paired, intensities are optionally normalized (`XRD_setting.normalize`), and `R = Σ(I_calc - I_obs)² / Σ(I_obs)²` is computed over the paired clusters.
  - **Wasserstein distance** (`wasserstein_dist`): both patterns are rendered as Gaussian-broadened curves on a shared grid and compared via 1D Wasserstein (earth-mover) distance — robust to slight peak-position shifts.

**Selection**: a structure survives into the next generation's candidate pool if it is within `delta` of the *best* value of **any enabled** metric (`energy`, `r_factor`, `wasserstein` — each independently toggleable via `structure.best_metric.<metric>.enabled`/`delta`).

**Deduplication**: among survivors, structures whose energy/R-factor/Wasserstein are all within `dedup.<metric>` tolerance of an existing kept structure **and** which structurally match (pymatgen `StructureMatcher`, tolerances under `dedup.structure_matcher`) are treated as duplicates and dropped, keeping the earlier (better-sorted) one.

**Pool capping**: if the deduplicated pool exceeds `dedup.keep_max`, it is trimmed by taking the best fraction of `keep_max` sorted by R-factor, then energy, then Wasserstein, according to `dedup.mix_ratios` (so the surviving pool retains diversity across all three criteria rather than collapsing onto a single metric).

Every candidate that survives becomes part of next generation's population automatically (as an unmutated "kept_best" carry-over) in addition to feeding the mutation parent pool.

---

## 4. Configuration Reference

Configuration is a single YAML file (see `assets/config_example.yaml` for a fully annotated example). All keys below except where marked "required" have defaults.

### `output_path` (required)
Directory where all outputs/logs are written. Created if missing.

### `seed`
Optional non-negative integer. When set, XCAPE seeds the global RNGs (Python `random`, NumPy, and Torch) once at startup — before generation, mutation, and the MQA distance-constraint MD — so a run is repeatable. Omit (or set `null`) to use fresh entropy (non-reproducible). Note: this makes the structure-search (CPU) portion deterministic, but GPU MLIP relaxation (SevenNet / torch-sim) may still be non-deterministic due to CUDA kernels, so it is a strong repeatability aid rather than a hard determinism guarantee.

### `input_setting`
| Key | Description |
|---|---|
| `xml_path` | Path to an ICDD PDF XML reference card. Exactly one of `xml_path`/`cif_path` must be set. |
| `cif_path` | Path to a reference CIF file. |

Either may point to a directory instead of a file, **only if** that directory contains exactly one matching file (`*.xml` or `*.cif`) — a direct file path is recommended for reproducibility.

### Distance-constraint settings (top-level)
| Key | Default | Description |
|---|---|---|
| `calc_dist_tol` | `True` | If `True`, run MQA MD to derive distance tolerances; if `False`, use covalent-radius heuristics. |
| `dist_tol_scale` | *(none — no scaling)* | Multiplicative scale applied to derived/heuristic tolerances. The example config uses `0.75`. |
| `dist_tol_limit` | `0` | Minimum allowed tolerance (Å) — clamps overly short MQA-derived values. The example config uses `1.2`. |
| `distance_constraint` | *(none)* | Optional explicit map, e.g. `{Li-O: 1.80, Li-Li: 2.20}`. If set, MQA is skipped entirely. |

### `MLIP_setting`
Currently SevenNet only.
| Key | Description |
|---|---|
| `model` | SevenNet model name, e.g. `7net-omni`. |
| `modal` | Optional modal/channel, e.g. `mpa`. |
| `backend` | `flash`, `cueq`, or unset — selects `enable_flash`/`enable_cueq` acceleration flags. |

### `XRD_setting`
| Key | Default | Description |
|---|---|---|
| `wave_length` | — (required) | One of the wavelength keys in `utils/xrd_utils.py::WAVELENGTHS` (e.g. `CuKa`, `MoKa`, `AgKa1`, ...). |
| `two_theta_tol` | `0.0` | Peak-pairing tolerance in degrees for R-factor clustering; must be ≥ 0. |
| `two_theta_range` | *(reference peak range)* | Optional `[min, max]` comparison window; if omitted, uses the min/max of the reference peaks, expanded by `two_theta_tol`. |
| `normalize` | `null` | Intensity normalization before R-factor comparison: `null`, `max`, `sum`, or `scale`. `scale` performs a least-squares scale fit of the calculated pattern to the observed one (Mint's `DR_SQUARED`), leaving the observed intensities unscaled. |

### `optimizer`
| Key | Default | Description |
|---|---|---|
| `type` | — (required) | `sequential` (ASE FIRE, one at a time) or `batch` (torch-sim FIRE, GPU-batched). |
| `ftol` | `0.05` (module default; example configs use `0.02`) | Force convergence tolerance (eV/Å). |
| `max_short_step_per_atom` | `1` (module default; example configs use `5`) | Steps/atom for the cheap initial distance-violation screening stage. |
| `max_step_per_atom` | `100` (module default; example configs use `20`) | Steps/atom for the full relaxation stage. |
| `fix_symmetry` | — (required, bool) | Preserve space-group symmetry during relaxation. |
| `max_memory_padding` | `0.8` | (`batch` only) Fraction of the probed GPU memory limit the torch-sim autobatcher may use. Lower = more headroom; raising it increases OOM risk. |

### `structure`
| Key | Default | Description |
|---|---|---|
| `generation` | — (required) | Number of generations to run; positive integer. |
| `population` | — (required) | Structures per generation; positive integer. |
| `random_ratio` | — (required) | Fraction of each generation's population generated randomly (rest via mutation); range `[0,1]`. |
| `early_stopping_threshold` | `0.005` | Stop early if any candidate's R-factor drops below this. |
| `add_no_symmetry_relax` | `False` | Enable the reporting-only non-symmetry relaxation gate (§3.2 step 2b). Adds `CONTCARs_no_sym` and `non-sym.log`; does not change the search itself. |
| `add_no_symmetry_relax_ratio` | `0.3` | Fraction of the population re-relaxed by that gate, rounded down but never below 1; range `[0,1]`. Ignored when the gate is off. |

#### `structure.mutation`
| Key | Default | Description |
|---|---|---|
| `selection_method.parent_metric` | `energy` | Metric used for Boltzmann parent weighting: `energy`, `r_factor`, or `wasserstein`. |
| `selection_method.weighted_parent_ratio` | `0.8` | Probability of weighted (vs. uniform random) parent sampling. |
| `selection_method.temperature` | `1.0` | Initial Boltzmann temperature. |
| `selection_method.temperature_decay` | `0.9` | Per-generation multiplicative decay of temperature. |
| `selection_method.min_temperature` | `0.5` | Floor for temperature decay. |
| `max_mutation_attempts` | `100` | Retries across whole mutation calls per offspring. |
| `mutation_ratios.*` | see table in §3.4 | Seven ratios; **must sum to 1.0** if provided (all seven keys required together). |
| `mutation_ratios.max_attempts` | `100` | Retries inside a single mutation move. |

#### `structure.best_metric`
Controls generation-to-generation selection. Each of `energy`, `r_factor`, `wasserstein` takes:
```yaml
energy:
  enabled: True
  delta: 0.1     # eV/atom for energy; unitless for r_factor/wasserstein(degrees)
```
Defaults: energy `enabled=True, delta=0.1`; r_factor `enabled=True, delta=0.05`; wasserstein `enabled=True, delta=0.5`.

#### `structure.dedup`
| Key | Default | Description |
|---|---|---|
| `keep_max` | *(none — no capping)* | Max candidates kept after dedup; must be > 0 if set. The example config uses `25`. |
| `energy` | `0.005` | eV/atom tolerance for duplicate detection. |
| `r_factor` | `0.01` | R-factor tolerance for duplicate detection. |
| `wasserstein` (or legacy `wd`) | `0.01` | Wasserstein tolerance for duplicate detection. Only one of `wasserstein`/`wd` may be set. |
| `mix_ratios.r_factor` / `.energy` / `.wasserstein` | `0.4` / `0.4` / `0.2` | Fractions of `keep_max` filled by each metric when trimming an oversized pool. If provided, all three keys are required and must sum to 1.0; use explicit zeros to lean on a single metric. Rounding remainder goes to the largest bucket (ties: `r_factor` > `energy` > `wasserstein`). |
| `structure_matcher.ltol` / `.stol` / `.angle_tol` / `.primitive_cell` | `0.2` / `0.3` / `5` / `False` | Pymatgen `StructureMatcher` tolerances for structural duplicate detection. |

---

## 5. Command-Line Usage

```bash
python -m xcape.main --config <path/to/config.yaml>
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `<cwd>/xcape/config.yaml` | Path to the YAML config file. |

The process exits with code `0` on completion (either all generations ran, or early stopping triggered).

---

## 6. Validation Rules

`validation.py::validate_config` runs before any computation and raises `ValueError` with a descriptive message if any of the following are violated:

- `output_path` must be set.
- `seed`, if set, must be a non-negative integer (booleans are rejected).
- Exactly one of `input_setting.xml_path` / `input_setting.cif_path` must be set (as a string).
- `structure.generation` and `structure.population` must be positive integers.
- `structure.random_ratio` must be defined and in `[0, 1]`.
- `structure.mutation.selection_method.weighted_parent_ratio` must be in `[0, 1]`.
- If `structure.mutation.mutation_ratios` is set, all 7 keys must be present and sum to 1.0.
- `structure.early_stopping_threshold` must be ≥ 0.
- `structure.add_no_symmetry_relax` must be a boolean.
- `structure.add_no_symmetry_relax_ratio` must be a number in `[0, 1]`.
- `optimizer.type` must be `sequential` or `batch`.
- `optimizer.fix_symmetry` must be a boolean.
- `XRD_setting.wave_length` must be a recognized wavelength key.
- `XRD_setting.two_theta_tol` must be ≥ 0.
- `XRD_setting.two_theta_range`, if set, must be a 2-element list/tuple with `min < max`.
- `XRD_setting.normalize` must be `null`, `max`, `sum`, or `scale`.
- `structure.dedup.keep_max`, if set, must be a positive integer.
- `structure.dedup` may set only one of `wasserstein` or legacy `wd`, not both.
- `structure.dedup.mix_ratios`, if set, must be a mapping defining all three keys (`r_factor`, `energy`, `wasserstein`) with numeric, non-negative values summing to 1.0. Partial mappings are rejected: the missing keys would otherwise fall back to the `DEFAULT_KEEP_MAX_*` ratios and push the total above 1.0. Omit `mix_ratios` entirely (or set it to `null`/`{}`) to use the defaults.

`validation.py::validate_reference_info` additionally checks (after parsing the XML/CIF) that the parsed reference contains `chemical_formula`, `lattice_parameters`, `z`, `spg_num`, `spg_symbol`, `two_theta`, and `intensity`.

---

## 7. Output Files

All written under `output_path`:

| File | Contents |
|---|---|
| `total.yaml` | Full merged configuration (including auto-derived lattice/space-group/distance-constraint info) dumped at run start, for reproducibility. |
| `xcape.log` | General runtime log (INFO level): setup steps, warnings, MQA progress, early-stopping notice, etc. |
| `generation.log` | Tabular per-trial log: `Gen Pop Type Parent E RF WD Gen_t Rel_t Status` for every structure evaluated each generation (new trials + carried-over candidates). |
| `memory.log` | RSS memory usage (MB) appended after each generation, for monitoring resource usage over long runs. |
| `POSCARs` | Appended POSCAR-format text for every *pre-relaxation* trial structure, in generation order. |
| `CONTCARs` | Appended POSCAR-format text for every *post-relaxation* (relaxed) structure, in the same order as `POSCARs`. |
| `non-sym.log` | Same columns as `generation.log`, for the unconstrained re-relaxations. Only created when `structure.add_no_symmetry_relax: True`. `Pop` is the population index of the symmetry-constrained trial the structure came from, and `Parent` is that trial as `"gen_pop"` — so each row pairs directly with the matching `generation.log` row. |
| `CONTCARs_no_sym` | Appended POSCAR-format text for every unconstrained re-relaxed structure, in the same order as `non-sym.log`. Only created when `structure.add_no_symmetry_relax: True`. |
| `mlip_mqa/` | MQA MD intermediate/trajectory files (`POSCAR_rand`, `premelt.*`, `Tm_*.traj/log`, `melt.*`, `distance_constraint.pkl`) — only created when `calc_dist_tol: True` and no explicit `distance_constraint` was given. Cached across reruns. |

### `generation.log` columns
| Column | Meaning |
|---|---|
| `Gen` | Generation index. |
| `Pop` | Population index within the generation. |
| `Type` | Origin type: `random`, `kept_best`, `swap_same_mult`, `split_site`, `merge_site`, `swap_diff_elem`, `site_reconfig`, `resample_pos`, `multi_resample_pos`, `spg_mismatch`, `max_attempts_reached`. |
| `Parent` | Parent identifier as `"gen_pop"`, or `none`. |
| `E` | Energy per atom (eV/atom); `99.9` indicates a non-converged/rejected trial. |
| `RF` | R-factor vs. reference pattern. |
| `WD` | Wasserstein distance vs. reference pattern. |
| `Gen_t` | Structure generation/mutation time (s). |
| `Rel_t` | Relaxation time (s). |
| `Status` | `converged`, `not_converged`, `distance_violation_initial`, or `distance_violation_final`. |

---

## 8. Module Map

(`assets/config_example.yaml` — the annotated example config — lives at the repo root, one level above the `xcape/` package.)

```
xcape/
├── main.py                    Entry point: CLI parsing, main loop, early stopping
├── runtime.py                 Environment/logging setup
├── reference.py               Reference pattern loading & distance-constraint bootstrap
├── validation.py              Config & reference validation
├── modules/
│   ├── extraction.py          Parse CIF/XML into ReferenceInfo (composition, lattice, spg, 2θ/I)
│   ├── generation.py          PyXtal random structure generation; mutation orchestration
│   ├── mutation.py            Mutator class: the 7 Wyckoff-site mutation operators
│   ├── distance_constraint.py Melt-quench-anneal (MQA) MD for empirical distance tolerances
│   ├── relaxation.py          ASE (sequential) / torch-sim (batch) FIRE relaxation
│   └── postprocess.py         XRD metrics, selection, deduplication, output writing
├── mlips/
│   └── sevennet.py             SevenNet calculator/model factory (ASE vs. torch-sim backends)
└── utils/
    ├── types.py                Environment, ReferenceInfo, RelaxResult dataclasses/TypedDicts
    ├── core_utils.py           Lattice/volume/Z helpers, YAML I/O, POSCAR formatting
    ├── xrd_utils.py            R-factor, Wasserstein distance, Gaussian XRD pattern helpers
    └── torchsim_utils.py       torch-sim autobatcher calibration & OOM-fallback helpers
```

---

## 9. Practical Notes & Tips

- **`batch` vs `sequential` optimizer**: `batch` is far faster on GPU for realistic population sizes (uses torch-sim's autobatching FIRE), but `sequential` (ASE) can be easier to debug for a single structure or when running on CPU-only machines.
- **`fix_symmetry: True`** is recommended when you trust the target space group — it keeps relaxed structures from drifting to a lower-symmetry local minimum, which also keeps mutation's Wyckoff-site logic well-defined.
- **MQA cost**: the melt-quench distance-constraint calculation runs a sizable MD simulation once and caches the result (`mlip_mqa/distance_constraint.pkl`); delete that file (or point to a fresh `output_path`) to force recomputation, e.g. after changing composition.
- **`temperature` tuning**: the recommended setting is `temperature: 1.0` decaying to `min_temperature: 0.5`; scale both to the energy scale of the MLIP you're using.
- **`two_theta_range`**: omit it to let XCAPE use exactly the reference pattern's peak range (expanded by `two_theta_tol`); set it explicitly to restrict comparison to a specific window (e.g. excluding low-angle noise or beyond a detector's practical range).
- **Reproducibility**: prefer direct file paths (not directories) for `xml_path`/`cif_path`, and keep the auto-written `total.yaml` alongside your results — it captures the fully resolved configuration, including auto-derived lattice/space-group/composition and any MQA-derived distance constraints.
