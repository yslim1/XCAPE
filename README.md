# XCAPE

**XRD-Constrained Accelerated Prediction Engine**

XCAPE performs crystal structure prediction (CSP) constrained by experimental
powder X-ray diffraction (PXRD) data. Instead of searching purely for the
lowest-energy structure, XCAPE searches for the structure whose *simulated*
PXRD pattern best matches a reference pattern, using a machine-learned
interatomic potential (MLIP) for fast relaxation and energy evaluation.

It combines three established tools:

- **[PyXtal](https://github.com/MaterSim/PyXtal)** — symmetry-constrained random structure generation and Wyckoff-site mutation.
- **[SevenNet](https://github.com/MDIL-SNU/SevenNet)** — an MLIP for fast structure relaxation and energy scoring (ASE and GPU-batched [torch-sim](https://github.com/Radical-AI/torch-sim) backends).
- **[pymatgen](https://pymatgen.org/)** — PXRD pattern simulation and structural deduplication.

Given a known space group, composition, lattice, and a reference PXRD pattern,
XCAPE generates many candidate structures, relaxes them, scores each by how
well its simulated pattern matches the reference, and evolves the population
over generations (mutation + selection) until a good match is found.

---

## How it works

```
config.yaml
   │
   ▼
[validate config] ─▶ [set up MLIP + XRDCalculator + StructureMatcher]
   │
   ▼
[parse reference XML/CIF] ─▶ formula, lattice, space group, Z, 2θ/intensity peaks
   │
   ▼
[distance constraints]  ── melt-quench MD (MQA) or covalent-radius heuristic
   │                        → per-element-pair minimum distances (cached)
   ▼
┌──────────────── generation loop (× N generations) ────────────────┐
│  1. Build population:  random_ratio via PyXtal + rest by mutation  │
│  2. Relax:             sequential (ASE FIRE) or batch (torch-sim)   │
│  3. Score:             R-factor + Wasserstein + energy vs. reference│
│  4. Select + dedup:    keep within δ of best; StructureMatcher     │
│  5. Early stop if best R-factor < threshold                        │
└────────────────────────────────────────────────────────────────────┘
```

Candidates are scored on three complementary metrics — an XRD **R-factor**
(peak-pairing), a **Wasserstein distance** between Gaussian-broadened patterns
(robust to small peak shifts), and MLIP **energy** — and a structure survives a
generation if it is within a tolerance `δ` of the best value on *any* enabled
metric.

See [`docs/MANUAL.md`](docs/MANUAL.md) for the full algorithm description,
configuration reference, and validation rules.

---

## Requirements

- Python ≥ 3.12
- A CUDA-capable GPU is strongly recommended. SevenNet uses `cuda` when
  `torch.cuda.is_available()`, otherwise falls back to CPU. The `batch`
  optimizer in particular is designed for GPU throughput.

Key dependencies (resolved by `pip`, see `pyproject.toml`):

```
pymatgen>=2025.10.7   ase>=3.27.0        pyxtal==1.1.2
numpy>=2.4.2          scipy>=1.15        torch>=2.10.0
sevenn[torchsim]>=0.12.1               pyyaml>=6.0.3   psutil>=7.2.2
torch-sim-atomistic[symmetry]>=0.6.0
```

## Installation

Editable install from this directory (the one containing `pyproject.toml`):

```bash
pip install -e .
```

With test dependencies:

```bash
pip install -e ".[test]"
```

This installs the `xcape` console-script entry point.

---

## Quick start

1. Obtain a reference PXRD source — either an ICDD **PDF XML card** or a **CIF**
   file of a known or similar structure.
2. Copy [`assets/config_example.yaml`](assets/config_example.yaml) and edit at
   least `output_path` and `input_setting` (set exactly one of `xml_path` /
   `cif_path`).
3. Run:

```bash
xcape --config /path/to/your-config.yaml
```

or equivalently:

```bash
python -m xcape.main --config /path/to/your-config.yaml
```

`--config` defaults to `<cwd>/xcape/config.yaml` if omitted. The process exits
with code `0` on completion (all generations finished, or early stopping
triggered).

4. Monitor progress via the log files written into `output_path` (see
   [Output files](#output-files)).

---

## Configuration

Configuration is a single YAML file. The essentials:

```yaml
output_path: /path/to/output/        # required; created if missing
# seed: 0                            # optional; seeds RNGs for repeatability

input_setting:
  xml_path: /path/to/card.xml        # exactly one of xml_path / cif_path
  # cif_path: /path/to/ref.cif

MLIP_setting:                        # currently SevenNet only
  model: 7net-omni
  modal: mpa
  backend: flash                     # flash | cueq | unset

XRD_setting:
  wave_length: CuKa                  # required; a key in utils/xrd_utils.WAVELENGTHS
  two_theta_tol: 0.3                 # peak-pairing tolerance (deg), >= 0; default 0
  two_theta_range: [1.0, 90.0]       # optional comparison window
  normalize: sum                     # null | max | sum | scale

optimizer:
  type: batch                        # batch (torch-sim, GPU) | sequential (ASE)
  fix_symmetry: True                 # required bool

structure:
  generation: 50                     # number of generations
  population: 50                     # structures per generation
  random_ratio: 0.1                  # fraction generated from scratch each gen
  early_stopping_threshold: 0.002    # stop if any candidate R-factor < this
```

Additional sections control mutation (`structure.mutation`), candidate
selection (`structure.best_metric`), deduplication and pool capping
(`structure.dedup`), and distance constraints (`calc_dist_tol`,
`dist_tol_scale`, `dist_tol_limit`, or an explicit `distance_constraint` map).
See [`assets/config_example.yaml`](assets/config_example.yaml) for a fully
annotated example and [`docs/MANUAL.md`](docs/MANUAL.md) §4 for every key,
default, and validation rule.

---

## Output files

All written under `output_path`:

| File | Contents |
|---|---|
| `total.yaml` | Full merged config (including auto-derived lattice/space-group/distance data) dumped at run start, for reproducibility. |
| `xcape.log` | General runtime log: setup steps, warnings, MQA progress, early-stopping notice. |
| `generation.log` | Per-trial table: `Gen Pop Type Parent E RF WD Gen_t Rel_t Status` for every structure evaluated. |
| `memory.log` | RSS memory (MB) appended after each generation. |
| `POSCARs` / `CONTCARs` | Pre-relaxation and post-relaxation structures, in generation order. |
| `mlip_mqa/` | Melt-quench MD intermediates and the cached `distance_constraint.pkl` (only when `calc_dist_tol: True` and no explicit `distance_constraint`). Reused across reruns. |

---

## Testing

```bash
pytest
```

The suite (`tests/`) contains roughly one test module per source module.

---

## Project layout

```
xcape/
├── main.py                     Entry point: CLI, main loop, early stopping
├── runtime.py                  Environment/logging setup
├── reference.py                Reference-pattern loading & distance-constraint bootstrap
├── validation.py               Config & reference validation
├── modules/
│   ├── extraction.py           Parse CIF/XML into ReferenceInfo
│   ├── generation.py           PyXtal random generation; mutation orchestration
│   ├── mutation.py             The seven Wyckoff-site mutation operators
│   ├── distance_constraint.py  Melt-quench-anneal (MQA) MD for distance tolerances
│   ├── relaxation.py           ASE (sequential) / torch-sim (batch) FIRE relaxation
│   └── postprocess.py          XRD metrics, selection, deduplication, output writing
├── mlips/
│   └── sevennet.py             SevenNet calculator/model factory
└── utils/
    ├── types.py                Environment, ReferenceInfo, RelaxResult data types
    ├── core_utils.py           Lattice/volume/Z helpers, YAML I/O, POSCAR formatting
    ├── xrd_utils.py            R-factor, Wasserstein distance, Gaussian XRD helpers
    └── torchsim_utils.py       torch-sim batching helpers
```

---

## Documentation

Full manual: [`docs/MANUAL.md`](docs/MANUAL.md).

## License

Distributed under the [MIT License](LICENSE).

## Citation / contact

Author: Yunsung Lim (<yunsunglim@snu.ac.kr>).
