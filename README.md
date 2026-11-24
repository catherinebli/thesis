# rowing-ee: wearable IMU energy expenditure for erg rowing

First-pass code for the full-body energy-expenditure wearable. Five ICM-20948 IMUs (forearm, upper arm, chest, thigh, shank, one side) on a Teensy 4.1. Each stroke is segmented, turned into a 904-value feature vector, and mapped to metabolic power with a linear model, following [Slade et al. 2021](https://doi.org/10.1038/s41467-021-24173-x). Power is then integrated into kcal.

> **Status: first pass.** The pipeline runs end to end, and the C++ core and the Python package agree to about 1e-13 on synthetic logs.
>
> **Not done yet:**
> * The model weights are a **placeholder (all zeros)**, so every stroke is billed at the resting floor until a model is trained on respirometry data.
> * The Teensy build has not been compiled on the Teensy toolchain yet. `lib/hal` and `main.cpp` are syntax-checked against stub headers; `lib/core` is compiled and tested natively.
> * `detector.thr_min` and the `mount:` matrices still need values from pilot data.

## Quick start

```bash
pip install numpy scipy pyyaml matplotlib pytest

# try everything without hardware
python scripts/make_synthetic_log.py data/raw/synthetic.bin --spm 20 24
python scripts/run_session.py data/raw/synthetic.bin --mass 72 --height 1.83 --age 22 --sex M --plot

# tests: Python units + C++ replay parity + header and HAL compile checks (needs g++ and make)
python -m pytest tests
python scripts/check_parity.py data/raw/synthetic.bin

# firmware (PlatformIO)
python scripts/gen_headers.py
cd firmware && pio run -e teensy41 -t upload && pio device monitor
```

On a real session: copy `raw_###.bin` and `est_###.csv` off the SD card into `data/raw/`, then `python scripts/run_session.py data/raw/raw_000.bin --subject S01 --plot`. Subjects go in `data/subjects.yaml`, which is gitignored, like all participant data.

## Docs

| | |
|---|---|
| [1. Method and math](docs/01_method_and_math.md) | Segmentation, filtering, binning, features, normalization, ridge/lasso, metrics, resting floor, energy |
| [2. Rowing adaptation](docs/02_rowing_adaptation.md) | 5 IMUs on one side, why strokes are not labels, why mirroring the other side adds nothing |
| [3. Issues and risks](docs/03_issues_and_risks.md) | Protocol, segmentation, features, validation, I2C and SD hardware, bugs in Slade's harness |
| [4. Pseudocode](docs/04_pseudocode.md) | Raw counts → watts → kcal, step by step |
| [5. Architecture](docs/05_architecture.md) | Class diagrams, file map, critical functions, workflow |
| [6. Firmware and data formats](docs/06_firmware_and_data_formats.md) | Wiring, ICM-20948 registers, build and flash, `subject.cfg`, `raw.bin` and `est.csv` layouts |

## Layout

```
config/pipeline.yaml    single source of truth; scripts/gen_headers.py turns it into firmware/include/*.h
firmware/               Teensy 4.1 (PlatformIO): src/main.cpp, lib/hal (hardware), lib/core (portable DSP)
python/rowing_ee/       line-for-line Python mirror of lib/core, plus log I/O, header export, synthetic data
scripts/                run_session, plot_session, gen_headers, check_parity, make_synthetic_log
tests/                  pytest suite
docs/                   math, pseudocode, issues, architecture, formats, diagrams
```

Based on: P. Slade, M. J. Kochenderfer, S. L. Delp, S. H. Collins, *Sensing leg movement enhances wearable monitoring of energy expenditure*, Nat. Commun. 12, 4312 (2021). Their code and data: [simtk.org/projects/energy-est](https://simtk.org/projects/energy-est). No code from that project is copied here.
