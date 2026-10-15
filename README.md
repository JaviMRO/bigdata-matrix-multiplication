# Big Data — Matrix Multiplication Assignments

Individual assignments for the Big Data course (Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027).
Author: Javier Martín Rodríguez.

| Folder | Assignment | Status |
|---|---|---|
| `assignment1/` | Basic triple-loop matrix multiplication in Python, Java and C: runtime and memory vs. matrix size | in progress |

## Environment

Windows 11 (native), PowerShell, Intel Core Ultra 9 275HX. Full details are collected automatically into
`assignment1/results/system_info.json`.

| Tool | Version |
|---|---|
| CPython | 3.12.10 |
| Java | Eclipse Temurin OpenJDK 25.0.4.1 LTS (HotSpot) |
| C compiler | gcc 16.1.0 (MinGW-W64 x86_64-ucrt-posix-seh) |
| Python packages | see `requirements.txt` (numpy 2.2.6, pandas 3.0.0, matplotlib 3.10.8, psutil 7.2.2) |

```powershell
pip install -r requirements.txt
```

## Assignment 1 — commands

All commands are run from the repository root.

```powershell
# 1. Generate the shared input matrices (data/A_n{n}.bin, data/B_n{n}.bin) and data/manifest.json
python assignment1/scripts/generate_inputs.py

# 2. Record the machine and toolchain description in results/system_info.json
python assignment1/scripts/system_info.py

# 3. Compile C (gcc -O2 -std=c17 -Wall -Wextra) and Java into assignment1/build/
powershell -ExecutionPolicy Bypass -File assignment1/scripts/build.ps1

# 4. Correctness tests for all three implementations -> results/validation.csv (~6 min)
python assignment1/scripts/validate.py

# 5. Benchmarks -> results/raw/benchmark_<timestamp>.csv (+ _meta.json)
#    Refuses to start unless on AC power, power mode "Best performance" and validation passed.
python assignment1/scripts/run_benchmarks.py
#    Quick test of the pipeline (writes to results/dryrun/, only warns on failed checks):
python assignment1/scripts/run_benchmarks.py --dry-run --sizes 64 128 --reps 2

# 6. Tables (results/tables/*.csv, *.tex) and figures (figures/*.png, *.pdf) from the latest raw CSV
python assignment1/scripts/analyze.py
```

The input `.bin` files are not committed; step 1 regenerates them bit-for-bit
(the SHA-256 of every file is stored in `data/manifest.json`).
