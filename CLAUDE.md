# CLAUDE.md — Big Data Individual Assignments (ULPGC 2026-2027)

Author: Javier Martín Rodríguez · Course: Big Data, Grado en Ciencia e Ingeniería de Datos (ULPGC)
This repository will hold the four matrix-multiplication assignments. **Current scope: Assignment 1 only.**
Deadline for Assignment 1: **16 October 2026**.

Everything committed to this repo (code, comments, README, report) must be in **English**.
The student must understand and be able to defend every line: keep code simple, readable and commented.

---

## Assignment 1 — research question

> How does matrix size *n* affect the runtime and memory use of the basic triple-loop dense
> matrix multiplication implemented in Python, Java and C, on one fixed Windows laptop?

Hypotheses (to be stated as predictions, not results):
1. Runtime of all three grows approximately as O(n³); the log-log slope approaches 3 for larger n,
   possibly above 3 once matrices exceed cache (column-wise access of B in i-j-k order).
2. C (-O2) < Java (after JIT warm-up) << CPython, by constant factors; the Java–C gap is small, CPython is ~1–2 orders slower.
3. Peak process memory is dominated by runtime overhead at small n and by matrix storage at large n;
   CPython uses far more memory per element (boxed float objects) than C/Java (8 bytes per double).

---

## Environment (fixed for all runs)

- OS: Windows 11 (native, no WSL). Editor: VS Code. Shell: PowerShell.
- CPU: Intel Core Ultra 9 275HX (hybrid: P-cores + E-cores). GPU not used.
- RAM, exact core counts, OS build: **collected automatically** by `scripts/system_info.py`.
- Python: CPython 3.12.10 (kernel uses **pure Python only**; NumPy allowed only for input generation and validation).
- Java: Eclipse Temurin OpenJDK 25.0.4.1 LTS (HotSpot, tiered JIT, default GC). Run with `-Xmx4g`.
- C: gcc 16.1.0 MinGW-W64 x86_64-ucrt-posix-seh. Flags: **`-O2` only** (no `-march=native`, no `-ffast-math`).
- Laptop on AC power, Windows power mode "Best performance", other apps closed. Record this in `results/run_notes.md`.

---

## Repository layout

```
.
├── CLAUDE.md
├── README.md                      # exact build + run commands, dependency versions
├── requirements.txt               # numpy, pandas, matplotlib, psutil (pinned versions)
├── .gitignore                     # ignore data/*.bin, build/, __pycache__, results/dumps/
├── assignment1/
│   ├── src/
│   │   ├── c/matmul.c
│   │   ├── java/MatMul.java
│   │   └── python/matmul.py
│   ├── scripts/
│   │   ├── build.ps1              # compiles C and Java into build/
│   │   ├── generate_inputs.py     # writes data/A_n{n}.bin, B_n{n}.bin + data/manifest.json
│   │   ├── validate.py            # correctness tests for the 3 implementations
│   │   ├── system_info.py         # writes results/system_info.json
│   │   ├── run_benchmarks.py      # orchestrator: timing + memory + timeouts → results/raw/*.csv
│   │   └── analyze.py             # tables (CSV + LaTeX) and figures from raw CSVs
│   ├── data/                      # generated, .bin files git-ignored, manifest.json committed
│   ├── results/
│   │   ├── raw/                   # one CSV row per measured/warm-up repetition (never delete rows)
│   │   ├── tables/
│   │   ├── system_info.json
│   │   └── run_notes.md
│   ├── figures/
│   └── report/                    # LaTeX report (Overleaf)
```

---

## Shared inputs

- Generated once by `generate_inputs.py` with `numpy.random.default_rng(seed)` (PCG64).
- Base seed `20261016`; A uses `seed`, B uses `seed + 1`, independently per n (record both in manifest).
- Values: uniform in [-1, 1), dtype float64.
- File format: raw **little-endian float64, row-major, no header**, exactly 8·n² bytes.
- `manifest.json`: for each n → file names, shape, dtype, seed, generator, value range, SHA-256 of each file,
  and the NumPy reference checksum (sum of all elements of A@B) and Frobenius norm of A@B.
- All three implementations read **the same files**. The kernel never generates data.

## Implementation rules (all three languages)

- Algorithm: textbook **i-j-k** triple loop, accumulating into a local scalar `sum`, then `C[i][j] = sum`.
  Same loop order and same arithmetic in all languages. O(n³) multiply-adds. No library matmul, no blocking,
  no loop reordering, no threads, no SIMD intrinsics. (Optimisations belong to Assignment 2.)
- Numeric type: IEEE-754 double (C `double`, Java `double`, Python `float`).
- Storage layouts (document in report):
  - C: one contiguous `malloc`'d `double` array per matrix, row-major, index `i*n + j`.
  - Java: one `double[]` per matrix, row-major, index `i*n + j` (contiguous, like C).
  - Python: list of lists of `float` (idiomatic CPython; each element is a boxed object — unavoidable difference).
- Output matrix C is allocated **once before** the timed region and overwritten in each repetition.
  Timed region = **multiplication kernel only** (no file I/O, no allocation, no checksum).
- Monotonic timers: C `QueryPerformanceCounter` (or `clock_gettime(CLOCK_MONOTONIC)`), Java `System.nanoTime()`,
  Python `time.perf_counter_ns()`. Report nanoseconds.
- After each repetition compute a checksum (sum of C) **outside** the timed region and print it,
  so the result is consumed and cannot be optimised away.

### Common command-line contract

```
<program> --n N --a PATH --b PATH --warmup W --reps R [--dump PATH]
```
- Runs W warm-up repetitions then R measured repetitions **in the same process**.
- Prints one line per repetition to stdout, flushed immediately:
  `language,n,phase,rep,time_ns,checksum` with phase ∈ {warmup, measured}.
- `--dump PATH` writes the final C as raw little-endian float64 (used for validation only).
- Exit code 0 on success, non-zero on error.

Run commands:
- C: `build\matmul_c.exe ...` built with `gcc -O2 -std=c17 -Wall -Wextra -o build/matmul_c.exe src/c/matmul.c`
- Java: `javac -d build src/java/MatMul.java` then `java -Xmx4g -cp build MatMul ...`
- Python: `python src/python/matmul.py ...`

---

## Correctness (validate.py) — must pass before any benchmark

Test cases, run on all three implementations through the same CLI with `--dump`:
1. Identity: A random, B = I → C = A (and I·B = B).
2. Zero matrix → C = 0 exactly.
3. Hand-checked 2×2 and 3×3 with integer-valued doubles → exact match.
4. Random matrices for n ∈ {1, 2, 7, 33, 64, 128, 256} compared against `numpy.matmul`.
5. Benchmark sizes: compare each implementation's dumped result against NumPy (done once, not timed).

Tolerance (justify in report): element-wise forward-error bound for inner products,
`|C − C_ref| ≤ γ_n · (|A|·|B|)` with `γ_n = n·u / (1 − n·u)`, `u = 2^-53` (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., 2002). Use `2·γ_n` to cover differences in the reference's own summation order.
Also report max absolute and max relative error per test. Write results to `results/validation.csv`.

---

## Benchmark protocol (run_benchmarks.py)

- Sizes: **n = 64, 128, 256, 512, 1024** for all three; **n = 2048** for C and Java.
  Python n = 2048 is **attempted once** to document the limit (expected timeout).
- Repetitions: **10 measured** for n ≤ 256, **5 measured** for n ≥ 512.
- Warm-up (same policy for every language, excluded from statistics but kept in raw CSV):
  **2** warm-ups for n ≤ 512, **1** for n ≥ 1024.
- Budgets (declared before running):
  - Time: a single multiplication may take at most **600 s**. The orchestrator kills the process if no new
    output line arrives within 600 s and records `status=timeout` (no invented time).
  - Memory: peak working set ≤ **50 % of physical RAM**; otherwise record `status=memory_exceeded`.
- Each (language, n) runs in a **fresh process**. Order: ascending n; inside each n, rotate the language order
  to avoid systematic thermal bias. Sleep 5 s between processes.
- Process control: set HIGH priority and pin the benchmark process to **one P-core** logical CPU.
  Detect P-cores with `GetSystemCpuSetInformation` (highest `EfficiencyClass`) via ctypes; if detection fails,
  fall back to no pinning and record that in system_info. Record the chosen CPU index.
- Memory metric (same tool for all three): **peak working set** of the process
  (`PeakWorkingSetSize` from `GetProcessMemoryInfo`, read via ctypes on the `Popen` handle after the process
  exits; `psutil` as fallback). Also store the theoretical matrix storage 3·8·n² bytes for comparison.
  Optional extra columns: Java heap used (`MemoryMXBean`), Python `tracemalloc` peak in a separate run only
  (tracemalloc slows execution, never mix with timing runs).
- Output `results/raw/benchmark_<timestamp>.csv` columns:
  `timestamp,language,n,phase,rep,time_ns,checksum,checksum_ref,checksum_ok,peak_working_set_bytes,`
  `theoretical_bytes,warmup_count,status,cpu_affinity,compiler_or_runtime,flags`

## Analysis (analyze.py)

- Per (language, n): median, Q1, Q3, IQR, min, max, count of measured reps → `results/tables/summary.csv`
  and `summary.tex` (booktabs).
- Figures (PNG + PDF, labelled axes with units, 300 dpi):
  1. Median time vs n, log-log, IQR error bars, three languages + dashed n³ reference line.
  2. Normalised time per inner-loop iteration `t / n³` (ns) vs n — shows constants and cache effects.
  3. Peak working set vs n (MB) per language, plus theoretical 24·n² bytes line.
  4. Runtime ratios Python/C and Java/C vs n.
- Fit log(t) = a + b·log(n) per language on n ≥ 256; report slope b with a 95 % CI.
- List timeouts / memory failures in a separate table: largest *feasible* n vs largest *attempted* n.

## Report (LaTeX, English)

Sections (as required by the course): Title + abstract · Introduction & objective · Background & references ·
Methodology · Results · Discussion · Conclusions · References & materials · AI-use disclosure
(tool: Claude / Claude Code; purpose: planning, code generation, debugging, language revision; all
measurements produced by running the code on the author's laptop). IEEE reference style.
File name: `Assignment01_MartinRodriguez_Report.pdf`. Materials: `Assignment01_MartinRodriguez_Materials.zip`.

Conclusions must be scoped to these implementations, settings, inputs and this laptop — never "language X is faster".
