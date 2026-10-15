"""Correctness tests for the C, Java and Python implementations.

Every implementation is run through its normal command line with --dump, and the
dumped C matrix is compared against the expected result:

  1. identity_right / identity_left: A*I = A and I*B = B (exact)
  2. zero:            A*0 = 0 (exact)
  3. hand_2x2 / hand_3x3: small integer-valued matrices checked by hand (exact)
  4. random:          random A, B for n in {1, 2, 7, 33, 64, 128, 256} vs numpy.matmul
  5. benchmark_input: the benchmark input files in data/ vs numpy.matmul (skip with --quick)

Non-exact tests use the element-wise forward-error bound for inner products
(Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., 2002):
    |C - C_ref| <= 2 * gamma_n * (|A| @ |B|),   gamma_n = n*u / (1 - n*u),   u = 2^-53
The factor 2 covers the rounding errors of the reference itself, which sums in a different order.

Every run also checks the stdout format, that each repetition prints the same checksum
(C is overwritten, not accumulated) and that this checksum equals the row-major sum of the dump.

Results go to results/validation.csv. Exit code 0 only if every test passes.

Usage (from any folder, after building with scripts/build.ps1):
    python assignment1/scripts/validate.py            # all tests (several minutes)
    python assignment1/scripts/validate.py --quick    # skip the benchmark-size tests
"""

import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_inputs import BASE_SEED, random_matrix, sha256_of, write_matrix

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
WORK_DIR = ASSIGNMENT_DIR / "results" / "dumps" / "validation"  # git-ignored
OUTPUT_PATH = ASSIGNMENT_DIR / "results" / "validation.csv"

# How each implementation is started. Paths are absolute because Windows resolves a
# relative executable against the parent's working directory, not subprocess's cwd.
COMMANDS = {
    "c": [str(ASSIGNMENT_DIR / "build" / "matmul_c.exe")],
    "java": ["java", "-Xmx4g", "-cp", str(ASSIGNMENT_DIR / "build"), "MatMul"],
    "python": [sys.executable, str(ASSIGNMENT_DIR / "src" / "python" / "matmul.py")],
}
# Built artefact -> source, used to detect a stale build.
BUILD_OUTPUTS = {
    "c": ("build/matmul_c.exe", "src/c/matmul.c"),
    "java": ("build/MatMul.class", "src/java/MatMul.java"),
}

RANDOM_SIZES = [1, 2, 7, 33, 64, 128, 256]
# Largest benchmark size validated per language: Python n = 2048 exceeds the 600 s budget.
MAX_BENCHMARK_N = {"c": 2048, "java": 2048, "python": 1024}
TIMEOUT_S = 600
UNIT_ROUNDOFF = 2.0 ** -53

# Hand-computed products of integer-valued matrices (every value is exact in float64).
HAND_CASES = {
    "hand_2x2": (
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[19, 22], [43, 50]],
    ),
    "hand_3x3": (
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        [[30, 24, 18], [84, 69, 54], [138, 114, 90]],
    ),
}


def gamma(n):
    """gamma_n = n*u / (1 - n*u) from Higham's error analysis."""
    return n * UNIT_ROUNDOFF / (1 - n * UNIT_ROUNDOFF)


def build_test_cases(quick):
    """Return the list of test cases: dicts with name, n, input paths, matrices, expected C, exact."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cases = []

    def add_case(name, a, b, expected, exact):
        n = a.shape[0]
        a_path = WORK_DIR / f"{name}_n{n}_A.bin"
        b_path = WORK_DIR / f"{name}_n{n}_B.bin"
        write_matrix(a, a_path)
        write_matrix(b, b_path)
        cases.append({"test": name, "n": n, "a_path": a_path, "b_path": b_path,
                      "a": a, "b": b, "expected": expected, "exact": exact, "max_n": None})

    # 1-2. Identity and zero matrices (n = 33: odd and not a power of two).
    n = 33
    a = random_matrix(n, BASE_SEED)
    b = random_matrix(n, BASE_SEED + 1)
    identity = np.eye(n)
    zero = np.zeros((n, n))
    add_case("identity_right", a, identity, a, exact=True)
    add_case("identity_left", identity, b, b, exact=True)
    add_case("zero", a, zero, zero, exact=True)

    # 3. Hand-checked small matrices. Guard against typos in the hand-written values.
    for name, (a_list, b_list, c_list) in HAND_CASES.items():
        a, b, c = (np.array(m, dtype=np.float64) for m in (a_list, b_list, c_list))
        assert np.array_equal(a @ b, c), f"hand-computed result of {name} is wrong"
        add_case(name, a, b, c, exact=True)

    # 4. Random matrices against NumPy (same seed rule as the benchmark inputs).
    for n in RANDOM_SIZES:
        a = random_matrix(n, BASE_SEED)
        b = random_matrix(n, BASE_SEED + 1)
        add_case("random", a, b, np.matmul(a, b), exact=False)

    # 5. The benchmark input files themselves, after checking their SHA-256.
    if not quick:
        with open(DATA_DIR / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest["sizes"].values():
            n = entry["n"]
            paths = {}
            for name in ("A", "B"):
                path = DATA_DIR / entry[name]["file"]
                if not path.exists() or sha256_of(path) != entry[name]["sha256"]:
                    sys.exit(f"{path.name} is missing or differs from the manifest: "
                             "run generate_inputs.py first")
                paths[name] = path
            a = np.fromfile(paths["A"], dtype="<f8").reshape(n, n)
            b = np.fromfile(paths["B"], dtype="<f8").reshape(n, n)
            cases.append({"test": "benchmark_input", "n": n, "a_path": paths["A"],
                          "b_path": paths["B"], "a": a, "b": b, "expected": np.matmul(a, b),
                          "exact": False, "max_n": MAX_BENCHMARK_N})
    return cases


def check_stdout(stdout, language, n, warmup, reps):
    """Check one CSV line per repetition in the contract format; return (problems, checksums)."""
    expected = [("warmup", r) for r in range(1, warmup + 1)] + \
               [("measured", r) for r in range(1, reps + 1)]
    lines = stdout.splitlines()
    if len(lines) != len(expected):
        return [f"expected {len(expected)} output lines, got {len(lines)}"], []

    problems = []
    checksums = []
    for line, (phase, rep) in zip(lines, expected):
        fields = line.split(",")
        if len(fields) != 6:
            problems.append(f"bad line: {line!r}")
            continue
        lang, n_text, phase_text, rep_text, time_text, checksum_text = fields
        try:
            if (lang, int(n_text), phase_text, int(rep_text)) != (language, n, phase, rep):
                problems.append(f"unexpected fields: {line!r}")
            if int(time_text) < 0:
                problems.append(f"negative time: {line!r}")
            checksums.append(float(checksum_text))
        except ValueError:
            problems.append(f"unparsable line: {line!r}")
    return problems, checksums


def compare(result, case):
    """Error statistics of the dumped result against the expected matrix."""
    expected = case["expected"]
    error = np.abs(result - expected)
    bound = 2 * gamma(case["n"]) * (np.abs(case["a"]) @ np.abs(case["b"]))

    nonzero = expected != 0
    max_rel = float(np.max(error[nonzero] / np.abs(expected[nonzero]))) if nonzero.any() else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bound > 0, error / bound, np.where(error == 0, 0.0, np.inf))

    if case["exact"]:
        passed = bool(np.array_equal(result, expected))
    else:
        passed = bool(np.all(error <= bound))
    return passed, float(error.max()), max_rel, float(ratio.max())


def run_case(language, case):
    """Run one implementation on one test case; return a row for validation.csv."""
    n = case["n"]
    # Small inputs: 1 warm-up + 2 measured reps, so we also see that C is overwritten.
    # Benchmark sizes: a single repetition is enough (and keeps the run time reasonable).
    warmup, reps = (1, 2) if case["test"] != "benchmark_input" else (0, 1)
    dump_path = WORK_DIR / f"dump_{language}_{case['test']}_n{n}.bin"
    dump_path.unlink(missing_ok=True)

    command = COMMANDS[language] + [
        "--n", str(n), "--a", str(case["a_path"]), "--b", str(case["b_path"]),
        "--warmup", str(warmup), "--reps", str(reps), "--dump", str(dump_path),
    ]
    row = {"test": case["test"], "language": language, "n": n, "exact": case["exact"],
           "status": "FAIL", "max_abs_error": None, "max_rel_error": None,
           "max_error_over_bound": None, "checksum_consistent": False,
           "wall_time_s": None, "note": ""}

    start = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=ASSIGNMENT_DIR, capture_output=True, text=True,
                              timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        row["note"] = f"timeout after {TIMEOUT_S} s"
        return row
    row["wall_time_s"] = round(time.perf_counter() - start, 2)

    problems = []
    if proc.returncode != 0:
        problems.append(f"exit code {proc.returncode}: {proc.stderr.strip()[:200]}")
    format_problems, checksums = check_stdout(proc.stdout, language, n, warmup, reps)
    problems += format_problems

    if dump_path.exists() and dump_path.stat().st_size == 8 * n * n:
        result = np.fromfile(dump_path, dtype="<f8").reshape(n, n)
        dump_path.unlink()
        passed, max_abs, max_rel, ratio = compare(result, case)
        row.update(max_abs_error=max_abs, max_rel_error=max_rel, max_error_over_bound=ratio)
        if not passed:
            problems.append("result outside tolerance" if not case["exact"] else "result not exact")

        # np.cumsum adds strictly left to right, like the kernels' checksum loop.
        dump_sum = float(np.cumsum(result.ravel())[-1])
        row["checksum_consistent"] = bool(checksums) and all(c == dump_sum for c in checksums)
        if not row["checksum_consistent"]:
            problems.append(f"printed checksums {checksums} != sum of dump {dump_sum!r}")
    else:
        problems.append("dump missing or wrong size")

    row["note"] = "; ".join(problems)
    row["status"] = "PASS" if not problems else "FAIL"
    return row


def check_build(languages):
    """Stop early if a compiled implementation is missing or older than its source."""
    for language in languages:
        if language not in BUILD_OUTPUTS:
            continue
        built, source = (ASSIGNMENT_DIR / p for p in BUILD_OUTPUTS[language])
        if not built.exists() or built.stat().st_mtime < source.stat().st_mtime:
            sys.exit(f"{built.name} is missing or older than {source.name}: run scripts/build.ps1")


def main():
    parser = argparse.ArgumentParser(description="Correctness tests for the three implementations.")
    parser.add_argument("--quick", action="store_true", help="skip the benchmark-size tests")
    parser.add_argument("--languages", nargs="+", choices=list(COMMANDS), default=list(COMMANDS))
    args = parser.parse_args()

    check_build(args.languages)
    cases = build_test_cases(args.quick)
    validated_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    rows = []
    for case in cases:
        for language in args.languages:
            if case["max_n"] is not None and case["n"] > case["max_n"][language]:
                continue  # documented limit, e.g. Python at n = 2048
            row = run_case(language, case)
            row["validated_at"] = validated_at
            rows.append(row)
            print(f"{row['status']}  {row['test']:<16} {language:<7} n={row['n']:<5} "
                  f"{row['wall_time_s']} s  {row['note']}", flush=True)

    columns = ["validated_at", "test", "language", "n", "exact", "status", "max_abs_error",
               "max_rel_error", "max_error_over_bound", "checksum_consistent", "wall_time_s", "note"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    table = pd.DataFrame(rows, columns=columns).drop(columns=["validated_at", "note"])
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    failed = sum(row["status"] != "PASS" for row in rows)
    print(f"\n{len(rows) - failed}/{len(rows)} passed. Wrote {OUTPUT_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
