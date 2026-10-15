"""Benchmark orchestrator for Assignment 1.

Runs every (language, n) in a fresh process, pinned to one P-core at HIGH priority,
and writes one CSV row per warm-up/measured repetition to
results/raw/benchmark_<timestamp>.csv (plus a _meta.json file describing the run).

Protocol (declared before running, see CLAUDE.md):
  * n = 64 ... 1024 for all languages, n = 2048 for C and Java; Python n = 2048 is
    attempted once (0 warm-ups, 1 repetition) to document the limit.
  * 10 measured repetitions for n <= 256, 5 for n >= 512;
    2 warm-ups for n <= 512, 1 for n >= 1024.
  * Ascending n; the language order is rotated for each n; 5 s pause between processes.
  * Time budget: the process is killed if no new output line arrives within 600 s
    (status=timeout). Memory budget: peak working set <= 50 % of physical RAM
    (status=memory_exceeded). No time is ever invented for a failed repetition.
  * Memory metric: PeakWorkingSetSize from GetProcessMemoryInfo, read on the process
    handle after the process exits (psutil as fallback).

Usage (from any folder):
    python assignment1/scripts/run_benchmarks.py                                 # full run
    python assignment1/scripts/run_benchmarks.py --dry-run --sizes 64 128 --reps 2

A dry run writes to results/dryrun/ (git-ignored) and only warns about failed
pre-flight checks (power settings, validation). A full run refuses to start
unless they pass (or --force is given).
"""

import argparse
import csv
import ctypes
import datetime
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import psutil

import system_info
from generate_inputs import sha256_of
from validate import check_build, gamma

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
RAW_DIR = ASSIGNMENT_DIR / "results" / "raw"
DRY_RUN_DIR = ASSIGNMENT_DIR / "results" / "dryrun"  # git-ignored
VALIDATION_PATH = ASSIGNMENT_DIR / "results" / "validation.csv"

SIZES = [64, 128, 256, 512, 1024, 2048]
LANGUAGES = ["c", "java", "python"]
PYTHON_LIMIT_N = 2048           # attempted once, expected to time out
LINE_TIMEOUT_S = 600            # max time between two output lines
SLEEP_BETWEEN_S = 5             # pause between processes
POLL_INTERVAL_S = 0.25          # how often the budgets are checked
FLAGS = {"c": "-O2 -std=c17 -Wall -Wextra", "java": "-Xmx4g", "python": ""}

CSV_COLUMNS = ["timestamp", "language", "n", "phase", "rep", "time_ns", "checksum",
               "checksum_ref", "checksum_ok", "peak_working_set_bytes", "theoretical_bytes",
               "warmup_count", "status", "cpu_affinity", "compiler_or_runtime", "flags"]

CREATE_SUSPENDED = 0x00000004


def kernel_python():
    """Interpreter that runs the Python kernel.

    Inside a venv on Windows, python.exe is a small launcher that starts the real
    interpreter as a child process; pinning and memory would then apply to the
    launcher. The base interpreter is used instead (the kernel needs no packages).
    """
    if sys.prefix != sys.base_prefix:
        return str(Path(sys.base_prefix) / "python.exe")
    return sys.executable


# Absolute paths: Windows resolves a relative executable against the parent's directory.
COMMANDS = {
    "c": [str(ASSIGNMENT_DIR / "build" / "matmul_c.exe")],
    "java": ["java", "-Xmx4g", "-cp", str(ASSIGNMENT_DIR / "build"), "MatMul"],
    "python": [kernel_python(), str(ASSIGNMENT_DIR / "src" / "python" / "matmul.py")],
}


def repetitions(language, n, reps_override=None):
    """(warm-ups, measured repetitions) for one process, as declared in the protocol.

    reps_override (dry runs only) replaces the number of measured repetitions.
    """
    if language == "python" and n >= PYTHON_LIMIT_N:
        return 0, 1
    warmup = 2 if n <= 512 else 1
    reps = 10 if n <= 256 else 5
    return warmup, reps_override or reps


def checksum_tolerance(n, abs_product_sum):
    """Worst-case bound for |checksum - checksum_ref|.

    Each element of C may differ from NumPy's by 2*gamma_n*(|A||B|)_ij (the bound used
    in validate.py). The kernels then add the n^2 elements one by one, which adds at
    most gamma_{n^2} * sum|C|, and sum|C| <= sum(|A||B|) to first order.
    """
    return (2 * gamma(n) + gamma(n * n)) * abs_product_sum


# ------------------------------------------------------------ process control

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_get_process_memory_info = ctypes.WinDLL("kernel32", use_last_error=True).K32GetProcessMemoryInfo
_get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                     wintypes.DWORD]
_get_process_memory_info.restype = wintypes.BOOL


def peak_working_set(proc):
    """Peak working set (bytes) of a Popen process; works during and after its life.

    Popen keeps the process handle returned by CreateProcess in the private attribute
    _handle until the Popen object is destroyed, so the counters can still be read
    after the process has exited.
    """
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    if _get_process_memory_info(int(proc._handle), ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize
    try:  # fallback: psutil only works while the process is still alive
        return psutil.Process(proc.pid).memory_info().peak_wset
    except (psutil.Error, AttributeError):
        return None


def launch(command, cpu):
    """Start a process at HIGH priority, pinned to `cpu` (None = no pinning).

    The process is created suspended, pinned, and only then resumed, so not a single
    instruction (including JVM or interpreter start-up) runs on another core.
    """
    proc = subprocess.Popen(command, cwd=ASSIGNMENT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1,
                            creationflags=CREATE_SUSPENDED | subprocess.HIGH_PRIORITY_CLASS)
    try:
        process = psutil.Process(proc.pid)
        if cpu is not None:
            process.cpu_affinity([cpu])
        process.resume()
    except psutil.Error:
        proc.kill()
        raise
    return proc


def _pump(stream, sink):
    """Reader thread: put every line of `stream` into the queue, then None at EOF."""
    for line in stream:
        sink.put(line)
    sink.put(None)


def _drain(source):
    """All text a _pump queue received, up to its EOF marker (call after the process ended)."""
    parts = []
    while (line := source.get()) is not None:
        parts.append(line)
    return "".join(parts)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


# --------------------------------------------------------------- one process

def run_process(language, n, entry, cpu, budget_bytes, runtime, reps_override):
    """Run one (language, n) process and return its CSV rows."""
    warmup, reps = repetitions(language, n, reps_override)
    expected = [("warmup", r) for r in range(1, warmup + 1)] + \
               [("measured", r) for r in range(1, reps + 1)]
    command = COMMANDS[language] + [
        "--n", str(n),
        "--a", str(DATA_DIR / entry["A"]["file"]),
        "--b", str(DATA_DIR / entry["B"]["file"]),
        "--warmup", str(warmup), "--reps", str(reps),
    ]
    reference = entry["reference"]
    tolerance = checksum_tolerance(n, reference["abs_product_sum"])

    print(f"[{language} n={n}] warmup={warmup} reps={reps}", flush=True)
    proc = launch(command, cpu)
    stdout_lines = queue.Queue()
    stderr_lines = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, stdout_lines), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, stderr_lines), daemon=True).start()

    received = []  # (timestamp, line)
    status = "ok"
    last_output = time.monotonic()
    try:
        while True:
            try:
                line = stdout_lines.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                line = ""  # nothing new: only check the budgets
            if line is None:
                break  # stdout closed: the process has finished
            if line.strip():
                received.append((now_iso(), line.strip()))
                last_output = time.monotonic()
                print(f"    {line.strip()}", flush=True)

            peak = peak_working_set(proc)
            if peak is not None and peak > budget_bytes:
                status = "memory_exceeded"
                break
            if time.monotonic() - last_output > LINE_TIMEOUT_S:
                status = "timeout"
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    peak = peak_working_set(proc)  # final value, read after the process exited
    stderr_text = _drain(stderr_lines).strip()
    if status == "ok" and proc.returncode != 0:
        status = "error"
        print(f"    exit code {proc.returncode}: {stderr_text[:300]}", flush=True)

    common = {"language": language, "n": n, "checksum_ref": reference["checksum"],
              "peak_working_set_bytes": peak, "theoretical_bytes": 3 * 8 * n * n,
              "warmup_count": warmup, "cpu_affinity": "none" if cpu is None else cpu,
              "compiler_or_runtime": runtime[language], "flags": FLAGS[language]}
    rows = []
    for (timestamp, line), (phase, rep) in zip(received, expected):
        fields = line.split(",")
        try:
            ok_line = len(fields) == 6 and fields[0] == language and int(fields[1]) == n \
                and fields[2] == phase and int(fields[3]) == rep
            time_ns, checksum = int(fields[4]), float(fields[5])
        except (ValueError, IndexError):
            ok_line = False
        if not ok_line:
            status = "error"
            print(f"    unexpected output line: {line!r}", flush=True)
            break
        rows.append(common | {"timestamp": timestamp, "phase": phase, "rep": rep,
                              "time_ns": time_ns, "checksum": checksum,
                              "checksum_ok": abs(checksum - reference["checksum"]) <= tolerance,
                              "status": "ok"})

    # A repetition that never reported gets a row with its status and no time.
    if len(rows) < len(expected):
        if status == "ok":
            status = "error"  # exited "successfully" without printing every repetition
        phase, rep = expected[len(rows)]
        rows.append(common | {"timestamp": now_iso(), "phase": phase, "rep": rep,
                              "time_ns": "", "checksum": "", "checksum_ok": "", "status": status})
    print(f"    -> {status}, peak working set {peak / 2**20:.1f} MiB" if peak else
          f"    -> {status}, peak working set unknown", flush=True)
    return rows


# ------------------------------------------------------------------ pre-flight

def preflight(sizes, languages, manifest, strict):
    """Check inputs, build, validation and power settings. Returns the problems found."""
    problems = []
    check_build(languages)  # exits if a binary is missing or stale

    for n in sizes:
        entry = manifest["sizes"].get(str(n))
        if entry is None:
            sys.exit(f"n={n} is not in data/manifest.json: run generate_inputs.py --sizes {n}")
        for name in ("A", "B"):
            path = DATA_DIR / entry[name]["file"]
            if not path.exists() or sha256_of(path) != entry[name]["sha256"]:
                sys.exit(f"{path.name} is missing or differs from the manifest")

    # Correctness must be established before benchmarking.
    if not VALIDATION_PATH.exists():
        problems.append("results/validation.csv not found: run validate.py")
    else:
        with open(VALIDATION_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        failed = [r for r in rows if r["status"] != "PASS"]
        validated = {(r["language"], int(r["n"])) for r in rows if r["test"] == "benchmark_input"}
        missing = [(lang, n) for lang in languages for n in sizes
                   if (lang, n) not in validated and not (lang == "python" and n >= PYTHON_LIMIT_N)]
        if failed:
            problems.append(f"{len(failed)} validation test(s) failed")
        if missing:
            problems.append(f"benchmark inputs not validated for {missing} (run validate.py without --quick)")

    power = system_info.collect_power()
    if power.get("ac_power") is not True:
        problems.append("laptop is not on AC power")
    if power.get("power_mode") != "Best performance":
        problems.append(f"Windows power mode is '{power.get('power_mode')}', not 'Best performance'")

    for problem in problems:
        print(f"{'ERROR' if strict else 'WARNING'}: {problem}")
    return problems, power


def jvm_flags_when_pinned(cpu):
    """Flags the JVM selects when pinned like the benchmark (CPU count changes GC ergonomics)."""
    proc = launch(["java", "-Xmx4g", "-XX:+PrintCommandLineFlags", "-version"], cpu)
    stdout, stderr = proc.communicate()
    return (stdout + stderr).strip().splitlines()[0]


# ------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Run the Assignment 1 benchmarks.")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    parser.add_argument("--reps", type=int, help="override the number of measured repetitions")
    parser.add_argument("--dry-run", action="store_true",
                        help="write to results/dryrun/ and only warn on failed pre-flight checks")
    parser.add_argument("--force", action="store_true", help="run even if pre-flight checks fail")
    args = parser.parse_args()
    sizes = sorted(args.sizes)
    if args.reps is not None and not args.dry_run:
        parser.error("--reps changes the declared protocol and is only allowed with --dry-run")
    if args.reps is not None and args.reps < 1:
        parser.error("--reps must be >= 1")

    with open(DATA_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    strict = not (args.dry_run or args.force)
    problems, power = preflight(sizes, args.languages, manifest, strict)
    if problems and strict:
        sys.exit("Pre-flight checks failed (use --dry-run to test, or --force to override).")

    # Pin to one P-core; if detection fails, run unpinned and record it.
    try:
        cpu = system_info.choose_benchmark_cpu(system_info.detect_cpu_sets())
        pinning = "single P-core logical CPU (GetSystemCpuSetInformation)"
    except (OSError, ValueError) as error:
        cpu, pinning = None, f"none: P-core detection failed ({error})"
    if cpu is not None:
        # Keep this orchestrator off the benchmark core (children are pinned explicitly).
        psutil.Process().cpu_affinity([c for c in range(psutil.cpu_count()) if c != cpu])

    toolchain = system_info.collect_toolchain()
    runtime = {
        "c": toolchain["c"]["version"],
        "java": toolchain["java"]["java_version"][1],
        "python": f"{platform.python_implementation()} {platform.python_version()}",
    }
    budget_bytes = int(psutil.virtual_memory().total * system_info.MEMORY_BUDGET_FRACTION)

    # Ascending n; rotate the language order for each n to spread thermal effects.
    schedule = []
    for index, n in enumerate(sizes):
        shift = index % len(LANGUAGES)
        for language in LANGUAGES[shift:] + LANGUAGES[:shift]:
            if language in args.languages:
                schedule.append((language, n))

    out_dir = DRY_RUN_DIR if args.dry_run else RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_{stamp}.csv"
    meta = {
        "csv": csv_path.name,
        "dry_run": args.dry_run,
        "forced": args.force,
        "preflight_problems": problems,
        "started_at": now_iso(),
        "sizes": sizes,
        "schedule": [f"{language}:{n}" for language, n in schedule],
        "repetitions": {f"{language}:{n}": repetitions(language, n, args.reps)
                        for language, n in schedule},
        "reps_override": args.reps,
        "line_timeout_s": LINE_TIMEOUT_S,
        "memory_budget_bytes": budget_bytes,
        "sleep_between_s": SLEEP_BETWEEN_S,
        "benchmark_cpu": cpu,
        "pinning": pinning,
        "priority": "HIGH_PRIORITY_CLASS",
        "power_at_start": power,
        "commands": COMMANDS,
        "jvm_flags_when_pinned": jvm_flags_when_pinned(cpu) if "java" in args.languages else None,
    }
    print(f"Benchmark CPU: {cpu} ({pinning})")
    print(f"Writing {csv_path}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, (language, n) in enumerate(schedule):
            if index > 0:
                time.sleep(SLEEP_BETWEEN_S)
            rows = run_process(language, n, manifest["sizes"][str(n)], cpu, budget_bytes, runtime,
                               args.reps)
            writer.writerows(rows)
            f.flush()  # keep every completed process on disk even if the run is interrupted

    meta["finished_at"] = now_iso()
    with open(csv_path.with_name(csv_path.stem + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
