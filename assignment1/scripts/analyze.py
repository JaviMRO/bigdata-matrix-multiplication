"""Tables and figures from the raw benchmark CSVs.

Input: one or more results/raw/benchmark_<timestamp>.csv files written by run_benchmarks.py
(by default the most recent one). Only rows with phase == "measured" from processes that
finished normally enter the statistics; warm-ups and failed processes stay in the raw CSV
and failed processes are reported in the limits table.

Outputs (results/tables/):
  summary.csv / summary.tex        median, Q1, Q3, IQR, min, max, count per (language, n),
                                   plus time per inner-loop iteration and peak working set
  fit.csv / fit.tex                log-log slope b of log(t) = a + b*log(n) for n >= 256, 95 % CI
  local_slopes.csv                 slope between consecutive sizes, log2(t(2n) / t(n))
  ratios.csv                       median-time ratios Python/C and Java/C per n
  limits.csv / limits.tex          largest feasible n vs largest attempted n, with failures

Outputs (figures/, PNG + PDF at 300 dpi):
  fig1_time_vs_n                   median time vs n (log-log), IQR bars, n^3 reference
  fig2_time_per_iteration          t / n^3 (ns) vs n
  fig3_peak_memory                 peak working set (MB) vs n, plus theoretical 24 n^2 bytes
  fig4_ratios                      runtime ratios Python/C and Java/C vs n

Quartiles use linear interpolation between order statistics (pandas/NumPy default).

Usage (from any folder):
    python assignment1/scripts/analyze.py                       # latest raw CSV
    python assignment1/scripts/analyze.py path/to/a.csv b.csv   # explicit files (pooled)
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ASSIGNMENT_DIR / "results" / "raw"
TABLES_DIR = ASSIGNMENT_DIR / "results" / "tables"
FIGURES_DIR = ASSIGNMENT_DIR / "figures"

LANGUAGES = ["c", "java", "python"]
LABELS = {"c": "C (gcc -O2)", "java": "Java (HotSpot JIT)", "python": "Python (CPython)"}
# First three slots of a colour-blind-validated categorical palette. Marker shapes add a
# second encoding so the figures also work in grayscale print.
COLORS = {"c": "#2a78d6", "java": "#eb6834", "python": "#1baf7a"}
MARKERS = {"c": "o", "java": "s", "python": "^"}
REFERENCE_COLOR = "#52514e"  # neutral gray for reference lines

FIT_MIN_N = 256  # slopes are fitted on n >= 256 (small n is dominated by fixed costs)
BYTES_PER_MB = 1e6
NS_PER_MS = 1e6


# ------------------------------------------------------------------ loading

def load_raw(paths):
    """Concatenate the raw CSVs and add a 'cell_ok' column per (language, n) process."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = Path(path).name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    # checksum_ok is "True"/"False", or empty for failed rows.
    raw["checksum_ok"] = raw["checksum_ok"].astype(str) == "True"

    # A (source, language, n) process is usable only if every one of its rows is "ok".
    # A process killed half-way may have some ok rows; they are kept out of the statistics.
    status_ok = raw["status"] == "ok"
    raw["cell_ok"] = status_ok.groupby([raw["source"], raw["language"], raw["n"]]).transform("all")

    bad = raw[raw["cell_ok"] & (raw["phase"] == "measured") & ~raw["checksum_ok"]]
    if not bad.empty:
        # A wrong result must never be analysed as a valid timing.
        print("ERROR: measured rows with checksum_ok == False:", file=sys.stderr)
        print(bad[["source", "language", "n", "rep", "checksum", "checksum_ref"]], file=sys.stderr)
        sys.exit(1)
    return raw


def measured_rows(raw):
    """Measured repetitions of processes that completed normally."""
    return raw[(raw["phase"] == "measured") & raw["cell_ok"]].copy()


# ------------------------------------------------------------------ statistics

def summarize(measured):
    """Descriptive statistics of time_ns per (language, n)."""
    rows = []
    for (language, n), group in measured.groupby(["language", "n"]):
        t = group["time_ns"]
        q1, median, q3 = t.quantile([0.25, 0.5, 0.75])
        rows.append({
            "language": language,
            "n": n,
            "count": len(t),
            "median_ns": median,
            "q1_ns": q1,
            "q3_ns": q3,
            "iqr_ns": q3 - q1,
            "min_ns": t.min(),
            "max_ns": t.max(),
            # Time of one multiply-add of the inner loop (n^3 of them per multiplication).
            "median_ns_per_iteration": median / n**3,
            # One value per process; max() only matters if several sessions are pooled.
            "peak_working_set_bytes": group["peak_working_set_bytes"].max(),
            "theoretical_bytes": group["theoretical_bytes"].iloc[0],
        })
    summary = pd.DataFrame(rows)
    summary["language"] = pd.Categorical(summary["language"], LANGUAGES, ordered=True)
    return summary.sort_values(["language", "n"]).reset_index(drop=True)


def fit_slopes(measured, summary):
    """Least-squares fit of log(t) = a + b*log(n) per language on n >= FIT_MIN_N.

    Every measured repetition is one point, so the 95 % confidence interval for b uses
    the t distribution with (points - 2) degrees of freedom. Repetitions of one process
    are not fully independent, so the interval reflects within-run noise and lack of fit,
    not run-to-run variation. The slope through the medians is given as a cross-check.
    """
    rows = []
    for language in LANGUAGES:
        group = measured[(measured["language"] == language) & (measured["n"] >= FIT_MIN_N)]
        sizes = sorted(group["n"].unique())
        if len(sizes) < 2:
            continue
        fit = stats.linregress(np.log(group["n"]), np.log(group["time_ns"]))
        points = len(group)
        t_crit = stats.t.ppf(0.975, points - 2)
        medians = summary[(summary["language"] == language) & (summary["n"] >= FIT_MIN_N)]
        slope_of_medians = np.polyfit(np.log(medians["n"]), np.log(medians["median_ns"]), 1)[0]
        rows.append({
            "language": language,
            "sizes": " ".join(str(n) for n in sizes),
            "points": points,
            "slope": fit.slope,
            "slope_stderr": fit.stderr,
            "ci95_low": fit.slope - t_crit * fit.stderr,
            "ci95_high": fit.slope + t_crit * fit.stderr,
            "r_squared": fit.rvalue**2,
            "slope_of_medians": slope_of_medians,
        })
    return pd.DataFrame(rows)


def local_slopes(summary):
    """Slope between consecutive sizes: log(t2/t1) / log(n2/n1)."""
    rows = []
    for language in LANGUAGES:
        lang = summary[summary["language"] == language].sort_values("n")
        for (_, a), (_, b) in zip(lang.iterrows(), lang.iloc[1:].iterrows()):
            rows.append({
                "language": language,
                "n_from": a["n"],
                "n_to": b["n"],
                "slope": math.log(b["median_ns"] / a["median_ns"]) / math.log(b["n"] / a["n"]),
            })
    return pd.DataFrame(rows)


def ratios(summary):
    """Median-time ratios relative to C, for every n measured in both languages."""
    medians = summary.pivot(index="n", columns="language", values="median_ns")
    result = pd.DataFrame(index=medians.index)
    for language in ["python", "java"]:
        if language in medians and "c" in medians:
            result[f"{language}_over_c"] = medians[language] / medians["c"]
    return result.reset_index()


def limits(raw):
    """Largest attempted vs largest feasible n per language, and the failures."""
    rows = []
    for language in LANGUAGES:
        lang = raw[raw["language"] == language]
        if lang.empty:
            continue
        cells = lang.groupby("n")
        feasible = [n for n, group in cells if group["cell_ok"].all()]
        failures = []
        for n, group in cells:
            failed = group[group["status"] != "ok"]
            if not failed.empty:
                peak_mb = failed["peak_working_set_bytes"].max() / BYTES_PER_MB
                failures.append(f"n={n}: {failed['status'].iloc[0]} "
                                f"(peak WS {peak_mb:.0f} MB when stopped)")
        rows.append({
            "language": language,
            "largest_attempted_n": lang["n"].max(),
            "largest_feasible_n": max(feasible) if feasible else None,
            "failures": "; ".join(failures) if failures else "none",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ LaTeX

def sig(x, digits=3):
    """Format x with `digits` significant figures and no exponent (e.g. 0.0539, 62700)."""
    if x == 0 or not math.isfinite(x):
        return str(x)
    rounded = float(f"{x:.{digits}g}")  # round to significant figures first
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(rounded))))
    return f"{rounded:.{decimals}f}"


def write_latex(path, header, rows, caption, label, align, group_breaks=()):
    """Write a booktabs table. rows are lists of already-formatted strings."""
    lines = [
        "% Generated by scripts/analyze.py - do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    for i, row in enumerate(rows):
        if i in group_breaks:
            lines.append("    \\midrule")
        lines.append("    " + " & ".join(row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_latex(summary, path):
    header = ["Language", "$n$", "Reps", "Median", "Q1", "Q3", "IQR", "Min", "Max",
              "ns/iter", "Peak WS (MB)"]
    rows, breaks = [], []
    previous = None
    for _, r in summary.iterrows():
        first_of_group = r["language"] != previous
        if previous is not None and first_of_group:
            breaks.append(len(rows))
        previous = r["language"]
        rows.append([
            LABELS[r["language"]] if first_of_group else "", str(r["n"]), str(r["count"]),
            *(sig(r[c] / NS_PER_MS) for c in ["median_ns", "q1_ns", "q3_ns", "iqr_ns",
                                              "min_ns", "max_ns"]),
            sig(r["median_ns_per_iteration"]),
            sig(r["peak_working_set_bytes"] / BYTES_PER_MB),
        ])
    write_latex(path, header, rows,
                "Kernel time per multiplication in milliseconds (measured repetitions only), "
                "median time per inner-loop iteration ($t/n^3$) and peak working set of the process.",
                "tab:summary", "lrrrrrrrrrr", breaks)


def fit_latex(fit, path):
    header = ["Language", "Sizes", "Points", "Slope $b$", "95\\,\\% CI", "$R^2$", "Slope of medians"]
    rows = [[LABELS[r["language"]], r["sizes"].replace(" ", ", "), str(r["points"]),
             f"{r['slope']:.3f}", f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]",
             f"{r['r_squared']:.4f}", f"{r['slope_of_medians']:.3f}"]
            for _, r in fit.iterrows()]
    write_latex(path, header, rows,
                f"Least-squares fit of $\\log t = a + b \\log n$ on every measured repetition "
                f"with $n \\ge {FIT_MIN_N}$.", "tab:fit", "llrrcrr")


def limits_latex(lim, path):
    header = ["Language", "Largest attempted $n$", "Largest feasible $n$", "Failures"]
    rows = [[LABELS[r["language"]], str(r["largest_attempted_n"]),
             str(r["largest_feasible_n"]),
             r["failures"].replace("n=", "$n=").replace(":", "$:")]
            for _, r in lim.iterrows()]
    write_latex(path, header, rows,
                "Feasibility under the declared budgets (600\\,s per multiplication, "
                "peak working set $\\le$ 50\\,\\% of RAM).", "tab:limits", "lrrl")


# ------------------------------------------------------------------ figures

def style_axes(ax, sizes):
    """Log2 x axis with a tick at every benchmark size, recessive grid, no top/right spines."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Matrix size $n$")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_series(ax, x, y, language, yerr=None):
    ax.errorbar(x, y, yerr=yerr, label=LABELS[language], color=COLORS[language],
                marker=MARKERS[language], markersize=6, linewidth=1.5, capsize=3)


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def iqr_errors(lang, scale=1.0):
    """Asymmetric error bars from the median to Q1 and Q3."""
    return [(lang["median_ns"] - lang["q1_ns"]) / scale, (lang["q3_ns"] - lang["median_ns"]) / scale]


def fig_time(summary, lim, sizes):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGES:
        lang = summary[summary["language"] == language]
        plot_series(ax, lang["n"], lang["median_ns"] / 1e9, language, iqr_errors(lang, 1e9))
    # n^3 reference through the C median at the smallest size.
    c = summary[summary["language"] == "c"]
    if not c.empty:
        n0, t0 = c["n"].iloc[0], c["median_ns"].iloc[0] / 1e9
        x = np.array(sizes, dtype=float)
        ax.plot(x, t0 * (x / n0) ** 3, linestyle="--", color=REFERENCE_COLOR, linewidth=1,
                label=f"$\\propto n^3$ (through C at $n={n0}$)")
    failures = lim[lim["failures"] != "none"]
    if not failures.empty:
        note = "\n".join(f"{LABELS[r['language']]}: {r['failures'].split(' (')[0]}"
                         for _, r in failures.iterrows())
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=REFERENCE_COLOR)
    ax.set_yscale("log")
    style_axes(ax, sizes)
    ax.set_ylabel("Median kernel time per multiplication (s)")
    ax.set_title("Runtime vs matrix size (error bars: IQR)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig1_time_vs_n")


def fig_per_iteration(summary, sizes):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGES:
        lang = summary[summary["language"] == language]
        cube = lang["n"].astype(float) ** 3
        yerr = [(lang["median_ns"] - lang["q1_ns"]) / cube, (lang["q3_ns"] - lang["median_ns"]) / cube]
        plot_series(ax, lang["n"], lang["median_ns_per_iteration"], language, yerr)
    ax.set_yscale("log")
    style_axes(ax, sizes)
    ax.set_ylabel("Median time per inner-loop iteration, $t/n^3$ (ns)")
    ax.set_title("Normalised time per multiply-add")
    ax.legend(frameon=False)
    save(fig, "fig2_time_per_iteration")


def fig_memory(summary, sizes):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for language in LANGUAGES:
        lang = summary[summary["language"] == language]
        plot_series(ax, lang["n"], lang["peak_working_set_bytes"] / BYTES_PER_MB, language)
    x = np.array(sizes, dtype=float)
    ax.plot(x, 24 * x**2 / BYTES_PER_MB, linestyle="--", color=REFERENCE_COLOR, linewidth=1,
            label="Theoretical storage $3 \\cdot 8 n^2$ bytes")
    ax.set_yscale("log")
    style_axes(ax, sizes)
    ax.set_ylabel("Peak working set (MB, $10^6$ bytes)")
    ax.set_title("Peak process memory vs matrix size")
    ax.legend(frameon=False)
    save(fig, "fig3_peak_memory")


def fig_ratios(ratio, sizes):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for language in ["python", "java"]:
        column = f"{language}_over_c"
        if column in ratio:
            valid = ratio.dropna(subset=[column])
            ax.plot(valid["n"], valid[column], color=COLORS[language], marker=MARKERS[language],
                    markersize=6, linewidth=1.5, label=f"{LABELS[language]} / C")
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Equal to C")
    ax.set_yscale("log")
    style_axes(ax, sizes)
    ax.set_ylabel("Ratio of median kernel times (relative to C)")
    ax.set_title("Runtime relative to C")
    ax.legend(frameon=False)
    save(fig, "fig4_ratios")


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="*", type=Path,
                        help="raw benchmark CSVs (default: the latest in results/raw/)")
    args = parser.parse_args()

    paths = args.csv or sorted(RAW_DIR.glob("benchmark_*.csv"))[-1:]
    if not paths:
        sys.exit(f"No benchmark CSV found in {RAW_DIR}")
    print("Analysing:", ", ".join(str(p) for p in paths))

    raw = load_raw(paths)
    measured = measured_rows(raw)
    summary = summarize(measured)
    fit = fit_slopes(measured, summary)
    local = local_slopes(summary)
    ratio = ratios(summary)
    lim = limits(raw)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "summary.csv", index=False)
    fit.to_csv(TABLES_DIR / "fit.csv", index=False)
    local.to_csv(TABLES_DIR / "local_slopes.csv", index=False)
    ratio.to_csv(TABLES_DIR / "ratios.csv", index=False)
    lim.to_csv(TABLES_DIR / "limits.csv", index=False)
    summary_latex(summary, TABLES_DIR / "summary.tex")
    fit_latex(fit, TABLES_DIR / "fit.tex")
    limits_latex(lim, TABLES_DIR / "limits.tex")

    sizes = sorted(raw["n"].unique())
    fig_time(summary, lim, sizes)
    fig_per_iteration(summary, sizes)
    fig_memory(summary, sizes)
    fig_ratios(ratio, sizes)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nMedian time (ms):")
        print((summary.pivot(index="n", columns="language", values="median_ns") / NS_PER_MS).round(3))
        print("\nLog-log slopes (n >=", FIT_MIN_N, "):")
        print(fit.round(4).to_string(index=False))
        print("\nLimits:")
        print(lim.to_string(index=False))
    print(f"\nTables written to {TABLES_DIR}\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
