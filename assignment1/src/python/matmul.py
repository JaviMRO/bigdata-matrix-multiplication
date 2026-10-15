"""Basic dense matrix multiplication C = A * B (Assignment 1, Python version).

Usage:
    python src/python/matmul.py --n N --a PATH --b PATH --warmup W --reps R [--dump PATH]

Reads A and B (raw little-endian float64, row-major, no header), runs W warm-up
and R measured repetitions of the textbook i-j-k triple loop in this process,
and prints one CSV line per repetition:
    language,n,phase,rep,time_ns,checksum
Only the multiplication is timed. --dump writes the final C as raw float64.

The kernel is pure Python: matrices are lists of lists of float, no NumPy.
"""

import argparse
import sys
import time
from array import array


def multiply(a, b, c, n):
    """The kernel: textbook i-j-k triple loop; a, b, c are lists of n rows of n floats."""
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s


def checksum(c):
    """Sum of all elements of C in row-major order (same order in all three languages).

    An explicit loop is used on purpose: since Python 3.12 the built-in sum() of floats
    uses compensated summation, which would differ from the plain sum in C and Java.
    """
    total = 0.0
    for row in c:
        for x in row:
            total += x
    return total


def read_matrix(path, n):
    """Read an n x n matrix; the file must hold exactly n*n little-endian doubles."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != 8 * n * n:
        raise ValueError(f"{path} does not contain exactly {n * n} doubles")
    values = array("d")  # C double = float64
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return [values[i * n:(i + 1) * n].tolist() for i in range(n)]


def write_matrix(path, c):
    """Write C as raw little-endian float64."""
    values = array("d", (x for row in c for x in row))
    if sys.byteorder != "little":
        values.byteswap()
    with open(path, "wb") as f:
        values.tofile(f)


def run_repetition(phase, rep, a, b, c, n):
    """One repetition: time only the multiplication, then print the result line."""
    start = time.perf_counter_ns()
    multiply(a, b, c, n)
    elapsed = time.perf_counter_ns() - start

    # repr gives the shortest text that reads back as exactly the same float.
    # flush: the orchestrator reads each line as soon as it is printed.
    print(f"python,{n},{phase},{rep},{elapsed},{checksum(c)!r}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Basic i-j-k matrix multiplication (pure Python).")
    parser.add_argument("--n", type=int, required=True, help="matrix size")
    parser.add_argument("--a", required=True, help="path of matrix A (raw float64)")
    parser.add_argument("--b", required=True, help="path of matrix B (raw float64)")
    parser.add_argument("--warmup", type=int, required=True, help="number of warm-up repetitions")
    parser.add_argument("--reps", type=int, required=True, help="number of measured repetitions")
    parser.add_argument("--dump", help="write the final C to this path (raw float64)")
    args = parser.parse_args()
    if args.n < 1 or args.warmup < 0 or args.reps < 1:
        parser.error("need n >= 1, warmup >= 0 and reps >= 1")
    return args


def main():
    args = parse_args()  # exits with code 2 on invalid arguments
    n = args.n
    try:
        a = read_matrix(args.a, n)
        b = read_matrix(args.b, n)
        c = [[0.0] * n for _ in range(n)]  # allocated once, before any timing

        for rep in range(1, args.warmup + 1):
            run_repetition("warmup", rep, a, b, c, n)
        for rep in range(1, args.reps + 1):
            run_repetition("measured", rep, a, b, c, n)
        if args.dump:
            write_matrix(args.dump, c)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
