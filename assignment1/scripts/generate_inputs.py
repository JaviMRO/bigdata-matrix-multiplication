"""Generate the shared input matrices used by all three implementations.

For every size n this script writes

    data/A_n{n}.bin   and   data/B_n{n}.bin

as raw little-endian float64 values in row-major order with no header
(exactly 8 * n * n bytes per file), and records everything needed to
reproduce and check them in data/manifest.json.

Values are uniform in [-1, 1). A fresh generator is created for every n:
A uses seed BASE_SEED and B uses seed BASE_SEED + 1.

Usage (from any folder):
    python assignment1/scripts/generate_inputs.py              # benchmark sizes
    python assignment1/scripts/generate_inputs.py --sizes 7 33 # extra sizes

Entries for sizes already in the manifest but not requested are kept.
"""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

BASE_SEED = 20261016
DEFAULT_SIZES = [64, 128, 256, 512, 1024, 2048]
VALUE_LOW = -1.0
VALUE_HIGH = 1.0

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def random_matrix(n, seed):
    """Return an n x n float64 matrix with values uniform in [VALUE_LOW, VALUE_HIGH)."""
    rng = np.random.default_rng(seed)  # PCG64 bit generator
    return rng.uniform(VALUE_LOW, VALUE_HIGH, size=(n, n))


def write_matrix(matrix, path):
    """Write the matrix as raw little-endian float64, row-major, no header."""
    # '<f8' = little-endian 8-byte float; ascontiguousarray guarantees C (row-major) order.
    np.ascontiguousarray(matrix, dtype="<f8").tofile(path)


def read_matrix(path, n):
    """Read a matrix written by write_matrix (used here only to check the file)."""
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def sha256_of(path):
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_size(n):
    """Create, write and check A and B for one size n; return its manifest entry."""
    seeds = {"A": BASE_SEED, "B": BASE_SEED + 1}
    matrices = {}
    files = {}

    for name, seed in seeds.items():
        matrix = random_matrix(n, seed)
        path = DATA_DIR / f"{name}_n{n}.bin"
        write_matrix(matrix, path)

        # Check the written file: exact size and exact round trip of every value.
        expected_bytes = 8 * n * n
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{path.name}: {path.stat().st_size} bytes, expected {expected_bytes}")
        if not np.array_equal(read_matrix(path, n), matrix):
            raise RuntimeError(f"{path.name}: values read back differ from values written")

        matrices[name] = matrix
        files[name] = {"file": path.name, "seed": seed, "sha256": sha256_of(path)}

    # Reference result computed once with NumPy (not timed, never used by the kernels).
    a, b = matrices["A"], matrices["B"]
    c_ref = np.matmul(a, b)
    reference = {
        "method": "numpy.matmul(A, B)",
        # math.fsum returns the correctly rounded sum of the float64 entries of C_ref.
        "checksum": math.fsum(c_ref.ravel()),
        "frobenius_norm": float(np.linalg.norm(c_ref, "fro")),
        # Sum of (|A| @ |B|): scales the forward-error bound used to judge checksums.
        "abs_product_sum": math.fsum(np.matmul(np.abs(a), np.abs(b)).ravel()),
    }

    return {
        "n": n,
        "shape": [n, n],
        "dtype": "float64",
        "file_bytes": 8 * n * n,
        "A": files["A"],
        "B": files["B"],
        "reference": reference,
    }


def load_manifest():
    """Return the existing manifest's per-size entries, or an empty dict."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f).get("sizes", {})
    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate shared input matrices.")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help=f"matrix sizes n (default: {DEFAULT_SIZES})")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sizes = load_manifest()

    for n in args.sizes:
        if n < 1:
            parser.error(f"n must be positive, got {n}")
        entry = generate_size(n)
        sizes[str(n)] = entry
        print(f"n={n:5d}  A sha256={entry['A']['sha256'][:12]}...  "
              f"B sha256={entry['B']['sha256'][:12]}...  "
              f"checksum={entry['reference']['checksum']!r}")

    manifest = {
        "description": "Shared inputs for Assignment 1 (dense matrix multiplication).",
        "file_format": "raw little-endian float64, row-major, no header, 8*n*n bytes",
        "generator": "numpy.random.default_rng(seed) (PCG64), Generator.uniform",
        "value_range": f"[{VALUE_LOW}, {VALUE_HIGH})",
        "base_seed": BASE_SEED,
        "seed_rule": "fresh generator per n; A uses base_seed, B uses base_seed + 1",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "sizes": dict(sorted(sizes.items(), key=lambda item: int(item[0]))),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
