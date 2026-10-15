/*
 * matmul.c - Basic dense matrix multiplication C = A * B (Assignment 1, C version).
 *
 * Usage:
 *   matmul_c.exe --n N --a PATH --b PATH --warmup W --reps R [--dump PATH]
 *
 * Reads A and B (raw little-endian float64, row-major, no header), runs W warm-up
 * and R measured repetitions of the textbook i-j-k triple loop in this process,
 * and prints one CSV line per repetition:
 *   language,n,phase,rep,time_ns,checksum
 * Only the multiplication is timed. --dump writes the final C as raw float64.
 *
 * Build: gcc -O2 -std=c17 -Wall -Wextra -o build/matmul_c.exe src/c/matmul.c
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L /* for clock_gettime on Linux/macOS */
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* The input files are little-endian and are read with a single fread. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "This program assumes a little-endian host."
#endif

typedef struct {
    long n;
    const char *a_path;
    const char *b_path;
    const char *dump_path; /* NULL when --dump is not given */
    long warmup;
    long reps;
} Options;

/* Monotonic time in nanoseconds. */
static long long now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER count, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    /* Split into whole seconds and remainder so the multiplication cannot overflow. */
    return (count.QuadPart / freq.QuadPart) * 1000000000LL
         + (count.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

/* The kernel: textbook i-j-k triple loop on row-major arrays (element (i, j) is at i*n + j). */
static void multiply(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/* Sum of all elements of C in row-major order (same order in all three languages). */
static double checksum(const double *c, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        total += c[i];
    }
    return total;
}

/* Shortest decimal text that reads back as exactly x (same digits as Python repr / Java). */
static void format_double(double x, char *text, size_t size) {
    for (int precision = 1; precision <= 17; precision++) {
        snprintf(text, size, "%.*g", precision, x);
        if (strtod(text, NULL) == x) {
            return;
        }
    }
}

/* Parse a whole decimal number >= min; returns 1 on success. */
static int parse_long(const char *text, long min, long *out) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < min) {
        return 0;
    }
    *out = value;
    return 1;
}

/* Parse "--name value" pairs; returns 1 if all required options are present and valid. */
static int parse_options(int argc, char **argv, Options *opt) {
    opt->n = 0;
    opt->a_path = NULL;
    opt->b_path = NULL;
    opt->dump_path = NULL;
    opt->warmup = -1;
    opt->reps = 0;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            return 0; /* every option needs a value */
        }
        const char *name = argv[i];
        const char *value = argv[i + 1];
        int ok = 1;
        if (strcmp(name, "--n") == 0) {
            ok = parse_long(value, 1, &opt->n);
        } else if (strcmp(name, "--a") == 0) {
            opt->a_path = value;
        } else if (strcmp(name, "--b") == 0) {
            opt->b_path = value;
        } else if (strcmp(name, "--warmup") == 0) {
            ok = parse_long(value, 0, &opt->warmup);
        } else if (strcmp(name, "--reps") == 0) {
            ok = parse_long(value, 1, &opt->reps);
        } else if (strcmp(name, "--dump") == 0) {
            opt->dump_path = value;
        } else {
            ok = 0; /* unknown option */
        }
        if (!ok) {
            return 0;
        }
    }
    return opt->n >= 1 && opt->a_path && opt->b_path && opt->warmup >= 0 && opt->reps >= 1;
}

/* Read an n x n matrix; the file must hold exactly n*n doubles. Returns NULL on error. */
static double *read_matrix(const char *path, size_t n) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return NULL;
    }
    double *m = malloc(n * n * sizeof(double));
    if (m == NULL) {
        fprintf(stderr, "error: out of memory reading %s\n", path);
        fclose(file);
        return NULL;
    }
    size_t count = fread(m, sizeof(double), n * n, file);
    int extra_bytes = fgetc(file) != EOF;
    fclose(file);
    if (count != n * n || extra_bytes) {
        fprintf(stderr, "error: %s does not contain exactly %zu doubles\n", path, n * n);
        free(m);
        return NULL;
    }
    return m;
}

/* Write C as raw little-endian float64. Returns 1 on success. */
static int write_matrix(const char *path, const double *m, size_t n) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "error: cannot create %s\n", path);
        return 0;
    }
    size_t count = fwrite(m, sizeof(double), n * n, file);
    int closed = fclose(file) == 0;
    if (count != n * n || !closed) {
        fprintf(stderr, "error: failed writing %s\n", path);
        return 0;
    }
    return 1;
}

/* One repetition: time only the multiplication, then print the result line. */
static void run_repetition(const char *phase, long rep,
                           const double *a, const double *b, double *c, size_t n) {
    long long start = now_ns();
    multiply(a, b, c, n);
    long long elapsed = now_ns() - start;

    char text[32];
    format_double(checksum(c, n), text, sizeof text);
    printf("c,%zu,%s,%ld,%lld,%s\n", n, phase, rep, elapsed, text);
    fflush(stdout); /* the orchestrator reads each line as soon as it is printed */
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_options(argc, argv, &opt)) {
        fprintf(stderr, "usage: %s --n N --a PATH --b PATH --warmup W --reps R [--dump PATH]\n",
                argv[0]);
        return 2;
    }
    size_t n = (size_t) opt.n;

    double *a = read_matrix(opt.a_path, n);
    double *b = a ? read_matrix(opt.b_path, n) : NULL;
    double *c = b ? malloc(n * n * sizeof(double)) : NULL;
    if (c == NULL) {
        if (b != NULL) {
            fprintf(stderr, "error: out of memory allocating C\n");
        }
        free(a);
        free(b);
        return 1;
    }
    /* Write every element once so page faults happen here, not inside the timed region. */
    for (size_t i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }

    for (long rep = 1; rep <= opt.warmup; rep++) {
        run_repetition("warmup", rep, a, b, c, n);
    }
    for (long rep = 1; rep <= opt.reps; rep++) {
        run_repetition("measured", rep, a, b, c, n);
    }

    int status = 0;
    if (opt.dump_path != NULL && !write_matrix(opt.dump_path, c, n)) {
        status = 1;
    }
    free(a);
    free(b);
    free(c);
    return status;
}
