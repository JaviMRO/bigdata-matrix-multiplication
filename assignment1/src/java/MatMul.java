/*
 * MatMul.java - Basic dense matrix multiplication C = A * B (Assignment 1, Java version).
 *
 * Usage:
 *   java -Xmx4g -cp build MatMul --n N --a PATH --b PATH --warmup W --reps R [--dump PATH]
 *
 * Reads A and B (raw little-endian float64, row-major, no header), runs W warm-up
 * and R measured repetitions of the textbook i-j-k triple loop in this JVM,
 * and prints one CSV line per repetition:
 *   language,n,phase,rep,time_ns,checksum
 * Only the multiplication is timed. --dump writes the final C as raw float64.
 *
 * Build: javac -d build src/java/MatMul.java
 */

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

public class MatMul {

    /** The kernel: textbook i-j-k triple loop on row-major arrays (element (i, j) is at i*n + j). */
    static void multiply(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /** Sum of all elements of C in row-major order (same order in all three languages). */
    static double checksum(double[] c) {
        double total = 0.0;
        for (int i = 0; i < c.length; i++) {
            total += c[i];
        }
        return total;
    }

    /** Read an n x n matrix; the file must hold exactly n*n little-endian doubles. */
    static double[] readMatrix(String path, int n) throws IOException {
        byte[] bytes = Files.readAllBytes(Path.of(path));
        if (bytes.length != 8L * n * n) {
            throw new IOException(path + " does not contain exactly " + (long) n * n + " doubles");
        }
        double[] m = new double[n * n];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(m);
        return m;
    }

    /** Write C as raw little-endian float64. */
    static void writeMatrix(String path, double[] m) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(8 * m.length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asDoubleBuffer().put(m);
        Files.write(Path.of(path), buffer.array());
    }

    /** One repetition: time only the multiplication, then print the result line. */
    static void runRepetition(String phase, int rep, double[] a, double[] b, double[] c, int n) {
        long start = System.nanoTime();
        multiply(a, b, c, n);
        long elapsed = System.nanoTime() - start;

        // Double.toString gives the shortest text that reads back as exactly the same double.
        System.out.println("java," + n + "," + phase + "," + rep + "," + elapsed + ","
                + Double.toString(checksum(c)));
        System.out.flush(); // the orchestrator reads each line as soon as it is printed
    }

    /** Parse a whole number >= min, or throw IllegalArgumentException. */
    static int parseInt(String name, String value, int min) {
        int result;
        try {
            result = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be an integer, got " + value);
        }
        if (result < min) {
            throw new IllegalArgumentException(name + " must be >= " + min + ", got " + value);
        }
        return result;
    }

    public static void main(String[] args) {
        int n = 0;
        int warmup = -1;
        int reps = 0;
        String aPath = null;
        String bPath = null;
        String dumpPath = null;

        // Parse "--name value" pairs.
        try {
            for (int i = 0; i < args.length; i += 2) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException(args[i] + " needs a value");
                }
                String name = args[i];
                String value = args[i + 1];
                switch (name) {
                    case "--n" -> n = parseInt(name, value, 1);
                    case "--a" -> aPath = value;
                    case "--b" -> bPath = value;
                    case "--warmup" -> warmup = parseInt(name, value, 0);
                    case "--reps" -> reps = parseInt(name, value, 1);
                    case "--dump" -> dumpPath = value;
                    default -> throw new IllegalArgumentException("unknown option " + name);
                }
            }
            if (n < 1 || aPath == null || bPath == null || warmup < 0 || reps < 1) {
                throw new IllegalArgumentException("missing required option");
            }
        } catch (IllegalArgumentException e) {
            System.err.println("error: " + e.getMessage());
            System.err.println("usage: java MatMul --n N --a PATH --b PATH --warmup W --reps R [--dump PATH]");
            System.exit(2);
        }

        try {
            double[] a = readMatrix(aPath, n);
            double[] b = readMatrix(bPath, n);
            double[] c = new double[n * n]; // allocated (and zeroed) once, before any timing

            for (int rep = 1; rep <= warmup; rep++) {
                runRepetition("warmup", rep, a, b, c, n);
            }
            for (int rep = 1; rep <= reps; rep++) {
                runRepetition("measured", rep, a, b, c, n);
            }
            if (dumpPath != null) {
                writeMatrix(dumpPath, c);
            }
        } catch (IOException e) {
            System.err.println("error: " + e.getMessage());
            System.exit(1);
        }
    }
}
