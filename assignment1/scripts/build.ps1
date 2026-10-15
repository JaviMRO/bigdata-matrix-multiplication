# Compile the C and Java implementations of Assignment 1 into assignment1/build/.
#
# Usage (from any folder):
#   powershell -ExecutionPolicy Bypass -File assignment1\scripts\build.ps1
#
# Flags are fixed by the benchmark protocol: C uses -O2 only
# (no -march=native, no -ffast-math). Java is compiled with default javac settings.
# A missing source file is reported and skipped; the script then exits with code 1
# so that a partial build is never mistaken for a complete one.

$root = Split-Path -Parent $PSScriptRoot   # the assignment1/ folder
Push-Location $root
$exitCode = 0

try {
    New-Item -ItemType Directory -Force build | Out-Null

    # ---- C ----
    $cSource = "src/c/matmul.c"
    if (Test-Path $cSource) {
        gcc -O2 -std=c17 -Wall -Wextra -o build/matmul_c.exe $cSource
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: gcc failed (exit code $LASTEXITCODE)"
            $exitCode = 1
        } else {
            Write-Host "OK: build/matmul_c.exe"
        }
    } else {
        Write-Host "SKIPPED: $cSource not found"
        $exitCode = 1
    }

    # ---- Java ----
    $javaSource = "src/java/MatMul.java"
    if (Test-Path $javaSource) {
        javac -d build $javaSource
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: javac failed (exit code $LASTEXITCODE)"
            $exitCode = 1
        } else {
            Write-Host "OK: build/MatMul.class"
        }
    } else {
        Write-Host "SKIPPED: $javaSource not found"
        $exitCode = 1
    }
} finally {
    Pop-Location
}

exit $exitCode
