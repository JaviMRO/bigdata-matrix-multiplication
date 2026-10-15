"""Collect a description of the machine and toolchain used for the benchmarks.

Writes results/system_info.json. Usage (from any folder):
    python assignment1/scripts/system_info.py

Windows only: hardware details are read through the Win32 API with ctypes.
detect_cpu_sets() and choose_benchmark_cpu() are also meant to be imported by
run_benchmarks.py, so both scripts pin to the same P-core.
"""

import ctypes
import datetime
import importlib.metadata
import json
import platform
import shutil
import struct
import subprocess
import sys
import uuid
import winreg
from ctypes import wintypes
from pathlib import Path

import psutil

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ASSIGNMENT_DIR / "results" / "system_info.json"

# Fraction of physical RAM a benchmark process may use (benchmark protocol).
MEMORY_BUDGET_FRACTION = 0.5

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


# --------------------------------------------------------------------------- CPU

def detect_cpu_sets():
    """Return one dict per logical CPU, using GetSystemCpuSetInformation.

    EfficiencyClass is 0 for the most power-efficient cores; on a hybrid Intel CPU
    the P-cores have the highest value.
    """
    func = kernel32.GetSystemCpuSetInformation
    func.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                     wintypes.HANDLE, wintypes.ULONG]
    func.restype = wintypes.BOOL

    # First call with no buffer only reports the number of bytes needed.
    needed = wintypes.ULONG(0)
    func(None, 0, ctypes.byref(needed), None, 0)
    buffer = ctypes.create_string_buffer(needed.value)
    if not func(buffer, needed, ctypes.byref(needed), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())

    # The buffer holds variable-size SYSTEM_CPU_SET_INFORMATION records:
    #   DWORD Size; DWORD Type; then (Type 0) DWORD Id; WORD Group;
    #   BYTE LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex, NumaNodeIndex,
    #        EfficiencyClass, AllFlags; ...
    raw = buffer.raw
    cpus = []
    offset = 0
    while offset < needed.value:
        size, info_type = struct.unpack_from("<II", raw, offset)
        if size == 0:
            break
        if info_type == 0:  # CpuSetInformation
            (cpu_set_id, group, logical_index, core_index, llc_index,
             numa_node, efficiency_class, flags) = struct.unpack_from("<IHBBBBBB", raw, offset + 8)
            cpus.append({
                "cpu_set_id": cpu_set_id,
                "group": group,
                "logical_processor_index": logical_index,
                "core_index": core_index,
                "last_level_cache_index": llc_index,
                "numa_node": numa_node,
                "efficiency_class": efficiency_class,
                "parked": bool(flags & 0x1),
            })
        offset += size
    return cpus


def choose_benchmark_cpu(cpus):
    """Pick the logical CPU to pin the benchmark process to.

    Among the cores with the highest EfficiencyClass (P-cores) in processor group 0,
    take the one with the highest logical index: CPU 0 usually services more
    interrupts and background work than the others.
    """
    group0 = [c for c in cpus if c["group"] == 0]
    best_class = max(c["efficiency_class"] for c in group0)
    p_cores = [c for c in group0 if c["efficiency_class"] == best_class]
    return max(c["logical_processor_index"] for c in p_cores)


def detect_caches():
    """Return the CPU caches reported by GetLogicalProcessorInformation."""
    func = kernel32.GetLogicalProcessorInformation
    func.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    func.restype = wintypes.BOOL

    needed = wintypes.DWORD(0)
    func(None, ctypes.byref(needed))
    buffer = ctypes.create_string_buffer(needed.value)
    if not func(buffer, ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())

    # On 64-bit Windows each SYSTEM_LOGICAL_PROCESSOR_INFORMATION record is 32 bytes:
    #   ULONG_PTR ProcessorMask; DWORD Relationship; (padding); 16-byte union.
    # For Relationship == 2 (RelationCache) the union is a CACHE_DESCRIPTOR:
    #   BYTE Level; BYTE Associativity; WORD LineSize; DWORD Size; DWORD Type.
    record_size = 32
    cache_types = {0: "unified", 1: "instruction", 2: "data", 3: "trace"}
    raw = buffer.raw
    caches = []
    for offset in range(0, needed.value, record_size):
        mask, relationship = struct.unpack_from("<QI", raw, offset)
        if relationship != 2:
            continue
        level, associativity, line_size, size, cache_type = struct.unpack_from("<BBHII", raw, offset + 16)
        caches.append({
            "level": level,
            "type": cache_types.get(cache_type, str(cache_type)),
            "size_bytes": size,
            "line_size_bytes": line_size,
            "associativity": associativity,
            "logical_cpus": [i for i in range(64) if mask >> i & 1],
        })
    return caches


def collect_cpu():
    """CPU model, core counts, P/E-core layout, caches and the chosen benchmark CPU."""
    key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
        model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

    freq = psutil.cpu_freq()
    info = {
        "model": model,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        # On Windows psutil reports the base frequency as "max", not the turbo frequency.
        "reported_base_frequency_mhz": freq.max if freq else None,
    }

    try:
        cpus = detect_cpu_sets()
        best_class = max(c["efficiency_class"] for c in cpus)
        p_logical = sorted(c["logical_processor_index"] for c in cpus if c["efficiency_class"] == best_class)
        e_logical = sorted(c["logical_processor_index"] for c in cpus if c["efficiency_class"] != best_class)
        info["cpu_sets"] = cpus
        info["hybrid"] = len(e_logical) > 0
        info["p_core_logical_cpus"] = p_logical
        info["e_core_logical_cpus"] = e_logical
        info["p_cores"] = len({c["core_index"] for c in cpus if c["efficiency_class"] == best_class})
        info["e_cores"] = len({c["core_index"] for c in cpus if c["efficiency_class"] != best_class})
        info["benchmark_cpu"] = choose_benchmark_cpu(cpus)
        info["pinning"] = "single P-core logical CPU (GetSystemCpuSetInformation)"
    except (OSError, ValueError) as error:
        info["benchmark_cpu"] = None
        info["pinning"] = f"none: P-core detection failed ({error})"

    try:
        caches = detect_caches()
        info["caches"] = caches
        if info.get("benchmark_cpu") is not None:
            # The caches that the pinned benchmark process will actually use.
            info["caches_of_benchmark_cpu"] = [
                {k: c[k] for k in ("level", "type", "size_bytes", "line_size_bytes")}
                | {"shared_by_logical_cpus": len(c["logical_cpus"])}
                for c in caches if info["benchmark_cpu"] in c["logical_cpus"]
            ]
    except OSError as error:
        info["caches"] = f"unavailable ({error})"
    return info


# ------------------------------------------------------------------ OS and power

def collect_os():
    """Windows edition, version and exact build number."""
    key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    values = {}
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
        for name in ("EditionID", "DisplayVersion", "CurrentBuild", "UBR"):
            try:
                values[name] = winreg.QueryValueEx(key, name)[0]
            except FileNotFoundError:
                values[name] = None
    build = int(values["CurrentBuild"])
    return {
        # The registry ProductName still says "Windows 10" on Windows 11; builds >= 22000 are Windows 11.
        "name": "Windows 11" if build >= 22000 else "Windows 10",
        "edition": values["EditionID"],
        "version": values["DisplayVersion"],
        "build": f"{values['CurrentBuild']}.{values['UBR']}",
        "platform": platform.platform(),
    }


# Power-mode overlays shown in Settings > System > Power ("Power mode").
POWER_MODE_OVERLAYS = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


def collect_power():
    """AC/battery state, active power plan and Windows power mode."""
    info = {}

    # SYSTEM_POWER_STATUS: BYTE ACLineStatus, BatteryFlag, BatteryLifePercent, ...
    status = ctypes.create_string_buffer(12)
    if kernel32.GetSystemPowerStatus(status):
        ac_line, _, battery_percent = struct.unpack_from("<BBB", status.raw)
        info["ac_power"] = {0: False, 1: True}.get(ac_line)  # None = unknown
        info["battery_percent"] = None if battery_percent == 255 else battery_percent

    try:
        guid = ctypes.create_string_buffer(16)
        powrprof = ctypes.WinDLL("powrprof")
        if powrprof.PowerGetEffectiveOverlayScheme(guid) == 0:
            overlay = str(uuid.UUID(bytes_le=guid.raw))
            info["power_mode"] = POWER_MODE_OVERLAYS.get(overlay, "unknown")
            info["power_mode_guid"] = overlay
    except (OSError, AttributeError) as error:
        info["power_mode"] = f"unavailable ({error})"

    info["active_power_plan"] = run_command(["powercfg", "/getactivescheme"])
    return info


# --------------------------------------------------------------------- toolchain

def run_command(command):
    """Run a command and return its combined output (Java prints versions to stderr)."""
    if shutil.which(command[0]) is None:
        return f"not found: {command[0]}"
    result = subprocess.run(command, capture_output=True, encoding="oem", errors="replace")
    return (result.stdout + result.stderr).strip()


def collect_toolchain():
    """Versions and locations of gcc, Java and Python."""
    packages = {}
    for name in ("numpy", "pandas", "matplotlib", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "c": {
            "path": shutil.which("gcc"),
            "version": run_command(["gcc", "--version"]).splitlines()[0],
        },
        "java": {
            "java_path": shutil.which("java"),
            "java_version": run_command(["java", "-version"]).splitlines(),
            "javac_version": run_command(["javac", "-version"]),
            # The flags HotSpot selects for the benchmark settings (shows the default GC).
            "jvm_flags_with_Xmx4g": run_command(["java", "-Xmx4g", "-XX:+PrintCommandLineFlags",
                                                 "-version"]).splitlines()[0],
        },
        "python": {
            "path": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "build": sys.version,
            "packages": packages,
        },
    }


# -------------------------------------------------------------------------- main

def main():
    memory = psutil.virtual_memory()
    info = {
        "collected_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "os": collect_os(),
        "cpu": collect_cpu(),
        "memory": {
            "physical_bytes": memory.total,
            "physical_gib": round(memory.total / 2**30, 2),
            "available_bytes_at_collection": memory.available,
            "benchmark_memory_budget_bytes": int(memory.total * MEMORY_BUDGET_FRACTION),
        },
        "power": collect_power(),
        "toolchain": collect_toolchain(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    cpu = info["cpu"]
    print(f"OS:     {info['os']['name']} {info['os']['version']} (build {info['os']['build']})")
    print(f"CPU:    {cpu['model']}  ({cpu.get('p_cores')} P + {cpu.get('e_cores')} E cores, "
          f"{cpu['logical_cpus']} logical)")
    print(f"Pin:    CPU {cpu['benchmark_cpu']}  [{cpu['pinning']}]")
    print(f"RAM:    {info['memory']['physical_gib']} GiB")
    print(f"Power:  AC={info['power'].get('ac_power')}  mode={info['power'].get('power_mode')}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
