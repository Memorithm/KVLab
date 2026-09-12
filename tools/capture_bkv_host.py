#!/usr/bin/env python3
"""Capture host evidence for Boolean-KV CPU/NUMA campaigns.

The output is descriptive evidence only.  It deliberately records unavailable
commands instead of inferring CPU SKU, NUMA topology, ISA support or memory
capacity from a chassis/model name.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 1


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def _command(argv: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def capture() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "sysfs": {
            "cpu_online": _read_text("/sys/devices/system/cpu/online"),
            "node_online": _read_text("/sys/devices/system/node/online"),
        },
        "proc": {
            "meminfo": _read_text("/proc/meminfo"),
            "cpuinfo": _read_text("/proc/cpuinfo"),
        },
        "commands": {
            "lscpu_json": _command(["lscpu", "-J"]),
            "lscpu_topology": _command(
                ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ"]
            ),
            "numactl_hardware": _command(["numactl", "--hardware"]),
        },
    }


def main() -> int:
    payload = capture()
    json.dump(payload, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
