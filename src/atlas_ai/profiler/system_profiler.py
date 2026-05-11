import os
import platform
import shutil
import sys


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to gigabytes."""
    return round(num_bytes / (1024 ** 3), 2)


def get_system_profile() -> dict:
    """Collect basic machine and runtime metrics."""

    total, used, free = shutil.disk_usage("/")

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "disk_total_gb": bytes_to_gb(total),
        "disk_used_gb": bytes_to_gb(used),
        "disk_free_gb": bytes_to_gb(free),
    }
