import platform
import sys


def get_system_info() -> dict:
    """Return basic runtime and machine information."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
