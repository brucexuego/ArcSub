from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from typing import Iterable, Mapping


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


def _normalize_requirements(requirements: Mapping[str, str] | Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    if isinstance(requirements, Mapping):
        return [(str(module), str(package)) for module, package in requirements.items()]
    return [(str(module), str(package)) for module, package in requirements]


def _auto_install_enabled(env_var: str, default: bool) -> bool:
    raw = str(os.environ.get(env_var, "")).strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def ensure_python_packages(
    requirements: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    env_var: str = "ARCSUB_AUTO_INSTALL_PYTHON_DEPS",
    default_enabled: bool = True,
    reason: str = "",
) -> list[str]:
    normalized = _normalize_requirements(requirements)
    missing_packages: list[str] = []
    missing_modules: list[str] = []
    seen_packages: set[str] = set()

    for module_name, package_name in normalized:
        if not _module_missing(module_name):
            continue
        missing_modules.append(module_name)
        if package_name not in seen_packages:
            seen_packages.add(package_name)
            missing_packages.append(package_name)

    if not missing_packages:
        return []

    if not _auto_install_enabled(env_var, default_enabled):
        detail = reason or "ArcSub helper runtime"
        raise ModuleNotFoundError(
            f"{detail} is missing Python packages: {', '.join(missing_packages)} "
            f"(modules: {', '.join(missing_modules)})."
        )

    detail = reason or "ArcSub helper runtime"
    sys.stderr.write(
        f"[arcsub-python-bootstrap] Installing missing Python packages for {detail}: "
        f"{', '.join(missing_packages)}\n"
    )
    sys.stderr.flush()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", *missing_packages],
        check=True,
    )
    importlib.invalidate_caches()

    still_missing = [module_name for module_name, _ in normalized if _module_missing(module_name)]
    if still_missing:
        raise ModuleNotFoundError(
            f"Python packages are still unavailable after installation attempt: {', '.join(still_missing)}"
        )

    return missing_packages
