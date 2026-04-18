from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, Iterable, Optional

_DLL_DIR_HANDLES: list[Any] = []


def resolve_workspace_root() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).resolve().parent.parent
    return pathlib.Path(__file__).resolve().parent.parent


def _merge_env_path(key: str, prepend_values: Iterable[str], sep: str) -> str:
    current = os.environ.get(key, "")
    existing = [item for item in current.split(sep) if item]
    merged: list[str] = []
    seen = set()

    for value in prepend_values:
        if not value:
            continue
        norm = os.path.normcase(os.path.normpath(value))
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(value)

    for value in existing:
        norm = os.path.normcase(os.path.normpath(value))
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(value)

    return sep.join(merged)


def _candidate_repo_bin_dirs(workspace_root: pathlib.Path) -> list[pathlib.Path]:
    return [
        workspace_root / "node_modules" / "openvino-node" / "bin",
        workspace_root / "node_modules" / "openvino-genai-node" / "bin",
    ]


def _candidate_intel_openvino_roots() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    for env_key in (
        "OPENVINO_HELPER_INTEL_ROOT",
        "INTEL_OPENVINO_DIR",
        "OPENVINO_INSTALL_DIR",
        "OPENVINO_DIR",
        "OPENVINO_GENAI_DIR",
    ):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            candidates.append(pathlib.Path(raw))

    default_parents = (
        [pathlib.Path(r"C:\Program Files (x86)\Intel"), pathlib.Path(r"C:\Program Files\Intel")]
        if sys.platform == "win32"
        else [pathlib.Path("/opt/intel")]
    )
    for parent in default_parents:
        if not parent.is_dir():
            continue
        candidates.extend(sorted(parent.glob("openvino*"), reverse=True))

    unique: list[pathlib.Path] = []
    seen = set()
    for item in candidates:
        norm = os.path.normcase(os.path.normpath(str(item)))
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(item)
    return unique


def _candidate_runtime_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    runtime_roots = [root]
    if root.name.lower() == "runtime":
        runtime_roots.append(root.parent)
    else:
        runtime_roots.append(root / "runtime")

    dirs: list[pathlib.Path] = []
    for candidate in runtime_roots:
        dirs.extend(
            [
                candidate / "bin" / "intel64" / "Release",
                candidate / "bin" / "intel64" / "Debug",
                candidate / "3rdparty" / "tbb" / "bin",
                candidate / "lib",
                candidate / "lib" / "intel64",
                candidate / "lib" / "intel64" / "Release",
                candidate / "lib" / "aarch64",
                candidate / "lib" / "aarch64" / "Release",
                candidate / "3rdparty" / "tbb" / "lib",
            ]
        )

    unique: list[pathlib.Path] = []
    seen = set()
    for item in dirs:
        norm = os.path.normcase(os.path.normpath(str(item)))
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(item)
    return unique


def _candidate_python_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    install_roots = [root.parent] if root.name.lower() == "runtime" else [root]
    dirs: list[pathlib.Path] = []
    for candidate in install_roots:
        dirs.extend([candidate / "python", candidate / "python" / "python3"])
    return dirs


def prepare_openvino_env() -> Dict[str, Any]:
    workspace_root = resolve_workspace_root()
    repo_bin_dirs = [str(path) for path in _candidate_repo_bin_dirs(workspace_root) if path.is_dir()]

    intel_root: Optional[pathlib.Path] = None
    intel_bin_dirs: list[str] = []
    intel_python_paths: list[str] = []
    for candidate in _candidate_intel_openvino_roots():
        runtime_dirs = [path for path in _candidate_runtime_dirs(candidate) if path.is_dir()]
        python_dirs = [path for path in _candidate_python_dirs(candidate) if path.is_dir()]
        if runtime_dirs:
            intel_root = candidate
            intel_bin_dirs = [str(path) for path in runtime_dirs]
            intel_python_paths = [str(path) for path in python_dirs]
            break

    all_bin_dirs = repo_bin_dirs + intel_bin_dirs
    os.environ["OPENVINO_LIB_PATHS"] = _merge_env_path("OPENVINO_LIB_PATHS", all_bin_dirs, os.pathsep)
    if sys.platform == "win32":
        os.environ["PATH"] = _merge_env_path("PATH", all_bin_dirs, os.pathsep)
    else:
        os.environ["LD_LIBRARY_PATH"] = _merge_env_path("LD_LIBRARY_PATH", all_bin_dirs, os.pathsep)
        if sys.platform == "darwin":
            os.environ["DYLD_LIBRARY_PATH"] = _merge_env_path("DYLD_LIBRARY_PATH", all_bin_dirs, os.pathsep)
    if intel_python_paths:
        os.environ["PYTHONPATH"] = _merge_env_path("PYTHONPATH", intel_python_paths, os.pathsep)
    if intel_root:
        os.environ["INTEL_OPENVINO_DIR"] = str(intel_root)

    if hasattr(os, "add_dll_directory"):
        for value in all_bin_dirs:
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(value))
            except OSError:
                continue

    return {
        "workspaceRoot": str(workspace_root),
        "repoBinDirs": repo_bin_dirs,
        "intelOpenvinoRoot": str(intel_root) if intel_root else None,
        "intelBinDirs": intel_bin_dirs,
        "openvinoLibPaths": os.environ.get("OPENVINO_LIB_PATHS", ""),
    }
