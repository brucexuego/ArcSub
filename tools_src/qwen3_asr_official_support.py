from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openvino_asr_env import prepare_openvino_env
from python_runtime_bootstrap import ensure_python_packages

QWEN3_ASR_REPO = "QwenLM/Qwen3-ASR"
QWEN3_ASR_COMMIT = "c17a131fe028b2e428b6e80a33d30bb4fa57b8df"
OPENVINO_NOTEBOOKS_HELPER_COMMIT = "2668aec533aaef88af43b1e2cc11d900cefa832a"
OPENVINO_NOTEBOOKS_HELPER_URL = (
    "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/"
    f"{OPENVINO_NOTEBOOKS_HELPER_COMMIT}/notebooks/qwen3-asr/qwen_3_asr_helper.py"
)

THINKER_LANGUAGE_NAME = "openvino_thinker_language_model.xml"
THINKER_AUDIO_NAME = "openvino_thinker_audio_model.xml"
THINKER_AUDIO_ENCODER_NAME = "openvino_thinker_audio_encoder_model.xml"
THINKER_EMBEDDING_NAME = "openvino_thinker_embedding_model.xml"

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _WORKSPACE_ROOT / "runtime" / "local" / "qwen3_asr_support"
_LEGACY_LOCAL_CACHE_ROOT = _WORKSPACE_ROOT / "local" / "qwen3_asr_support"
_LEGACY_TMP_CACHE_ROOT = _WORKSPACE_ROOT / "tmp" / "qwen3_asr_support"
_HELPER_MODULE: Any = None


def _ensure_qwen_runtime_packages(include_torch: bool = True) -> None:
    requirements: dict[str, str] = {
        "nagisa": "nagisa==0.2.11",
        "nncf": "nncf",
        "openvino": "openvino",
        "soynlp": "soynlp==0.0.493",
        "transformers": "transformers",
        "huggingface_hub": "huggingface_hub",
        "sentencepiece": "sentencepiece",
    }
    if include_torch:
        requirements["torch"] = "torch"
    ensure_python_packages(
        requirements,
        reason="Qwen3-ASR OpenVINO helper",
    )


def _normalize_torch_dtype_name(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if value in ("", "auto", "default"):
        return "float32"
    if value in ("float32", "fp32"):
        return "float32"
    if value in ("float16", "fp16"):
        return "float16"
    if value in ("bfloat16", "bf16"):
        return "bfloat16"
    raise RuntimeError(
        f'Unsupported Qwen3-ASR forced aligner dtype "{value}". Expected one of: float32, float16, bfloat16.'
    )


def _download_to_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temp_path.replace(destination)
    return destination


def _download_and_extract_github_archive(repo: str, commit: str, destination: Path) -> Path:
    marker = destination / ".ready"
    if marker.exists():
        return destination

    archive_url = f"https://github.com/{repo}/archive/{commit}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="arcsub_qwen3asr_", dir=str(destination.parent)))
    archive_path = temp_root / "repo.zip"
    extracted_root = temp_root / "extract"
    try:
        _download_to_file(archive_url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extracted_root)

        extracted_items = list(extracted_root.iterdir())
        if len(extracted_items) != 1 or not extracted_items[0].is_dir():
            raise RuntimeError(f"Unexpected archive layout for {repo}@{commit}")

        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        shutil.move(str(extracted_items[0]), str(destination))
        marker.write_text(commit, encoding="utf-8")
        return destination
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _ensure_helper_source(helper_path: Path) -> Path:
    if helper_path.exists():
        return helper_path
    return _download_to_file(OPENVINO_NOTEBOOKS_HELPER_URL, helper_path)


def _migrate_legacy_cache() -> None:
    if _CACHE_ROOT.exists():
        return

    for legacy_root in (_LEGACY_LOCAL_CACHE_ROOT, _LEGACY_TMP_CACHE_ROOT):
        if not legacy_root.exists():
            continue
        _CACHE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_root), str(_CACHE_ROOT))
        return


def ensure_official_support_sources() -> Dict[str, str]:
    _migrate_legacy_cache()
    qwen_root = _download_and_extract_github_archive(
        QWEN3_ASR_REPO,
        QWEN3_ASR_COMMIT,
        _CACHE_ROOT / f"Qwen3-ASR-{QWEN3_ASR_COMMIT}",
    )
    helper_path = _ensure_helper_source(
        _CACHE_ROOT / f"qwen_3_asr_helper_{OPENVINO_NOTEBOOKS_HELPER_COMMIT}.py"
    )
    return {
        "cacheRoot": str(_CACHE_ROOT),
        "qwenRepoRoot": str(qwen_root),
        "qwenPackageRoot": str(qwen_root / "qwen_asr"),
        "openvinoHelperPath": str(helper_path),
    }


def _ensure_qwen_package(package_root: Path) -> None:
    module = sys.modules.get("qwen_asr")
    if module is None:
        module = types.ModuleType("qwen_asr")
        sys.modules["qwen_asr"] = module
    module.__file__ = str(package_root / "__init__.py")
    module.__path__ = [str(package_root)]
    if not hasattr(module, "Qwen3ASRModel"):
        module.Qwen3ASRModel = object


def _register_qwen_transformers_types(package_root: Path) -> None:
    _ensure_qwen_runtime_packages(include_torch=False)
    _ensure_qwen_package(package_root)
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    from transformers import AutoConfig, AutoProcessor

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ValueError:
        pass
    try:
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except ValueError:
        pass


def ensure_official_metadata_files(repo_id: str, output_dir: str) -> Dict[str, Any]:
    prepare_openvino_env()
    _ensure_qwen_runtime_packages(include_torch=False)
    paths = ensure_official_support_sources()
    package_root = Path(paths["qwenPackageRoot"])
    _register_qwen_transformers_types(package_root)

    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from transformers import AutoProcessor

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    config = Qwen3ASRConfig.from_pretrained(repo_id)
    config.save_pretrained(target_dir)

    processor = AutoProcessor.from_pretrained(repo_id, fix_mistral_regex=True)
    processor.save_pretrained(target_dir)

    return {
        "metadataEnsured": True,
        "modelPath": str(target_dir),
    }


def load_official_helper() -> Any:
    global _HELPER_MODULE
    if _HELPER_MODULE is not None:
        return _HELPER_MODULE

    prepare_openvino_env()
    _ensure_qwen_runtime_packages(include_torch=True)
    paths = ensure_official_support_sources()
    helper_path = Path(paths["openvinoHelperPath"])
    _register_qwen_transformers_types(Path(paths["qwenPackageRoot"]))

    source = helper_path.read_text(encoding="utf-8")
    source = source.replace('sys.path.insert(0, str(Path(__file__).parent / "Qwen3-ASR"))', "")

    module_name = "arcsub_qwen3_asr_official_helper"
    module = types.ModuleType(module_name)
    module.__file__ = str(helper_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, str(helper_path), "exec"), module.__dict__)
    _HELPER_MODULE = module
    return module


def _resolve_compression_mode(raw: str | None) -> str | None:
    value = str(raw or "").strip().lower()
    if not value or value in ("none", "off", "false", "0"):
        return None
    if value in ("int4", "int4_asym", "int4-sym", "int4_sym"):
        return "int4"
    if value in ("int8", "int8_asym", "int8-sym", "int8_sym"):
        return "int8"
    raise RuntimeError(
        f'Unsupported Qwen3-ASR compression mode "{value}". Expected one of: none, int8, int4.'
    )


def _build_quantization_config(helper: Any, compression_mode: str | None) -> Dict[str, Any] | None:
    resolved_mode = _resolve_compression_mode(compression_mode)
    if not resolved_mode:
        return None

    nncf_module = getattr(helper, "nncf", None)
    mode_enum = getattr(nncf_module, "CompressWeightsMode", None) if nncf_module is not None else None
    if mode_enum is None:
        raise RuntimeError("NNCF CompressWeightsMode is unavailable for Qwen3-ASR conversion.")

    if resolved_mode == "int4":
        mode = getattr(mode_enum, "INT4_ASYM", None)
        if mode is None:
            raise RuntimeError("NNCF INT4_ASYM compression mode is unavailable.")
        return {
            "mode": mode,
            "group_size": 128,
            "ratio": 1.0,
        }

    mode = getattr(mode_enum, "INT8_ASYM", None)
    if mode is None:
        raise RuntimeError("NNCF INT8_ASYM compression mode is unavailable.")
    return {
        "mode": mode,
    }


def convert_model(
    repo_id: str,
    output_dir: str,
    use_local_dir: bool = False,
    compression_mode: str | None = None,
) -> Dict[str, Any]:
    helper = load_official_helper()
    target_dir = Path(output_dir).resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved_compression_input = (
        compression_mode
        if compression_mode is not None
        else (os.environ.get("OPENVINO_QWEN3_ASR_CONVERSION_COMPRESSION_MODE") or "int8")
    )
    quantization_config = _build_quantization_config(
        helper,
        resolved_compression_input,
    )
    helper.convert_qwen3_asr_model(
        repo_id,
        str(target_dir),
        quantization_config=quantization_config,
        use_local_dir=use_local_dir,
    )
    ensure_official_metadata_files(repo_id, str(target_dir))
    return {
        "converted": True,
        "modelPath": str(target_dir),
        "runtimeKind": "qwen3_asr_official",
        "compressionMode": _resolve_compression_mode(resolved_compression_input),
    }


def _resolve_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        value = int(raw) if raw else int(default)
    except Exception:
        value = int(default)
    return max(minimum, min(value, maximum))


def create_model(model_dir: str, device: str = "CPU", max_batch_size: int | None = None, max_new_tokens: int | None = None) -> Any:
    helper = load_official_helper()
    resolved_batch_size = max_batch_size if max_batch_size is not None else _resolve_int_env(
        "OPENVINO_QWEN3_ASR_MAX_BATCH_SIZE",
        1,
        1,
        32,
    )
    resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else _resolve_int_env(
        "OPENVINO_QWEN3_ASR_MAX_NEW_TOKENS",
        256,
        16,
        4096,
    )
    return helper.OVQwen3ASRModel.from_pretrained(
        model_dir=str(Path(model_dir).resolve()),
        device=device,
        max_inference_batch_size=resolved_batch_size,
        max_new_tokens=resolved_max_new_tokens,
    )


def create_forced_aligner(
    repo_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> Any:
    prepare_openvino_env()
    _ensure_qwen_runtime_packages(include_torch=True)
    paths = ensure_official_support_sources()
    package_root = Path(paths["qwenPackageRoot"])
    _register_qwen_transformers_types(package_root)

    import torch
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    resolved_repo_id = str(
        repo_id
        or os.environ.get("OPENVINO_QWEN3_ASR_FORCED_ALIGNER_REPO")
        or "Qwen/Qwen3-ForcedAligner-0.6B"
    ).strip()
    resolved_device = str(
        device
        or os.environ.get("OPENVINO_QWEN3_ASR_FORCED_ALIGNER_DEVICE")
        or "cpu"
    ).strip().lower()
    resolved_dtype_name = _normalize_torch_dtype_name(
        dtype or os.environ.get("OPENVINO_QWEN3_ASR_FORCED_ALIGNER_DTYPE")
    )
    resolved_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[resolved_dtype_name]

    if resolved_device in ("cuda", "gpu") and torch.cuda.is_available():
        device_map = "cuda:0"
    else:
        device_map = "cpu"
        if resolved_dtype in (torch.float16, torch.bfloat16):
            resolved_dtype = torch.float32

    return Qwen3ForcedAligner.from_pretrained(
        resolved_repo_id,
        dtype=resolved_dtype,
        device_map=device_map,
    )
