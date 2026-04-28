#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Iterable

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _resolve_workspace_path(raw_path: str | None, fallback: pathlib.Path) -> pathlib.Path:
    value = str(raw_path or "").strip()
    if not value:
        return fallback.resolve()
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = SCRIPT_DIR.parent / candidate
    return candidate.resolve()


def _resolve_runtime_root() -> pathlib.Path:
    return _resolve_workspace_path(
        os.environ.get("ARCSUB_RUNTIME_DIR") or os.environ.get("APP_RUNTIME_DIR"),
        SCRIPT_DIR.parent / "runtime",
    )


def _resolve_runtime_tmp_root() -> pathlib.Path:
    root = _resolve_workspace_path(
        os.environ.get("ARCSUB_RUNTIME_TMP_DIR") or os.environ.get("APP_TMP_DIR"),
        _resolve_runtime_root() / "tmp",
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


RUNTIME_TMP_ROOT = _resolve_runtime_tmp_root()
CONVERTER_TMP_ROOT = RUNTIME_TMP_ROOT / "openvino-convert"
HF_CACHE_ROOT = RUNTIME_TMP_ROOT / "huggingface"
PIP_CACHE_ROOT = RUNTIME_TMP_ROOT / "pip-cache"
TORCH_CACHE_ROOT = RUNTIME_TMP_ROOT / "torch-cache"
CONVERTER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
PIP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
for env_key in ("TMPDIR", "TEMP", "TMP"):
    os.environ[env_key] = str(RUNTIME_TMP_ROOT)
os.environ["ARCSUB_RUNTIME_TMP_DIR"] = str(RUNTIME_TMP_ROOT)
os.environ["APP_TMP_DIR"] = str(RUNTIME_TMP_ROOT)
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ["HF_XET_CACHE"] = str(HF_CACHE_ROOT / "xet")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_ROOT / "datasets")
os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE_ROOT)
os.environ["TORCH_HOME"] = str(TORCH_CACHE_ROOT)
tempfile.tempdir = str(RUNTIME_TMP_ROOT)

from openvino_asr_env import prepare_openvino_env
from python_runtime_bootstrap import ensure_python_packages


prepare_openvino_env()
ensure_python_packages(
    {
        "nncf": "nncf",
        "openvino": "openvino",
        "huggingface_hub": "huggingface_hub",
        "hf_xet": "hf_xet",
        "openvino_tokenizers": "openvino-tokenizers",
        "transformers": "transformers",
        "optimum.intel": "optimum-intel[openvino]",
        "sentencepiece": "sentencepiece",
    },
    reason="OpenVINO Hugging Face conversion helper",
)

logging.getLogger("nncf").setLevel(logging.ERROR)
logging.getLogger("optimum").setLevel(logging.ERROR)

import nncf  # type: ignore
import openvino as ov  # type: ignore
from huggingface_hub import snapshot_download  # type: ignore
from openvino_tokenizers import convert_tokenizer  # type: ignore
from transformers import AutoProcessor, AutoTokenizer, Wav2Vec2FeatureExtractor  # type: ignore


COPY_METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "normalizer.json",
    "chat_template.jinja",
    "chat_template.json",
]

SOURCE_ARTIFACT_SUFFIXES = (
    ".onnx",
    ".pdmodel",
    ".pdiparams",
    ".pt",
    ".pt2",
    ".pth",
    ".bin",
    ".safetensors",
    ".msgpack",
    ".h5",
    ".keras",
    ".gguf",
    ".pb",
    ".meta",
    ".index",
)

HF_SNAPSHOT_IGNORE_PATTERNS = [
    "optimizer.bin",
    "checkpoint-*/optimizer.bin",
    "scheduler.bin",
    "checkpoint-*/scheduler.bin",
    "random_states_*.pkl",
    "checkpoint-*/random_states_*.pkl",
    "rng_state*.pth",
    "checkpoint-*/rng_state*.pth",
    "trainer_state.json",
    "checkpoint-*/trainer_state.json",
    "training_args.bin",
    "checkpoint-*/training_args.bin",
    "events.out.tfevents.*",
    "checkpoint-*/events.out.tfevents.*",
    "runs/*",
    "checkpoint-*/runs/*",
    "whisper-github/*/*.pt",
]

ENCODER_DECODER_OV_XML_FILES = [
    "openvino_encoder_model.xml",
    "openvino_decoder_model.xml",
    "openvino_decoder_with_past_model.xml",
]

COHERE_ASR_OV_MARKER_FILE = "arcsub_cohere_asr_ov_config.json"


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    return default


def _resolve_hf_token() -> str | None:
    token = str(os.environ.get("HF_TOKEN", "")).strip()
    return token or None


def _ensure_dir(path_like: str | pathlib.Path) -> pathlib.Path:
    path_obj = pathlib.Path(path_like).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _assert_safe_output_dir(output_dir: pathlib.Path) -> None:
    resolved = output_dir.resolve()
    cwd = pathlib.Path.cwd().resolve()
    repo_root = SCRIPT_DIR.parent.resolve()
    home = pathlib.Path.home().resolve()
    anchor = pathlib.Path(resolved.anchor).resolve()

    unsafe_exact = {cwd, repo_root, home, anchor}
    if resolved in unsafe_exact:
        raise RuntimeError(f"Refusing unsafe output directory for model conversion: {resolved}")
    if (resolved / ".git").exists() or (resolved / "package.json").exists():
        raise RuntimeError(f"Refusing to clear a source tree as model output directory: {resolved}")
    if len(resolved.parts) < 4:
        raise RuntimeError(f"Refusing suspiciously shallow model output directory: {resolved}")


def _iter_repo_files(root_dir: pathlib.Path) -> list[str]:
    files: list[str] = []
    for path_obj in root_dir.rglob("*"):
        if path_obj.is_file():
            files.append(path_obj.relative_to(root_dir).as_posix())
    return sorted(files)


def _cleanup_non_ov_sources(output_dir: pathlib.Path) -> list[str]:
    removed: list[str] = []
    for path_obj in output_dir.rglob("*"):
        if not path_obj.is_file():
            continue
        relative = path_obj.relative_to(output_dir).as_posix()
        lower_relative = relative.lower()
        if lower_relative.startswith("openvino_") and (lower_relative.endswith(".xml") or lower_relative.endswith(".bin")):
            continue
        if "/openvino_" in lower_relative and (lower_relative.endswith(".xml") or lower_relative.endswith(".bin")):
            continue
        if lower_relative.startswith("thinker/openvino_") and (lower_relative.endswith(".xml") or lower_relative.endswith(".bin")):
            continue
        if lower_relative in COPY_METADATA_FILES:
            continue
        if lower_relative.endswith(SOURCE_ARTIFACT_SUFFIXES) or "/variables/" in lower_relative:
            path_obj.unlink(missing_ok=True)
            removed.append(relative)
    return sorted(removed)


def _pick_existing(files: Iterable[str], preferred: Iterable[str]) -> list[str]:
    file_set = set(files)
    return [item for item in preferred if item in file_set]


def _copy_repo_metadata(snapshot_dir: pathlib.Path, output_dir: pathlib.Path) -> list[str]:
    copied: list[str] = []
    available = set(_iter_repo_files(snapshot_dir))
    for relative_name in _pick_existing(available, COPY_METADATA_FILES):
        source_path = snapshot_dir / relative_name
        target_path = output_dir / relative_name
        _copy_file_with_retry(source_path, target_path)
        copied.append(relative_name)
    return copied


def _compress_to_int8(model: ov.Model) -> ov.Model:
    compress_weights = getattr(nncf, "compress_weights", None)
    if not callable(compress_weights):
        raise RuntimeError("nncf.compress_weights is unavailable.")

    mode_enum = getattr(nncf, "CompressWeightsMode", None)
    mode = getattr(mode_enum, "INT8_ASYM", None) if mode_enum is not None else None
    if mode is not None:
        return compress_weights(model, mode=mode)
    return compress_weights(model)


def _retry_file_operation(operation_name: str, callback, *, missing_ok: bool = False):
    attempts = max(1, int(os.environ.get("OPENVINO_HF_CONVERTER_FILE_RETRY_ATTEMPTS", "24") or "24"))
    delay_sec = max(0.05, float(os.environ.get("OPENVINO_HF_CONVERTER_FILE_RETRY_DELAY_SEC", "0.25") or "0.25"))
    last_error: BaseException | None = None

    for attempt in range(attempts):
        try:
            return callback()
        except FileNotFoundError as error:
            if missing_ok:
                return None
            raise error
        except PermissionError as error:
            last_error = error
        except OSError as error:
            if getattr(error, "winerror", None) not in (32, 33):
                raise
            last_error = error

        gc.collect()
        if attempt + 1 < attempts:
            time.sleep(delay_sec * min(4, attempt + 1))

    raise RuntimeError(f"{operation_name} failed after {attempts} attempts: {last_error}") from last_error


def _replace_file_with_retry(source_path: pathlib.Path, target_path: pathlib.Path) -> None:
    def replace() -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)

    _retry_file_operation(f"Replacing {target_path}", replace)


def _copy_file_with_retry(source_path: pathlib.Path, target_path: pathlib.Path) -> None:
    def copy() -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    _retry_file_operation(f"Copying {source_path} to {target_path}", copy)


def _remove_tree_with_retry(path_obj: pathlib.Path) -> None:
    if not path_obj.exists():
        return

    def remove() -> None:
        shutil.rmtree(path_obj)

    _retry_file_operation(f"Removing {path_obj}", remove, missing_ok=True)


def _temporary_directory(prefix: str):
    CONVERTER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=str(CONVERTER_TMP_ROOT),
        ignore_cleanup_errors=True,
    )


def _save_model_preserve_precision(model: ov.Model, xml_path: pathlib.Path) -> None:
    _save_ov_model(model, xml_path)


def _save_ov_model(model: ov.Model, xml_path: pathlib.Path) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    final_bin = xml_path.with_suffix(".bin")
    with _temporary_directory(f"{xml_path.stem}-save-") as temp_dir:
        temp_xml = pathlib.Path(temp_dir) / xml_path.name
        temp_bin = temp_xml.with_suffix(".bin")
        ov.save_model(model, str(temp_xml), compress_to_fp16=False)
        gc.collect()

        _replace_file_with_retry(temp_bin, final_bin)
        _replace_file_with_retry(temp_xml, xml_path)


def _compress_exported_ov_files(output_dir: pathlib.Path, xml_names: Iterable[str]) -> list[str]:
    compressed: list[str] = []
    for xml_name in xml_names:
        xml_path = output_dir / xml_name
        if not xml_path.exists():
            continue
        bin_path = xml_path.with_suffix(".bin")
        if not bin_path.exists():
            raise RuntimeError(f"Cannot compress OpenVINO IR without matching bin file: {bin_path}")

        with _temporary_directory(f"{xml_path.stem}-compress-") as temp_dir:
            source_xml = pathlib.Path(temp_dir) / xml_path.name
            source_bin = source_xml.with_suffix(".bin")
            _copy_file_with_retry(xml_path, source_xml)
            _copy_file_with_retry(bin_path, source_bin)

            source_model = ov.Core().read_model(str(source_xml))
            compressed_model = _compress_to_int8(source_model)
            del source_model
            gc.collect()

            _save_ov_model(compressed_model, xml_path)
            del compressed_model
            gc.collect()
        compressed.append(xml_name)
    return compressed


def _convert_single_model_to_int8(source_path: pathlib.Path, output_xml_path: pathlib.Path) -> None:
    model = ov.convert_model(str(source_path))
    compressed = _compress_to_int8(model)
    _save_ov_model(compressed, output_xml_path)


def _generate_openvino_tokenizers(snapshot_dir: pathlib.Path, output_dir: pathlib.Path, trust_remote_code: bool) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    converted = convert_tokenizer(tokenizer, with_detokenizer=True)
    if isinstance(converted, tuple):
        tokenizer_model, detokenizer_model = converted
    else:
        tokenizer_model = converted
        detokenizer_model = None

    saved: list[str] = []
    tokenizer_xml = output_dir / "openvino_tokenizer.xml"
    _save_ov_model(tokenizer_model, tokenizer_xml)
    saved.extend(["openvino_tokenizer.xml", "openvino_tokenizer.bin"])

    if detokenizer_model is not None:
        detokenizer_xml = output_dir / "openvino_detokenizer.xml"
        _save_ov_model(detokenizer_model, detokenizer_xml)
        saved.extend(["openvino_detokenizer.xml", "openvino_detokenizer.bin"])

    return saved


def _save_ctc_processor_bundle(
    repo_id: str,
    output_dir: pathlib.Path,
    trust_remote_code: bool,
    token: str | None,
) -> dict:
    try:
        processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        processor.save_pretrained(output_dir)
        return {
            "processorMode": "auto-processor",
        }
    except Exception as error:
        message = str(error)
        if "pyctcdecode" not in message.lower():
            raise

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        repo_id,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    feature_extractor.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    tokenizer.save_pretrained(output_dir)
    return {
        "processorMode": "feature-extractor-plus-tokenizer",
    }


def _download_snapshot(repo_id: str, work_dir: pathlib.Path) -> pathlib.Path:
    target_dir = work_dir / "snapshot"
    token = _resolve_hf_token()
    print(f"[arcsub-convert] Downloading filtered Hugging Face snapshot: {repo_id}", file=sys.stderr, flush=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=HF_SNAPSHOT_IGNORE_PATTERNS,
        token=token,
    )
    return pathlib.Path(snapshot_path).resolve()


def _find_first_matching(files: list[str], patterns: Iterable[str]) -> str | None:
    file_set = set(files)
    for pattern in patterns:
        if pattern in file_set:
            return pattern
    return None


def _find_first_regex(files: list[str], regex_text: str) -> str | None:
    import re

    regex = re.compile(regex_text, re.IGNORECASE)
    for file_name in files:
        if regex.search(file_name):
            return file_name
    return None


def _resolve_primary_model_source(snapshot_dir: pathlib.Path, source_format: str) -> pathlib.Path:
    files = _iter_repo_files(snapshot_dir)
    if source_format == "onnx":
        selected = (
            _find_first_matching(files, ["model.onnx", "onnx/model.onnx"])
            or _find_first_regex(files, r"(^|/)(model|decoder_model_merged|decoder_model|encoder_model).*\.onnx$")
            or _find_first_regex(files, r"\.onnx$")
        )
    elif source_format == "tensorflow-lite":
        selected = _find_first_regex(files, r"\.tflite$")
    elif source_format == "paddle":
        selected = _find_first_regex(files, r"\.pdmodel$")
    elif source_format == "tensorflow":
        selected = (
            _find_first_matching(files, ["saved_model.pb"])
            or _find_first_regex(files, r"(^|/)saved_model\.pb$")
            or _find_first_regex(files, r"\.pb$")
            or _find_first_regex(files, r"\.h5$")
        )
    elif source_format == "keras":
        selected = _find_first_regex(files, r"\.(keras|h5)$")
    elif source_format == "pytorch":
        selected = _find_first_regex(files, r"\.(pt2|pt|pth)$")
    else:
        selected = None

    if not selected:
        raise RuntimeError(f"Unable to find a primary source file for source format: {source_format}")
    selected_path = snapshot_dir / selected
    if source_format == "tensorflow" and selected_path.name.lower() == "saved_model.pb":
        return selected_path.parent
    return selected_path


def _convert_translate_llm_from_snapshot(
    snapshot_dir: pathlib.Path,
    output_dir: pathlib.Path,
    source_format: str,
    trust_remote_code: bool,
) -> dict:
    model_source = _resolve_primary_model_source(snapshot_dir, source_format)
    _convert_single_model_to_int8(model_source, output_dir / "openvino_model.xml")
    copied = _copy_repo_metadata(snapshot_dir, output_dir)
    tokenizers = _generate_openvino_tokenizers(snapshot_dir, output_dir, trust_remote_code)
    return {
        "mode": "single-model-llm",
        "sourcePath": model_source.relative_to(snapshot_dir).as_posix(),
        "copiedFiles": copied,
        "generatedFiles": ["openvino_model.xml", "openvino_model.bin", *tokenizers],
    }


def _select_whisper_onnx_components(snapshot_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    files = _iter_repo_files(snapshot_dir)
    encoder = (
        _find_first_matching(
            files,
            [
                "onnx/encoder_model_int8.onnx",
                "onnx/encoder_model_quantized.onnx",
                "onnx/encoder_model.onnx",
            ],
        )
        or _find_first_regex(files, r"encoder_model.*\.onnx$")
    )
    decoder = (
        _find_first_matching(
            files,
            [
                "onnx/decoder_model_merged_int8.onnx",
                "onnx/decoder_model_merged_quantized.onnx",
                "onnx/decoder_model_merged.onnx",
                "onnx/decoder_model_int8.onnx",
                "onnx/decoder_model_quantized.onnx",
                "onnx/decoder_model.onnx",
                "onnx/decoder_with_past_model_int8.onnx",
                "onnx/decoder_with_past_model_quantized.onnx",
                "onnx/decoder_with_past_model.onnx",
            ],
        )
        or _find_first_regex(files, r"(decoder_model_merged|decoder_model|decoder_with_past_model).*\.onnx$")
    )

    if not encoder or not decoder:
        raise RuntimeError("Whisper ONNX conversion requires encoder_model.onnx and decoder_model*.onnx artifacts.")

    return snapshot_dir / encoder, snapshot_dir / decoder


def _convert_whisper_from_snapshot(snapshot_dir: pathlib.Path, output_dir: pathlib.Path, source_format: str, trust_remote_code: bool) -> dict:
    if source_format != "onnx":
        raise RuntimeError("Direct Whisper conversion currently requires ONNX component files or an Optimum export path.")

    encoder_source, decoder_source = _select_whisper_onnx_components(snapshot_dir)
    _convert_single_model_to_int8(encoder_source, output_dir / "openvino_encoder_model.xml")
    _convert_single_model_to_int8(decoder_source, output_dir / "openvino_decoder_model.xml")
    copied = _copy_repo_metadata(snapshot_dir, output_dir)
    tokenizers = _generate_openvino_tokenizers(snapshot_dir, output_dir, trust_remote_code)
    return {
        "mode": "whisper-onnx-components",
        "sourcePaths": [
            encoder_source.relative_to(snapshot_dir).as_posix(),
            decoder_source.relative_to(snapshot_dir).as_posix(),
        ],
        "copiedFiles": copied,
        "generatedFiles": [
            "openvino_encoder_model.xml",
            "openvino_encoder_model.bin",
            "openvino_decoder_model.xml",
            "openvino_decoder_model.bin",
            *tokenizers,
        ],
    }


def _convert_ctc_from_snapshot(snapshot_dir: pathlib.Path, output_dir: pathlib.Path, source_format: str) -> dict:
    if source_format != "onnx":
        raise RuntimeError("Direct CTC conversion currently requires an ONNX source file.")

    model_source = _resolve_primary_model_source(snapshot_dir, source_format)
    output_xml = output_dir / "openvino_model.xml"
    if _bool_env("OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION", False):
        model = ov.convert_model(str(model_source))
        _save_model_preserve_precision(model, output_xml)
    else:
        _convert_single_model_to_int8(model_source, output_xml)
    copied = _copy_repo_metadata(snapshot_dir, output_dir)
    return {
        "mode": "ctc-onnx-direct",
        "sourcePath": model_source.relative_to(snapshot_dir).as_posix(),
        "preservedPrecision": _bool_env("OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION", False),
        "copiedFiles": copied,
        "generatedFiles": [
            "openvino_model.xml",
            "openvino_model.bin",
        ],
    }


def _export_ctc_asr_with_ovmodel(
    repo_id: str,
    output_dir: pathlib.Path,
    trust_remote_code: bool,
    *,
    from_tf: bool = False,
) -> dict:
    _maybe_bootstrap_optimum()
    from optimum.intel import OVModelForCTC  # type: ignore

    token = _resolve_hf_token()
    with _temporary_directory("arcsub-ov-ctc-export-") as export_tmp_dir:
        export_dir = pathlib.Path(export_tmp_dir)
        model = OVModelForCTC.from_pretrained(
            repo_id,
            export=True,
            compile=False,
            trust_remote_code=trust_remote_code,
            token=token,
            from_tf=from_tf,
        )
        model.save_pretrained(export_dir)
        model = None
        gc.collect()

        processor_info = _save_ctc_processor_bundle(
            repo_id,
            output_dir,
            trust_remote_code,
            token,
        )

        exported_xml = export_dir / "openvino_model.xml"
        if not exported_xml.exists():
            raise RuntimeError("OVModelForCTC export did not produce openvino_model.xml.")
        for metadata_file in ("config.json", "openvino_config.json"):
            source_metadata = export_dir / metadata_file
            if source_metadata.exists():
                _copy_file_with_retry(source_metadata, output_dir / metadata_file)

        preserve_precision = _bool_env("OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION", False)
        if preserve_precision:
            _copy_file_with_retry(exported_xml, output_dir / "openvino_model.xml")
            _copy_file_with_retry(exported_xml.with_suffix(".bin"), output_dir / "openvino_model.bin")
        else:
            source_model = ov.Core().read_model(str(exported_xml))
            compressed = _compress_to_int8(source_model)
            del source_model
            gc.collect()
            _save_ov_model(compressed, output_dir / "openvino_model.xml")
            del compressed
            gc.collect()

    return {
        "mode": "ovmodel-for-ctc-export",
        "repoId": repo_id,
        "fromTf": from_tf,
        "preservedPrecision": preserve_precision,
        **processor_info,
        "generatedFiles": _iter_repo_files(output_dir),
    }


def _export_transformers_from_tf(repo_id: str, output_dir: pathlib.Path, runtime_layout: str, trust_remote_code: bool) -> dict:
    _maybe_bootstrap_optimum()
    token = _resolve_hf_token()

    if runtime_layout == "asr-whisper":
        from optimum.intel import OVModelForSpeechSeq2Seq  # type: ignore

        model = OVModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            export=True,
            compile=False,
            trust_remote_code=trust_remote_code,
            token=token,
            from_tf=True,
        )
        model.save_pretrained(output_dir)
        model = None
        gc.collect()

        processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        processor.save_pretrained(output_dir)

        tokenizers = _generate_openvino_tokenizers(output_dir, output_dir, trust_remote_code)
        compressed = _compress_exported_ov_files(output_dir, ENCODER_DECODER_OV_XML_FILES)
        return {
            "mode": "ovmodel-from-tf-whisper",
            "repoId": repo_id,
            "generatedFiles": _iter_repo_files(output_dir),
            "compressedFiles": compressed,
            "tokenizerFiles": tokenizers,
        }

    if runtime_layout == "asr-ctc":
        return _export_ctc_asr_with_ovmodel(repo_id, output_dir, trust_remote_code, from_tf=True)

    if runtime_layout == "translate-seq2seq":
        from optimum.intel import OVModelForSeq2SeqLM  # type: ignore

        model = OVModelForSeq2SeqLM.from_pretrained(
            repo_id,
            export=True,
            compile=False,
            trust_remote_code=trust_remote_code,
            token=token,
            from_tf=True,
        )
        model.save_pretrained(output_dir)
        model = None
        gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        tokenizer.save_pretrained(output_dir)

        tokenizers = _generate_openvino_tokenizers(output_dir, output_dir, trust_remote_code)
        compressed = _compress_exported_ov_files(output_dir, ENCODER_DECODER_OV_XML_FILES)
        return {
            "mode": "ovmodel-from-tf-seq2seq",
            "repoId": repo_id,
            "generatedFiles": _iter_repo_files(output_dir),
            "compressedFiles": compressed,
            "tokenizerFiles": tokenizers,
        }

    if runtime_layout == "translate-llm":
        from optimum.intel import OVModelForCausalLM  # type: ignore

        model = OVModelForCausalLM.from_pretrained(
            repo_id,
            export=True,
            compile=False,
            trust_remote_code=trust_remote_code,
            token=token,
            from_tf=True,
        )
        model.save_pretrained(output_dir)
        model = None
        gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        tokenizer.save_pretrained(output_dir)

        tokenizers = _generate_openvino_tokenizers(output_dir, output_dir, trust_remote_code)
        compressed = _compress_exported_ov_files(output_dir, ["openvino_model.xml"])
        return {
            "mode": "ovmodel-from-tf-causal-lm",
            "repoId": repo_id,
            "generatedFiles": _iter_repo_files(output_dir),
            "compressedFiles": compressed,
            "tokenizerFiles": tokenizers,
        }

    raise RuntimeError(f"TensorFlow export is not supported for runtime layout: {runtime_layout}")


def _export_whisper_asr_with_ovmodel(repo_id: str, output_dir: pathlib.Path, trust_remote_code: bool) -> dict:
    _maybe_bootstrap_optimum()

    from optimum.intel import OVModelForSpeechSeq2Seq  # type: ignore

    with _temporary_directory("arcsub-ov-whisper-") as tmp_dir:
        snapshot_dir = _download_snapshot(repo_id, pathlib.Path(tmp_dir))
        print(
            f"[arcsub-convert] Exporting Whisper ASR with OVModelForSpeechSeq2Seq from {snapshot_dir}",
            file=sys.stderr,
            flush=True,
        )
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            str(snapshot_dir),
            export=True,
            compile=False,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model.save_pretrained(output_dir)
        model = None
        gc.collect()

        processor = AutoProcessor.from_pretrained(
            str(snapshot_dir),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        processor.save_pretrained(output_dir)

        tokenizers = _generate_openvino_tokenizers(output_dir, output_dir, trust_remote_code)
        compressed = _compress_exported_ov_files(output_dir, ENCODER_DECODER_OV_XML_FILES)
        return {
            "mode": "ovmodel-for-speech-seq2seq-export",
            "repoId": repo_id,
            "snapshotFiles": _iter_repo_files(snapshot_dir),
            "generatedFiles": _iter_repo_files(output_dir),
            "compressedFiles": compressed,
            "tokenizerFiles": tokenizers,
        }


def _build_cohere_decoder_prompt_ids(tokenizer, language: str = "en"):
    tokens = [
        "▁",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        f"<|{language}|>",
        f"<|{language}|>",
        "<|pnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    ]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if not isinstance(ids, list) or any(item is None or int(item) < 0 for item in ids):
        raise RuntimeError("Unable to build Cohere ASR decoder prompt ids from tokenizer special tokens.")
    return [int(item) for item in ids]


def _export_cohere_asr_with_traced_ov(repo_id: str, output_dir: pathlib.Path, trust_remote_code: bool) -> dict:
    ensure_python_packages(
        {
            "numpy": "numpy",
            "torch": "torch",
        },
        reason="Cohere ASR OpenVINO export helper",
    )
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from transformers import AutoModelForSpeechSeq2Seq  # type: ignore

    class _CohereForwardWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, length, decoder_input_ids, decoder_attention_mask):
            return self.model(
                input_features=input_features,
                length=length,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
            ).logits

    with _temporary_directory("arcsub-ov-cohere-asr-") as tmp_dir:
        snapshot_dir = _download_snapshot(repo_id, pathlib.Path(tmp_dir))
        with open(snapshot_dir / "config.json", "r", encoding="utf-8") as file_obj:
            config = json.load(file_obj)
        model_type = str(config.get("model_type") or "").strip().lower()
        if model_type != "cohere_asr":
            raise RuntimeError(f"Cohere ASR exporter only supports model_type=cohere_asr, got {model_type or 'unknown'}.")

        sample_rate = int(config.get("sample_rate") or config.get("preprocessor", {}).get("sample_rate") or 16000)
        max_audio_clip_s = float(config.get("max_audio_clip_s") or 35.0)
        example_seconds = float(os.environ.get("OPENVINO_COHERE_ASR_EXPORT_EXAMPLE_SECONDS", "8") or "8")
        example_seconds = max(1.0, min(example_seconds, max_audio_clip_s))
        dummy_audio = np.zeros((max(1, int(round(sample_rate * example_seconds))),), dtype=np.float32)

        print(
            f"[arcsub-convert] Exporting Cohere ASR traced OpenVINO INT8 from {snapshot_dir} "
            f"(example_seconds={example_seconds:g})",
            file=sys.stderr,
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(
            str(snapshot_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
        prompt_ids = _build_cohere_decoder_prompt_ids(processor.tokenizer, "en")
        example_inputs = (
            inputs["input_features"].clone(),
            inputs["length"].clone(),
            torch.tensor([prompt_ids], dtype=torch.long),
            torch.ones((1, len(prompt_ids)), dtype=torch.long),
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(snapshot_dir),
            dtype=torch.float32,
            local_files_only=True,
            trust_remote_code=True,
        )
        model.eval()
        wrapper = _CohereForwardWrapper(model).eval()
        torch.set_grad_enabled(False)
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
        ov_model = ov.convert_model(traced, example_input=example_inputs)
        del traced
        del wrapper
        del model
        gc.collect()

        compressed = _compress_to_int8(ov_model)
        del ov_model
        gc.collect()
        _save_ov_model(compressed, output_dir / "openvino_model.xml")
        del compressed
        gc.collect()

        copied = _copy_repo_metadata(snapshot_dir, output_dir)
        marker = {
            "runtimeKind": "cohere_openvino_asr",
            "repoId": repo_id,
            "modelType": model_type,
            "exportMode": "torchscript-forward-int8",
            "exampleSeconds": example_seconds,
            "sampleRate": sample_rate,
            "maxAudioClipSec": max_audio_clip_s,
            "inputOrder": ["input_features", "length", "decoder_input_ids", "decoder_attention_mask"],
            "promptLanguage": "en",
            "decoderPromptIds": prompt_ids,
            "notes": [
                "This IR is a Cohere ASR forward model. ArcSub performs greedy autoregressive decoding in the ASR helper.",
                "The model card requires an explicit source language; native timestamps and diarization are not emitted by this model.",
            ],
        }
        (output_dir / COHERE_ASR_OV_MARKER_FILE).write_text(json.dumps(marker, ensure_ascii=True, indent=2), encoding="utf-8")

        return {
            "mode": "cohere-asr-traced-forward-int8",
            "repoId": repo_id,
            "snapshotFiles": _iter_repo_files(snapshot_dir),
            "generatedFiles": _iter_repo_files(output_dir),
            "copiedMetadataFiles": copied,
            "compressedFiles": ["openvino_model.xml"],
            "markerFile": COHERE_ASR_OV_MARKER_FILE,
            "exampleSeconds": example_seconds,
        }


def _export_visual_causal_lm(repo_id: str, output_dir: pathlib.Path, trust_remote_code: bool) -> dict:
    _maybe_bootstrap_optimum()
    from optimum.intel import OVModelForVisualCausalLM  # type: ignore

    token = _resolve_hf_token()

    model = OVModelForVisualCausalLM.from_pretrained(
        repo_id,
        export=True,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model.save_pretrained(output_dir)
    model = None
    gc.collect()

    processor = AutoProcessor.from_pretrained(
        repo_id,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    processor.save_pretrained(output_dir)

    tokenizers = _generate_openvino_tokenizers(output_dir, output_dir, trust_remote_code)
    return {
        "mode": "ovmodel-visual-causal-lm",
        "repoId": repo_id,
        "generatedFiles": _iter_repo_files(output_dir),
        "tokenizerFiles": tokenizers,
    }


def _maybe_bootstrap_optimum() -> None:
    optimum_intel_spec = importlib.util.find_spec("optimum.intel")
    optimum_export_spec = importlib.util.find_spec("optimum.exporters.openvino")
    if optimum_intel_spec is not None and optimum_export_spec is not None:
        return
    if not _bool_env("OPENVINO_HF_CONVERTER_AUTO_INSTALL_OPTIMUM", False):
        raise RuntimeError(
            "optimum OpenVINO exporter modules are not available. Install optimum-intel[openvino], or set OPENVINO_HF_CONVERTER_AUTO_INSTALL_OPTIMUM=1."
        )

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "optimum-intel[openvino]",
    ]
    subprocess.run(command, check=True)
    importlib.invalidate_caches()
    optimum_intel_spec = importlib.util.find_spec("optimum.intel")
    optimum_export_spec = importlib.util.find_spec("optimum.exporters.openvino")
    if optimum_intel_spec is None or optimum_export_spec is None:
        raise RuntimeError("optimum Python modules are still unavailable after attempting installation.")


def _run_optimum_export(repo_id: str, output_dir: pathlib.Path, hf_task: str | None, trust_remote_code: bool) -> dict:
    _maybe_bootstrap_optimum()
    snapshot_files: list[str] = []
    with _temporary_directory("arcsub-ov-export-") as tmp_dir:
        snapshot_dir = _download_snapshot(repo_id, pathlib.Path(tmp_dir))
        snapshot_files = _iter_repo_files(snapshot_dir)
        command = [
            sys.executable,
            "-m",
            "optimum.commands.optimum_cli",
            "export",
            "openvino",
            "--model",
            str(snapshot_dir),
            "--weight-format",
            "int8",
        ]
        if hf_task:
            command.extend(["--task", hf_task])
        if trust_remote_code:
            command.append("--trust-remote-code")
        command.append(str(output_dir))

        env = os.environ.copy()
        token = _resolve_hf_token()
        if token:
            env["HF_TOKEN"] = token

        print(
            "[arcsub-convert] Running Optimum OpenVINO export: "
            + " ".join(str(part) for part in command),
            file=sys.stderr,
            flush=True,
        )
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as error:
            stdout_tail = [line for line in str(error.stdout or "").splitlines()[-20:] if str(line).strip()]
            stderr_tail = [line for line in str(error.stderr or "").splitlines()[-80:] if str(line).strip()]
            detail_parts: list[str] = [f"optimum-cli failed with exit code {error.returncode}"]
            if stdout_tail:
                detail_parts.append("stdout tail:\n" + "\n".join(stdout_tail))
            if stderr_tail:
                detail_parts.append("stderr tail:\n" + "\n".join(stderr_tail))
            raise RuntimeError("\n\n".join(detail_parts)) from error
    return {
        "mode": "optimum-export-openvino",
        "command": command,
        "snapshotFiles": snapshot_files,
        "stdoutTail": str(completed.stdout or "").splitlines()[-20:],
        "stderrTail": str(completed.stderr or "").splitlines()[-20:],
    }


def _convert_via_snapshot(
    repo_id: str,
    output_dir: pathlib.Path,
    source_format: str,
    runtime_layout: str,
    trust_remote_code: bool,
) -> dict:
    with _temporary_directory("arcsub-ov-convert-") as tmp_dir:
        work_dir = pathlib.Path(tmp_dir)
        snapshot_dir = _download_snapshot(repo_id, work_dir)

        if runtime_layout == "translate-llm":
            result = _convert_translate_llm_from_snapshot(snapshot_dir, output_dir, source_format, trust_remote_code)
        elif runtime_layout == "asr-whisper":
            result = _convert_whisper_from_snapshot(snapshot_dir, output_dir, source_format, trust_remote_code)
        elif runtime_layout == "asr-ctc":
            result = _convert_ctc_from_snapshot(snapshot_dir, output_dir, source_format)
        elif runtime_layout == "translate-vlm":
            raise RuntimeError(
                "Direct file conversion for VLM layouts is not supported in ArcSub yet. Use an Optimum-based export path."
            )
        else:
            raise RuntimeError(f"Unsupported runtime layout for direct conversion: {runtime_layout}")

        result["snapshotFiles"] = _iter_repo_files(snapshot_dir)
        return result


def _infer_optimum_task(model_type: str, runtime_layout: str, requested_type: str) -> str | None:
    if requested_type == "asr":
        if runtime_layout == "asr-whisper":
            return "automatic-speech-recognition-with-past"
        return "automatic-speech-recognition"
    if runtime_layout == "translate-vlm":
        return "image-text-to-text"
    if runtime_layout == "translate-seq2seq":
        return "text2text-generation"
    if model_type == "translate":
        return "text-generation-with-past"
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--type", required=True, choices=["asr", "translate"])
    parser.add_argument("--source-format", required=True)
    parser.add_argument("--conversion-method", required=True)
    parser.add_argument("--runtime-layout", required=True)
    parser.add_argument("--hf-task", default="")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    _assert_safe_output_dir(output_dir)
    if output_dir.exists():
        _remove_tree_with_retry(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trust_remote_code = _bool_env("OPENVINO_HF_EXPORT_TRUST_REMOTE_CODE", False)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.conversion_method == "optimum-export-openvino":
                if args.source_format in ("tensorflow", "keras"):
                    result = _export_transformers_from_tf(args.repo_id, output_dir, args.runtime_layout, trust_remote_code)
                elif args.runtime_layout == "asr-whisper":
                    result = _export_whisper_asr_with_ovmodel(args.repo_id, output_dir, trust_remote_code)
                elif args.runtime_layout == "translate-vlm":
                    result = _export_visual_causal_lm(args.repo_id, output_dir, trust_remote_code)
                else:
                    task_name = args.hf_task.strip() or _infer_optimum_task(args.type, args.runtime_layout, args.type)
                    result = _run_optimum_export(args.repo_id, output_dir, task_name or None, trust_remote_code)
            elif args.conversion_method == "openvino-ctc-asr-export":
                result = _export_ctc_asr_with_ovmodel(args.repo_id, output_dir, trust_remote_code)
            elif args.conversion_method == "openvino-cohere-asr-export":
                result = _export_cohere_asr_with_traced_ov(args.repo_id, output_dir, trust_remote_code=True)
            elif args.conversion_method == "openvino-convert-model":
                result = _convert_via_snapshot(
                    args.repo_id,
                    output_dir,
                    args.source_format,
                    args.runtime_layout,
                    trust_remote_code,
                )
            else:
                raise RuntimeError(f"Unsupported conversion method: {args.conversion_method}")

        payload = {
            "converted": True,
            "repoId": args.repo_id,
            "type": args.type,
            "sourceFormat": args.source_format,
            "conversionMethod": args.conversion_method,
            "runtimeLayout": args.runtime_layout,
            "outputDir": str(output_dir),
            "outputFiles": _iter_repo_files(output_dir),
            "detail": result,
        }
        payload["removedSourceArtifacts"] = _cleanup_non_ov_sources(output_dir)
        payload["outputFiles"] = _iter_repo_files(output_dir)
        if not payload["outputFiles"]:
            raise RuntimeError(
                "Model export finished without producing any output files. "
                "This usually means the Hugging Face model is gated without download access, "
                "or the selected export path did not emit OpenVINO artifacts."
            )
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.stdout.flush()
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.write(
            json.dumps(
                {
                    "converted": False,
                    "repoId": args.repo_id,
                    "error": str(error) or error.__class__.__name__,
                },
                ensure_ascii=True,
            )
        )
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
