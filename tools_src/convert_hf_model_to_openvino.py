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
import traceback
from typing import Iterable

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openvino_asr_env import prepare_openvino_env
from python_runtime_bootstrap import ensure_python_packages


prepare_openvino_env()
ensure_python_packages(
    {
        "nncf": "nncf",
        "openvino": "openvino",
        "huggingface_hub": "huggingface_hub",
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
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
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


def _save_model_preserve_precision(model: ov.Model, xml_path: pathlib.Path) -> None:
    _save_ov_model(model, xml_path)


def _save_ov_model(model: ov.Model, xml_path: pathlib.Path) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    temp_xml = xml_path.with_name(f"{xml_path.stem}.tmp.xml")
    temp_bin = temp_xml.with_suffix(".bin")
    final_bin = xml_path.with_suffix(".bin")
    temp_xml.unlink(missing_ok=True)
    temp_bin.unlink(missing_ok=True)
    ov.save_model(model, str(temp_xml), compress_to_fp16=False)
    xml_path.unlink(missing_ok=True)
    final_bin.unlink(missing_ok=True)
    shutil.move(temp_xml, xml_path)
    shutil.move(temp_bin, final_bin)


def _compress_exported_ov_files(output_dir: pathlib.Path, xml_names: Iterable[str]) -> list[str]:
    compressed: list[str] = []
    for xml_name in xml_names:
        xml_path = output_dir / xml_name
        if not xml_path.exists():
            continue
        source_model = ov.Core().read_model(str(xml_path))
        compressed_model = _compress_to_int8(source_model)
        del source_model

        temp_xml = xml_path.with_name(f"{xml_path.stem}.compressed.xml")
        temp_bin = temp_xml.with_suffix(".bin")
        final_bin = xml_path.with_suffix(".bin")
        temp_xml.unlink(missing_ok=True)
        temp_bin.unlink(missing_ok=True)
        ov.save_model(compressed_model, str(temp_xml), compress_to_fp16=False)
        del compressed_model
        gc.collect()

        xml_path.unlink(missing_ok=True)
        final_bin.unlink(missing_ok=True)
        shutil.move(temp_xml, xml_path)
        shutil.move(temp_bin, final_bin)
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
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
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
    model = OVModelForCTC.from_pretrained(
        repo_id,
        export=True,
        compile=False,
        trust_remote_code=trust_remote_code,
        token=token,
        from_tf=from_tf,
    )
    model.save_pretrained(output_dir)

    processor_info = _save_ctc_processor_bundle(
        repo_id,
        output_dir,
        trust_remote_code,
        token,
    )

    exported_xml = output_dir / "openvino_model.xml"
    if not exported_xml.exists():
        raise RuntimeError("OVModelForCTC export did not produce openvino_model.xml.")

    preserve_precision = _bool_env("OPENVINO_HF_CONVERTER_CTC_PRESERVE_PRECISION", False)
    if not preserve_precision:
        source_model = ov.Core().read_model(str(exported_xml))
        compressed = _compress_to_int8(source_model)
        _save_ov_model(compressed, exported_xml)

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
        compressed = _compress_exported_ov_files(
            output_dir,
            ["openvino_encoder_model.xml", "openvino_decoder_model.xml"],
        )
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
        compressed = _compress_exported_ov_files(
            output_dir,
            ["openvino_encoder_model.xml", "openvino_decoder_model.xml"],
        )
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
    command = [
        sys.executable,
        "-m",
        "optimum.exporters.openvino",
        "--model",
        repo_id,
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
    with tempfile.TemporaryDirectory(prefix="arcsub-ov-convert-") as tmp_dir:
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
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    trust_remote_code = _bool_env("OPENVINO_HF_EXPORT_TRUST_REMOTE_CODE", False)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.conversion_method == "optimum-export-openvino":
                if args.source_format in ("tensorflow", "keras"):
                    result = _export_transformers_from_tf(args.repo_id, output_dir, args.runtime_layout, trust_remote_code)
                elif args.runtime_layout == "translate-vlm":
                    result = _export_visual_causal_lm(args.repo_id, output_dir, trust_remote_code)
                else:
                    task_name = args.hf_task.strip() or _infer_optimum_task(args.type, args.runtime_layout, args.type)
                    result = _run_optimum_export(args.repo_id, output_dir, task_name or None, trust_remote_code)
            elif args.conversion_method == "openvino-ctc-asr-export":
                result = _export_ctc_asr_with_ovmodel(args.repo_id, output_dir, trust_remote_code)
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
