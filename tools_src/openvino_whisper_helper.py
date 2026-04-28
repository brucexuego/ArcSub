#!/usr/bin/env python3
"""
OpenVINO ASR helper process.

JSON IPC over stdin/stdout (one JSON object per line):
- Request: {"requestId": "...", "method": "...", "params": {...}}
- Response: {"requestId": "...", "ok": true, "result": {...}}
            {"requestId": "...", "ok": false, "error": "..."}
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import time
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openvino_asr_env import prepare_openvino_env
from python_runtime_bootstrap import ensure_python_packages
from qwen3_asr_official_support import (
    THINKER_AUDIO_ENCODER_NAME,
    THINKER_AUDIO_NAME,
    THINKER_EMBEDDING_NAME,
    THINKER_LANGUAGE_NAME,
    convert_model as convert_official_qwen3_asr_model,
    create_forced_aligner as create_official_qwen3_forced_aligner,
    create_model as create_official_qwen3_asr_model,
    ensure_official_support_sources,
)


def _version_tuple(value: str) -> tuple[int, int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", str(value or ""))[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def _runtime_root() -> pathlib.Path:
    configured = (
        os.environ.get("ARCSUB_RUNTIME_DIR")
        or os.environ.get("APP_RUNTIME_DIR")
        or os.environ.get("APP_RUNTIME_PATH")
    )
    if configured:
        return pathlib.Path(configured).expanduser().resolve()
    return (SCRIPT_DIR.parent / "runtime").resolve()


def _module_loaded_from(target_dir: pathlib.Path, module_name: str) -> bool:
    module = sys.modules.get(module_name)
    if module is None:
        return False
    locations = []
    origin = getattr(module, "__file__", None)
    if origin:
        locations.append(origin)
    spec = getattr(module, "__spec__", None)
    if spec is not None:
        spec_origin = getattr(spec, "origin", None)
        if spec_origin:
            locations.append(spec_origin)
        search_locations = getattr(spec, "submodule_search_locations", None) or []
        locations.extend([str(item) for item in search_locations])
    if not locations:
        return False
    try:
        target = target_dir.resolve()
        return any(pathlib.Path(item).resolve().is_relative_to(target) for item in locations)
    except Exception:
        return False


def _clear_loaded_module_tree(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules.keys()):
        if name in prefixes or any(name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def _cohere_transformers_site() -> pathlib.Path:
    return _runtime_root() / "tools" / "python" / "cohere-transcribe-03-2026-site"


def _cohere_transformers_ready(target_dir: pathlib.Path) -> bool:
    target_text = str(target_dir)
    if target_text not in sys.path:
        sys.path.insert(0, target_text)
        importlib.invalidate_caches()

    if "transformers" in sys.modules and not _module_loaded_from(target_dir, "transformers"):
        _clear_loaded_module_tree(("transformers", "huggingface_hub", "tokenizers", "safetensors"))

    try:
        import transformers  # pylint: disable=import-error,import-outside-toplevel
    except Exception:
        return False

    return _module_loaded_from(target_dir, "transformers") and _version_tuple(
        getattr(transformers, "__version__", "0.0.0")
    ) >= (5, 4, 0)


def _ensure_cohere_transformers_runtime() -> pathlib.Path:
    target_dir = _cohere_transformers_site()
    target_dir.mkdir(parents=True, exist_ok=True)
    if _cohere_transformers_ready(target_dir):
        return target_dir

    if not str(os.environ.get("ARCSUB_AUTO_INSTALL_PYTHON_DEPS", "1")).strip().lower() in (
        "",
        "1",
        "true",
        "yes",
        "on",
    ):
        raise ModuleNotFoundError(
            "Cohere Transcribe requires isolated Python packages under runtime/tools/python: "
            "transformers>=5.4.0, protobuf, sentencepiece."
        )

    sys.stderr.write(
        "[arcsub-python-bootstrap] Installing isolated Python packages for Cohere Transcribe ASR: "
        "transformers>=5.4.0, protobuf, sentencepiece\n"
    )
    sys.stderr.flush()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(target_dir),
            "transformers>=5.4.0",
            "protobuf",
            "sentencepiece",
        ],
        check=True,
    )
    _clear_loaded_module_tree(("transformers", "huggingface_hub", "tokenizers", "safetensors", "google.protobuf"))
    importlib.invalidate_caches()
    if not _cohere_transformers_ready(target_dir):
        raise ModuleNotFoundError(
            f"Cohere Transcribe isolated Transformers runtime is unavailable after installation: {target_dir}"
        )
    return target_dir


def _normalize_whisper_language(raw: Optional[str]) -> Optional[str]:
    value = str(raw or "").strip()
    if not value:
        return None

    lower = value.lower()
    if lower in ("auto", "none", "null"):
        return None
    if value.startswith("<|") and value.endswith("|>"):
        return value

    mapped = {
        "jp": "ja",
        "kr": "ko",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh-hk": "zh",
    }.get(lower, lower)
    return f"<|{mapped}|>"


def _detect_runtime_kind(model_dir: pathlib.Path) -> str:
    cohere_ov_required = [
        "arcsub_cohere_asr_ov_config.json",
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "openvino_model.xml",
        "openvino_model.bin",
    ]
    if all((model_dir / name).exists() for name in cohere_ov_required):
        return "cohere_openvino_asr"

    ctc_required = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "openvino_model.xml",
        "openvino_model.bin",
    ]
    ctc_has_tokenizer = any((model_dir / name).exists() for name in ("tokenizer.json", "vocab.json", "vocab.txt"))
    if ctc_has_tokenizer and all((model_dir / name).exists() for name in ctc_required):
        return "ctc_asr"
    official_qwen_base_required = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        f"thinker/{THINKER_LANGUAGE_NAME}",
        f"thinker/{THINKER_AUDIO_NAME}",
        f"thinker/{THINKER_AUDIO_ENCODER_NAME}",
        f"thinker/{THINKER_EMBEDDING_NAME}",
    ]
    has_official_template = (model_dir / "chat_template.jinja").exists() or (model_dir / "chat_template.json").exists()
    if has_official_template and all((model_dir / name).exists() for name in official_qwen_base_required):
        return "qwen3_asr_official"
    cohere_required = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors",
        "configuration_cohere_asr.py",
        "modeling_cohere_asr.py",
        "processing_cohere_asr.py",
        "tokenization_cohere_asr.py",
    ]
    if all((model_dir / name).exists() for name in cohere_required):
        return "cohere_transformers_asr"
    if (model_dir / "openvino_encoder_model.xml").exists():
        return "whisper"
    raise RuntimeError(f"Unsupported ASR model layout: {model_dir}")


def _coerce_text_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts).strip()
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">") and "object at 0x" in text:
        return ""
    return text


def _read_field(value: Any, *names: str) -> Any:
    if value is None:
        return None
    for name in names:
        if isinstance(value, dict) and name in value:
            candidate = value.get(name)
            if candidate is not None:
                return candidate
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _coerce_float_payload(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number != number:
        return None
    return number


@dataclass
class RuntimeState:
    env_info: Dict[str, Any]
    openvino_genai: Any = None
    librosa: Any = None
    openvino: Any = None
    numpy: Any = None
    whisper_feature_extractor_cls: Any = None
    pipeline: Any = None
    ctc_model: Any = None
    ctc_processor: Any = None
    hf_asr_model: Any = None
    hf_asr_processor: Any = None
    cohere_ov_compiled_model: Any = None
    cohere_ov_marker: Optional[Dict[str, Any]] = None
    torch: Any = None
    qwen_official_model: Any = None
    qwen_official_forced_aligner: Any = None
    runtime_kind: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = None

    def ensure_common_modules(self) -> None:
        if self.openvino is not None and self.librosa is not None and self.numpy is not None:
            return
        ensure_python_packages(
            {
                "librosa": "librosa",
                "openvino": "openvino",
                "transformers": "transformers",
            },
            reason="OpenVINO ASR helper",
        )
        import librosa  # pylint: disable=import-error
        import numpy  # pylint: disable=import-error
        import openvino as ov  # pylint: disable=import-error
        from transformers import WhisperFeatureExtractor  # pylint: disable=import-error

        self.librosa = librosa
        self.numpy = numpy
        self.openvino = ov
        self.whisper_feature_extractor_cls = WhisperFeatureExtractor

    def ensure_whisper_modules(self) -> None:
        self.ensure_common_modules()
        if self.openvino_genai is not None:
            return
        ensure_python_packages(
            {
                "openvino_genai": "openvino-genai",
            },
            reason="OpenVINO Whisper helper",
        )
        import openvino_genai  # pylint: disable=import-error

        self.openvino_genai = openvino_genai

    def unload(self) -> Dict[str, Any]:
        self.pipeline = None
        self.ctc_model = None
        self.ctc_processor = None
        self.hf_asr_model = None
        self.hf_asr_processor = None
        self.cohere_ov_compiled_model = None
        self.cohere_ov_marker = None
        self.qwen_official_model = None
        self.qwen_official_forced_aligner = None
        self.runtime_kind = None
        self.model_path = None
        self.device = None
        gc.collect()
        return {"unloaded": True}

    def load(
        self,
        model_path: str,
        device: str,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_dir = pathlib.Path(model_path).resolve()
        if not model_dir.exists() or not model_dir.is_dir():
            raise RuntimeError(f"Model path does not exist: {model_dir}")

        self.unload()
        runtime_kind = _detect_runtime_kind(model_dir)
        if runtime_kind == "whisper":
            self.ensure_whisper_modules()
            whisper_kwargs: Dict[str, Any] = {
                "word_timestamps": True,
            }
            if cache_dir:
                whisper_kwargs["CACHE_DIR"] = cache_dir
            self.pipeline = self.openvino_genai.WhisperPipeline(str(model_dir), device, **whisper_kwargs)
            self.ctc_model = None
            self.ctc_processor = None
            self.qwen_official_model = None
        elif runtime_kind == "ctc_asr":
            ensure_python_packages(
                {
                    "optimum.intel": "optimum-intel[openvino]",
                },
                reason="OpenVINO CTC ASR helper",
            )
            from optimum.intel import OVModelForCTC  # pylint: disable=import-error
            from transformers import AutoProcessor  # pylint: disable=import-error

            self.ensure_common_modules()
            self.pipeline = None
            self.qwen_official_model = None
            self.hf_asr_model = None
            self.hf_asr_processor = None
            self.ctc_processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
            self.ctc_model = OVModelForCTC.from_pretrained(
                str(model_dir),
                compile=False,
                device=device,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.ctc_model.compile()
        elif runtime_kind == "cohere_openvino_asr":
            ensure_python_packages(
                {
                    "torch": "torch",
                    "soundfile": "soundfile",
                    "librosa": "librosa",
                },
                reason="OpenVINO Cohere ASR helper",
            )
            self.ensure_common_modules()
            _ensure_cohere_transformers_runtime()
            import torch  # pylint: disable=import-error
            from transformers import AutoProcessor  # pylint: disable=import-error

            marker_path = model_dir / "arcsub_cohere_asr_ov_config.json"
            try:
                self.cohere_ov_marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except Exception:
                self.cohere_ov_marker = {}

            self.pipeline = None
            self.ctc_model = None
            self.ctc_processor = None
            self.hf_asr_model = None
            self.qwen_official_model = None
            self.hf_asr_processor = AutoProcessor.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
            core = self.openvino.Core()
            self.cohere_ov_compiled_model = core.compile_model(str(model_dir / "openvino_model.xml"), device)
            self.torch = torch
        elif runtime_kind == "cohere_transformers_asr":
            ensure_python_packages(
                {
                    "torch": "torch",
                    "soundfile": "soundfile",
                    "librosa": "librosa",
                },
                reason="HF Transformers ASR helper",
            )
            self.ensure_common_modules()
            _ensure_cohere_transformers_runtime()
            import torch  # pylint: disable=import-error
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # pylint: disable=import-error

            self.pipeline = None
            self.ctc_model = None
            self.ctc_processor = None
            self.qwen_official_model = None
            requested = str(device or "").strip().lower()
            torch_device = "cuda" if requested not in ("cpu",) and torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32
            self.hf_asr_processor = AutoProcessor.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
            self.hf_asr_model = CohereAsrForConditionalGeneration.from_pretrained(
                str(model_dir),
                local_files_only=True,
                dtype=torch_dtype,
            )
            self.hf_asr_model.to(torch_device)
            self.hf_asr_model.eval()
            self.torch = torch
            device = "CPU" if torch_device == "cpu" else "CUDA"
        else:
            self.pipeline = None
            self.ctc_model = None
            self.ctc_processor = None
            self.hf_asr_model = None
            self.hf_asr_processor = None
            self.qwen_official_model = create_official_qwen3_asr_model(str(model_dir), device=device)

        self.runtime_kind = runtime_kind
        self.model_path = str(model_dir)
        self.device = device
        return {
            "loaded": True,
            "modelPath": self.model_path,
            "device": self.device,
            "cacheDir": cache_dir or None,
            "runtimeKind": self.runtime_kind,
        }

    def _read_audio(self, audio_path: str):
        self.ensure_common_modules()
        file_path = pathlib.Path(audio_path).resolve()
        if not file_path.exists():
            raise RuntimeError(f"Audio file not found: {file_path}")
        audio, _sr = self.librosa.load(str(file_path), sr=16000, mono=True)
        return audio

    def _normalize_cohere_language(self, raw: Optional[str]) -> str:
        value = str(raw or "").strip().lower().replace("_", "-")
        aliases = {
            "zh-cn": "zh",
            "zh-tw": "zh",
            "zh-hk": "zh",
            "zh-hans": "zh",
            "zh-hant": "zh",
            "cmn": "zh",
            "cmn-hans-cn": "zh",
            "yue": "zh",
            "yue-hant-hk": "zh",
            "jp": "ja",
            "ja-jp": "ja",
            "kr": "ko",
            "ko-kr": "ko",
            "pt-br": "pt",
            "pt-pt": "pt",
            "es-419": "es",
            "en-us": "en",
            "en-gb": "en",
            "ar-eg": "ar",
            "de-de": "de",
            "el-gr": "el",
            "fr-fr": "fr",
            "it-it": "it",
            "nl-nl": "nl",
            "pl-pl": "pl",
            "vi-vn": "vi",
        }
        normalized = aliases.get(value, value)
        if normalized in ("auto", "none", "null", ""):
            raise RuntimeError(
                "Cohere Transcribe requires an explicit source language. Supported languages: "
                "ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh."
            )
        supported = {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
        if normalized not in supported:
            raise RuntimeError(
                f'Cohere Transcribe does not support language "{raw}". Supported languages: '
                "ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh."
            )
        return normalized

    def _move_batch_to_model_device(self, inputs: Any) -> Any:
        if self.torch is None or self.hf_asr_model is None:
            return inputs
        device = next(self.hf_asr_model.parameters()).device
        dtype = getattr(self.hf_asr_model, "dtype", self.torch.float32)
        if hasattr(inputs, "to"):
            try:
                return inputs.to(device, dtype=dtype)
            except TypeError:
                return inputs.to(device)
        for key, value in list(inputs.items()):
            if not hasattr(value, "to"):
                continue
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)
        return inputs

    def _decode_cohere_output(self, outputs: Any, audio_chunk_index: Any, language: str) -> str:
        if self.hf_asr_processor is None:
            return ""
        try:
            decoded = self.hf_asr_processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=language,
            )
        except TypeError:
            decoded = self.hf_asr_processor.decode(outputs, skip_special_tokens=True)
        if isinstance(decoded, (list, tuple)):
            return " ".join(str(item).strip() for item in decoded if str(item).strip()).strip()
        return str(decoded or "").strip()

    def _as_numpy(self, value: Any, dtype: Any = None):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy()
        else:
            value = self.numpy.asarray(value)
        return value.astype(dtype) if dtype is not None else value

    def _cohere_join_texts(self, texts: list[str], language: str) -> str:
        parts = [str(item or "").strip() for item in texts if str(item or "").strip()]
        if not parts:
            return ""
        separator = "" if language in {"ja", "zh"} else " "
        return separator.join(parts).replace("  ", " ").strip()

    def _cohere_split_audio(self, audio: Any) -> list[tuple[float, Any]]:
        total_samples = int(len(audio)) if hasattr(audio, "__len__") else 0
        if total_samples <= 0:
            return []
        marker = self.cohere_ov_marker or {}
        try:
            max_clip_sec = float(marker.get("maxAudioClipSec") or 35.0)
        except Exception:
            max_clip_sec = 35.0
        raw_chunk_sec = str(os.environ.get("OPENVINO_COHERE_ASR_CHUNK_SEC", "")).strip()
        try:
            chunk_sec = float(raw_chunk_sec) if raw_chunk_sec else min(30.0, max_clip_sec)
        except Exception:
            chunk_sec = min(30.0, max_clip_sec)
        chunk_sec = max(1.0, min(chunk_sec, max_clip_sec))
        chunk_samples = max(1, int(round(chunk_sec * 16000)))
        chunks: list[tuple[float, Any]] = []
        for start in range(0, total_samples, chunk_samples):
            end = min(total_samples, start + chunk_samples)
            if end <= start:
                continue
            chunks.append((start / 16000.0, audio[start:end]))
        return chunks

    def _cohere_split_text_for_display(self, text: str, language: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return []
        no_space = language in {"ja", "zh"}
        max_chars = 42 if no_space else 118
        hard_max_chars = 58 if no_space else 160
        primary_pattern = r"(?<=[。！？!?])" if no_space else r"(?<=[.!?])\s+"
        raw_parts = [part.strip() for part in re.split(primary_pattern, normalized) if part.strip()]
        if not raw_parts:
            raw_parts = [normalized]

        parts: list[str] = []
        for raw in raw_parts:
            pending = raw
            while len(pending) > hard_max_chars:
                window = pending[:hard_max_chars]
                break_at = -1
                for pattern in (r"[、，,;；:：]\s*", r"\s+"):
                    matches = list(re.finditer(pattern, window))
                    matches = [match for match in matches if match.end() >= max_chars]
                    if matches:
                        break_at = matches[-1].end()
                        break
                if break_at <= 0:
                    break_at = hard_max_chars
                head = pending[:break_at].strip()
                if head:
                    parts.append(head)
                pending = pending[break_at:].strip()
            if pending:
                parts.append(pending)
        return parts or [normalized]

    def _cohere_make_display_chunks(self, text: str, start_sec: float, end_sec: float, language: str) -> list[Dict[str, Any]]:
        parts = self._cohere_split_text_for_display(text, language)
        if not parts:
            return []
        start = float(start_sec)
        end = float(end_sec)
        if not math.isfinite(end) or end <= start:
            end = start + max(0.4, len(parts) * 0.8)
        duration = max(0.4, end - start)
        weights = [max(1, len(re.sub(r"\s+", "", part))) for part in parts]
        total_weight = max(1, sum(weights))
        cursor = start
        chunks: list[Dict[str, Any]] = []
        for index, part in enumerate(parts):
            if index + 1 == len(parts):
                part_end = end
            else:
                part_end = start + duration * (sum(weights[: index + 1]) / total_weight)
            if part_end <= cursor:
                part_end = cursor + 0.4
            chunks.append(
                {
                    "start_ts": round(cursor, 3),
                    "end_ts": round(part_end, 3),
                    "text": part,
                    "source": "cohere_display_sentence",
                }
            )
            cursor = part_end
        return chunks

    def _cohere_decode_ids(self, token_ids: Any, audio_chunk_index: Any, language: str) -> str:
        if self.hf_asr_processor is None:
            return ""
        decoded = self.hf_asr_processor.decode(
            token_ids,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
        return _coerce_text_payload(decoded[0] if isinstance(decoded, list) and decoded else decoded)

    def _transcribe_cohere_openvino(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.cohere_ov_compiled_model is None or self.hf_asr_processor is None or self.numpy is None:
            raise RuntimeError("OpenVINO Cohere ASR model is not loaded.")

        audio_path = str(params.get("audioPath") or "").strip()
        if not audio_path:
            raise RuntimeError("audioPath is required.")

        language = self._normalize_cohere_language(params.get("language"))
        audio = self._read_audio(audio_path)
        started_at = time.time()
        duration_sec = float(len(audio) / 16000.0) if hasattr(audio, "__len__") else 0.0

        raw_max_new_tokens = str(
            os.environ.get("OPENVINO_COHERE_ASR_MAX_NEW_TOKENS")
            or os.environ.get("HF_TRANSFORMERS_ASR_MAX_NEW_TOKENS")
            or ""
        ).strip()
        try:
            base_max_new_tokens = int(raw_max_new_tokens) if raw_max_new_tokens else int(math.ceil(duration_sec * 8.0) + 24)
        except ValueError:
            base_max_new_tokens = int(math.ceil(duration_sec * 8.0) + 24)
        base_max_new_tokens = max(32, min(base_max_new_tokens, 256))

        eos_token_id = int(getattr(self.hf_asr_processor.tokenizer, "eos_token_id", 3) or 3)
        chunks = self._cohere_split_audio(audio)
        transcript_chunks: list[Dict[str, Any]] = []
        generated_texts: list[str] = []
        inference_ms = 0
        token_count = 0
        return_timestamps = bool(params.get("returnTimestamps", True))

        for chunk_index, (offset_sec, chunk_audio) in enumerate(chunks):
            chunk_duration_sec = float(len(chunk_audio) / 16000.0) if hasattr(chunk_audio, "__len__") else 0.0
            chunk_max_new_tokens = max(16, min(base_max_new_tokens, int(math.ceil(chunk_duration_sec * 8.0) + 24)))
            inputs = self.hf_asr_processor(
                chunk_audio,
                sampling_rate=16000,
                return_tensors="pt",
                language=language,
                punctuation=True,
            )
            audio_chunk_index = inputs.get("audio_chunk_index") if hasattr(inputs, "get") else None
            input_features = self._as_numpy(inputs["input_features"], self.numpy.float32)
            if input_features.ndim == 3 and input_features.shape[-1] == 128:
                input_features = self.numpy.transpose(input_features, (0, 2, 1)).astype(self.numpy.float32)
            attention_mask = self._as_numpy(inputs.get("attention_mask"), self.numpy.int64)
            length = attention_mask.sum(axis=1).astype(self.numpy.int64)
            decoder_input_ids = self._as_numpy(inputs["decoder_input_ids"], self.numpy.int64)

            for _step in range(chunk_max_new_tokens):
                decoder_attention_mask = self.numpy.ones_like(decoder_input_ids, dtype=self.numpy.int64)
                feed = {
                    self.cohere_ov_compiled_model.inputs[0]: input_features,
                    self.cohere_ov_compiled_model.inputs[1]: length,
                    self.cohere_ov_compiled_model.inputs[2]: decoder_input_ids,
                    self.cohere_ov_compiled_model.inputs[3]: decoder_attention_mask,
                }
                infer_started = time.time()
                result = self.cohere_ov_compiled_model(feed)
                inference_ms += int(max(0, time.time() - infer_started) * 1000)
                logits = list(result.values())[0]
                next_token_id = int(self.numpy.argmax(logits[0, -1, :]))
                decoder_input_ids = self.numpy.concatenate(
                    [decoder_input_ids, self.numpy.array([[next_token_id]], dtype=self.numpy.int64)],
                    axis=1,
                )
                token_count += 1
                if next_token_id == eos_token_id:
                    break

            text = self._cohere_decode_ids(decoder_input_ids, audio_chunk_index, language)
            if text:
                generated_texts.append(text)
                if return_timestamps:
                    transcript_chunks.extend(
                        self._cohere_make_display_chunks(text, offset_sec, offset_sec + chunk_duration_sec, language)
                    )
                else:
                    transcript_chunks.append(
                        {
                            "start_ts": offset_sec,
                            "end_ts": offset_sec + chunk_duration_sec,
                            "text": text,
                            "source": "cohere_model_window",
                        }
                    )

        text = self._cohere_join_texts(generated_texts, language)
        return {
            "text": text,
            "chunks": transcript_chunks,
            "words": [],
            "runtimeKind": self.runtime_kind,
            "modelPath": self.model_path,
            "device": self.device,
            "language": language,
            "chunking": {
                "strategy": "cohere_openvino_greedy",
                "chunkCount": len(chunks),
                "maxNewTokens": base_max_new_tokens,
                "generatedTokens": token_count,
            },
            "timing": {
                "generateMs": int(max(0, time.time() - started_at) * 1000),
                "inferenceMs": inference_ms,
                "audioDurationSec": duration_sec,
            },
        }

    def _transcribe_cohere_transformers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.hf_asr_model is None or self.hf_asr_processor is None or self.torch is None:
            raise RuntimeError("HF Transformers ASR model is not loaded.")

        audio_path = str(params.get("audioPath") or "").strip()
        if not audio_path:
            raise RuntimeError("audioPath is required.")

        language = self._normalize_cohere_language(params.get("language"))
        audio = self._read_audio(audio_path)
        started_at = time.time()

        duration_sec = float(len(audio) / 16000.0) if hasattr(audio, "__len__") else 0.0
        raw_max_new_tokens = str(os.environ.get("HF_TRANSFORMERS_ASR_MAX_NEW_TOKENS", "")).strip()
        try:
            max_new_tokens = int(raw_max_new_tokens) if raw_max_new_tokens else int(math.ceil(duration_sec * 8.0) + 24)
        except ValueError:
            max_new_tokens = int(math.ceil(duration_sec * 8.0) + 24)
        max_new_tokens = max(32, min(max_new_tokens, 256))

        inputs = self.hf_asr_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language=language,
            punctuation=True,
        )
        audio_chunk_index = inputs.get("audio_chunk_index") if hasattr(inputs, "get") else None
        inputs = self._move_batch_to_model_device(inputs)

        with self.torch.inference_mode():
            outputs = self.hf_asr_model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.hf_asr_processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
        text = _coerce_text_payload(decoded[0] if isinstance(decoded, list) and decoded else decoded)
        if bool(params.get("returnTimestamps", True)):
            chunks = self._cohere_make_display_chunks(text, 0.0, duration_sec, language)
        else:
            chunk = {"start_ts": 0.0, "text": text, "source": "cohere_model_window"}
            if duration_sec > 0:
                chunk["end_ts"] = duration_sec
            chunks = [chunk] if text else []
        return {
            "text": text,
            "chunks": chunks,
            "words": [],
            "runtimeKind": self.runtime_kind,
            "modelPath": self.model_path,
            "device": self.device,
            "language": language,
            "chunking": {
                "strategy": "cohere_builtin_generate",
                "maxNewTokens": max_new_tokens,
                "audioChunkIndex": bool(audio_chunk_index is not None),
            },
            "timing": {
                "generateMs": int(max(0, time.time() - started_at) * 1000),
                "audioDurationSec": duration_sec,
            },
        }

    def _transcribe_whisper(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Whisper model is not loaded.")

        audio_path = str(params.get("audioPath") or "").strip()
        if not audio_path:
            raise RuntimeError("audioPath is required.")

        audio = self._read_audio(audio_path).tolist()
        kwargs: Dict[str, Any] = {
            "task": str(params.get("task") or "transcribe").strip() or "transcribe",
            "return_timestamps": bool(params.get("returnTimestamps", True)),
            "word_timestamps": bool(params.get("wordTimestamps", False)),
        }
        language = _normalize_whisper_language(str(params.get("language") or "").strip())
        prompt = str(params.get("prompt") or "").strip()
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["initial_prompt"] = prompt

        result = self.pipeline.generate(audio, **kwargs)
        text = _coerce_text_payload(
            _read_field(result, "texts", "text", "transcript", "transcription", "output_text", "outputText")
        )
        chunks = []
        chunk_texts = []
        for chunk in (_read_field(result, "chunks", "segments", "results") or []):
            item_text = _coerce_text_payload(
                _read_field(chunk, "text", "transcript", "transcription", "output_text", "outputText")
            )
            if not item_text:
                continue
            chunk_texts.append(item_text)
            start_ts = _coerce_float_payload(_read_field(chunk, "start_ts", "start", "startTs", "start_time", "startTime"))
            end_ts = _coerce_float_payload(_read_field(chunk, "end_ts", "end", "endTs", "end_time", "endTime"))
            if start_ts is None:
                continue
            chunks.append(
                {
                    "start_ts": float(start_ts),
                    "end_ts": float(end_ts) if end_ts is not None else 0.0,
                    "text": item_text,
                }
            )

        words = []
        for word in (_read_field(result, "words", "tokens") or []):
            item_text = _coerce_text_payload(_read_field(word, "word", "text", "token"))
            if not item_text:
                continue
            start_ts = _coerce_float_payload(_read_field(word, "start_ts", "start", "startTs", "start_time", "startTime"))
            end_ts = _coerce_float_payload(_read_field(word, "end_ts", "end", "endTs", "end_time", "endTime"))
            if start_ts is None:
                continue
            words.append(
                {
                    "word": item_text,
                    "start_ts": float(start_ts),
                    "end_ts": float(end_ts) if end_ts is not None else 0.0,
                }
            )

        if not text and chunk_texts:
            text = " ".join(chunk_texts).strip()

        return {
            "text": text,
            "chunks": chunks,
            "words": words,
            "runtimeKind": self.runtime_kind,
            "modelPath": self.model_path,
            "device": self.device,
        }

    def _normalize_qwen_language(self, raw: Optional[str]) -> Optional[str]:
        value = str(raw or "").strip()
        if not value:
            return None

        lower = value.lower().replace("_", "-")
        if lower in ("auto", "none", "null"):
            return None

        aliases = {
            "ar": "Arabic",
            "cs": "Czech",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "es": "Spanish",
            "fa": "Persian",
            "fi": "Finnish",
            "fil": "Filipino",
            "fr": "French",
            "hi": "Hindi",
            "hu": "Hungarian",
            "id": "Indonesian",
            "it": "Italian",
            "ja": "Japanese",
            "jp": "Japanese",
            "ko": "Korean",
            "kr": "Korean",
            "mk": "Macedonian",
            "ms": "Malay",
            "nl": "Dutch",
            "pl": "Polish",
            "pt": "Portuguese",
            "ro": "Romanian",
            "ru": "Russian",
            "sv": "Swedish",
            "th": "Thai",
            "tr": "Turkish",
            "vi": "Vietnamese",
            "yue": "Cantonese",
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh-hans": "Chinese",
            "zh-hant": "Chinese",
            "zh-sg": "Chinese",
            "zh-tw": "Chinese",
            "zh-hk": "Cantonese",
            "zh-mo": "Cantonese",
        }
        return aliases.get(lower, value[:1].upper() + value[1:].lower())

    def _resolve_qwen_official_chunk_seconds(self) -> int:
        raw = str(os.environ.get("OPENVINO_QWEN3_ASR_MAX_CHUNK_SEC", "")).strip()
        try:
            value = int(raw) if raw else 30
        except Exception:
            value = 30
        return max(15, min(value, 1200))

    def _split_qwen_official_audio(self, audio: Any, sample_rate: int, max_chunk_sec: int) -> list[tuple[Any, float]]:
        chunk_samples = max(sample_rate, int(sample_rate * max_chunk_sec))
        chunks: list[tuple[Any, float]] = []
        for offset in range(0, len(audio), chunk_samples):
            end = min(len(audio), offset + chunk_samples)
            chunk = audio[offset:end]
            if len(chunk) == 0:
                continue
            chunks.append((chunk, offset / float(sample_rate)))
        return chunks

    def _should_run_qwen_official_forced_alignment(self, params: Dict[str, Any]) -> bool:
        return self.runtime_kind == "qwen3_asr_official" and bool(params.get("wordTimestamps"))

    def _resolve_ctc_chunk_seconds(self) -> int:
        raw = str(os.environ.get("OPENVINO_CTC_ASR_MAX_CHUNK_SEC", "")).strip()
        try:
            value = int(raw) if raw else 30
        except Exception:
            value = 30
        return max(5, min(value, 120))

    def _split_ctc_audio(self, audio: Any, sample_rate: int, max_chunk_sec: int) -> list[tuple[Any, float]]:
        chunk_samples = max(sample_rate, int(sample_rate * max_chunk_sec))
        chunks: list[tuple[Any, float]] = []
        for offset in range(0, len(audio), chunk_samples):
            end = min(len(audio), offset + chunk_samples)
            chunk = audio[offset:end]
            if len(chunk) == 0:
                continue
            chunks.append((chunk, offset / float(sample_rate)))
        return chunks

    def _to_numpy(self, value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return self.numpy.asarray(value)

    def _normalize_ctc_text(self, text: str) -> str:
        normalized = re.sub(r"<unk>", "", str(text or ""), flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _transcribe_ctc(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.ctc_model is None or self.ctc_processor is None:
            raise RuntimeError("CTC ASR model is not loaded.")

        audio_path = str(params.get("audioPath") or "").strip()
        if not audio_path:
            raise RuntimeError("audioPath is required.")

        audio = self._read_audio(audio_path)
        max_chunk_sec = self._resolve_ctc_chunk_seconds()
        chunk_inputs = self._split_ctc_audio(audio, 16000, max_chunk_sec)
        return_timestamps = bool(params.get("returnTimestamps", True))
        chunks = []
        decode_started_at = time.perf_counter()

        for chunk_audio, offset_sec in chunk_inputs:
            inputs = self.ctc_processor(chunk_audio, sampling_rate=16000, return_tensors="pt")
            outputs = self.ctc_model(**inputs)
            logits = self._to_numpy(outputs.logits)
            pred_ids = self.numpy.argmax(logits, axis=-1)
            decoded = self.ctc_processor.batch_decode(pred_ids, skip_special_tokens=True)
            item_text = self._normalize_ctc_text(decoded[0] if decoded else "")
            if not item_text:
                continue
            chunks.append(
                {
                    "start_ts": round(offset_sec, 3),
                    "end_ts": round(offset_sec + (len(chunk_audio) / 16000.0), 3),
                    "text": item_text,
                }
            )

        if not chunks:
            raise RuntimeError("Local CTC ASR model returned empty transcription.")

        decode_ms = round((time.perf_counter() - decode_started_at) * 1000.0, 3)
        if not return_timestamps:
            chunks = [
                {
                    "start_ts": 0.0,
                    "end_ts": round(len(audio) / 16000.0, 3),
                    "text": " ".join(item["text"] for item in chunks).strip(),
                }
            ]

        return {
            "text": " ".join(item["text"] for item in chunks).strip(),
            "chunks": chunks,
            "words": [],
            "runtimeKind": self.runtime_kind,
            "modelPath": self.model_path,
            "device": self.device,
            "chunking": {
                "strategy": "fixed_window",
                "maxChunkSec": max_chunk_sec,
                "chunkCount": len(chunks),
            },
            "timing": {
                "ctcDecodeMs": decode_ms,
            },
        }

    def _get_qwen_official_forced_aligner(self) -> Any:
        if self.qwen_official_forced_aligner is None:
            self.qwen_official_forced_aligner = create_official_qwen3_forced_aligner()
        return self.qwen_official_forced_aligner

    def _align_qwen_official_chunks(
        self,
        chunk_inputs: list[tuple[Any, float]],
        chunk_texts: list[str],
        chunk_languages: list[Optional[str]],
    ) -> Dict[str, Any]:
        aligner = self._get_qwen_official_forced_aligner()
        effective_languages = [lang or "English" for lang in chunk_languages]
        words = []
        errors = []
        aligned_chunk_count = 0

        for index, ((chunk_audio, offset_sec), chunk_text, language) in enumerate(
            zip(chunk_inputs, chunk_texts, effective_languages)
        ):
            try:
                results = aligner.align(
                    audio=[(chunk_audio, 16000)],
                    text=[chunk_text],
                    language=[language],
                )
            except Exception as error:
                errors.append({"chunk": index, "error": str(error)})
                continue
            finally:
                del chunk_audio

            aligned_items = getattr((results or [None])[0], "items", []) or []
            if aligned_items:
                aligned_chunk_count += 1
            for item in aligned_items:
                text = str(getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                start_time = float(getattr(item, "start_time", 0.0) or 0.0)
                end_time = float(getattr(item, "end_time", 0.0) or 0.0)
                words.append(
                    {
                        "text": text,
                        "start_ts": round(offset_sec + start_time, 3),
                        "end_ts": round(offset_sec + end_time, 3) if end_time > start_time else None,
                    }
                )
        return {
            "words": words,
            "backend": "qwen3-forced-aligner",
            "modelId": str(
                os.environ.get("OPENVINO_QWEN3_ASR_FORCED_ALIGNER_REPO") or "Qwen/Qwen3-ForcedAligner-0.6B"
            ).strip(),
            "language": effective_languages[0] if effective_languages else "",
            "alignedChunkCount": aligned_chunk_count,
            "skippedChunkCount": len(errors),
            "errors": errors[:5],
        }

    def _transcribe_qwen_official(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.qwen_official_model is None:
            raise RuntimeError("Official Qwen3-ASR model is not loaded.")

        audio_path = str(params.get("audioPath") or "").strip()
        if not audio_path:
            raise RuntimeError("audioPath is required.")

        audio = self._read_audio(audio_path)
        forced_language = self._normalize_qwen_language(params.get("language"))
        context = str(params.get("prompt") or "").strip()
        max_chunk_sec = self._resolve_qwen_official_chunk_seconds()
        chunk_inputs = self._split_qwen_official_audio(audio, 16000, max_chunk_sec)
        results = self.qwen_official_model.transcribe(
            audio=[(chunk_audio, 16000) for chunk_audio, _ in chunk_inputs],
            context=context,
            language=forced_language,
            return_time_stamps=False,
        )
        chunks = []
        aligned_chunk_inputs = []
        chunk_texts = []
        chunk_languages = []
        for (chunk_audio, offset_sec), item in zip(chunk_inputs, results or []):
            item_text = str(getattr(item, "text", "") or "").strip()
            if "<asr_text>" in item_text:
                item_text = item_text.split("<asr_text>", 1)[1].strip()
            item_text = re.sub(
                r"^(?:system\s*user\s*assistant\s*)+",
                "",
                item_text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            if not item_text:
                continue
            aligned_chunk_inputs.append((chunk_audio, offset_sec))
            chunks.append(
                {
                    "start_ts": round(offset_sec, 3),
                    "end_ts": round(offset_sec + (len(chunk_audio) / 16000.0), 3),
                    "text": item_text,
                }
            )
            chunk_texts.append(item_text)
            chunk_languages.append(self._normalize_qwen_language(getattr(item, "language", None)) or forced_language)
        if not chunks:
            raise RuntimeError("Local Qwen3-ASR model returned empty transcription.")

        first = results[0] if isinstance(results, list) and results else None
        text = " ".join(chunk["text"] for chunk in chunks).strip()
        words = []
        forced_alignment: Dict[str, Any] | None = None
        if self._should_run_qwen_official_forced_alignment(params):
            try:
                aligned = self._align_qwen_official_chunks(aligned_chunk_inputs, chunk_texts, chunk_languages)
                words = aligned["words"]
                forced_alignment = {
                    "backend": aligned["backend"],
                    "modelId": aligned["modelId"],
                    "language": aligned["language"],
                    "applied": len(words) > 0,
                    "alignedChunkCount": aligned.get("alignedChunkCount", 0),
                    "skippedChunkCount": aligned.get("skippedChunkCount", 0),
                    "errors": aligned.get("errors", []),
                }
            except Exception as error:
                forced_alignment = {
                    "backend": "qwen3-forced-aligner",
                    "modelId": str(
                        os.environ.get("OPENVINO_QWEN3_ASR_FORCED_ALIGNER_REPO") or "Qwen/Qwen3-ForcedAligner-0.6B"
                    ).strip(),
                    "language": forced_language or str(getattr(first, "language", "") or "").strip(),
                    "applied": False,
                    "error": str(error),
                }
        return {
            "text": text,
            "chunks": chunks,
            "words": words,
            "language": str(getattr(first, "language", "") or "").strip(),
            "runtimeKind": self.runtime_kind,
            "modelPath": self.model_path,
            "device": self.device,
            "chunking": {
                "strategy": "fixed_window",
                "maxChunkSec": max_chunk_sec,
                "chunkCount": len(chunks),
            },
            "forcedAlignment": forced_alignment,
        }

    def transcribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.runtime_kind == "whisper":
            return self._transcribe_whisper(params)
        if self.runtime_kind == "ctc_asr":
            return self._transcribe_ctc(params)
        if self.runtime_kind == "qwen3_asr_official":
            return self._transcribe_qwen_official(params)
        if self.runtime_kind == "cohere_openvino_asr":
            return self._transcribe_cohere_openvino(params)
        if self.runtime_kind == "cohere_transformers_asr":
            return self._transcribe_cohere_transformers(params)
        raise RuntimeError("ASR model is not loaded.")

    def health(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "modelLoaded": (
                self.pipeline is not None
                or self.ctc_model is not None
                or self.qwen_official_model is not None
                or self.cohere_ov_compiled_model is not None
                or self.hf_asr_model is not None
            ),
            "modelPath": self.model_path,
            "device": self.device,
            "runtimeKind": self.runtime_kind,
            "env": self.env_info,
            "pythonVersion": sys.version,
        }


def _write_response(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _handle_request(state: RuntimeState, request: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(request.get("requestId") or "")
    method = str(request.get("method") or "").strip()
    params = request.get("params") or {}
    if not isinstance(params, dict):
        raise RuntimeError("params must be an object.")
    if not request_id:
        raise RuntimeError("requestId is required.")
    if not method:
        raise RuntimeError("method is required.")

    if method == "health":
        return {"requestId": request_id, "ok": True, "result": state.health()}
    if method == "load":
        model_path = str(params.get("modelPath") or "").strip()
        device = str(params.get("device") or "AUTO").strip() or "AUTO"
        cache_dir = str(params.get("cacheDir") or "").strip() or None
        if not model_path:
            raise RuntimeError("modelPath is required.")
        return {"requestId": request_id, "ok": True, "result": state.load(model_path, device, cache_dir)}
    if method == "transcribe":
        return {"requestId": request_id, "ok": True, "result": state.transcribe(params)}
    if method == "convertOfficialQwen3Asr":
        repo_id = str(params.get("repoId") or "").strip()
        output_dir = str(params.get("outputDir") or "").strip()
        use_local_dir = bool(params.get("useLocalDir", False))
        if not repo_id:
            raise RuntimeError("repoId is required.")
        if not output_dir:
            raise RuntimeError("outputDir is required.")
        ensure_official_support_sources()
        return {
            "requestId": request_id,
            "ok": True,
            "result": convert_official_qwen3_asr_model(repo_id, output_dir, use_local_dir=use_local_dir),
        }
    if method == "unload":
        return {"requestId": request_id, "ok": True, "result": state.unload()}
    if method == "shutdown":
        _write_response({"requestId": request_id, "ok": True, "result": {"shuttingDown": True}})
        raise SystemExit(0)
    raise RuntimeError(f"Unknown method: {method}")


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    state = RuntimeState(env_info=prepare_openvino_env())
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RuntimeError("Request payload must be an object.")
            _write_response(_handle_request(state, request))
        except SystemExit:
            return 0
        except Exception as error:
            request_id = ""
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    request_id = str(parsed.get("requestId") or "")
            except Exception:
                request_id = ""

            message = str(error) or error.__class__.__name__
            if str(os.environ.get("ARC_SUB_HELPER_DEBUG", "")).lower() in ("1", "true", "yes"):
                message = f"{message}\n{traceback.format_exc()}"
            _write_response({"requestId": request_id, "ok": False, "error": message})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
