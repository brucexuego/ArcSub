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
import json
import os
import pathlib
import re
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
            self.ctc_processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
            self.ctc_model = OVModelForCTC.from_pretrained(
                str(model_dir),
                compile=False,
                device=device,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.ctc_model.compile()
        else:
            self.pipeline = None
            self.ctc_model = None
            self.ctc_processor = None
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
        raise RuntimeError("ASR model is not loaded.")

    def health(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "modelLoaded": self.pipeline is not None or self.ctc_model is not None or self.qwen_official_model is not None,
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
