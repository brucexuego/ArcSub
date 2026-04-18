#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openvino_asr_env import prepare_openvino_env
from python_runtime_bootstrap import ensure_python_packages


prepare_openvino_env()
ensure_python_packages(
    {
        "optimum.intel": "optimum-intel[openvino]",
        "transformers": "transformers",
        "torch": "torch",
    },
    reason="OpenVINO translation helper",
)

from optimum.intel import OVModelForSeq2SeqLM  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import torch  # type: ignore


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


class RuntimeState:
    def __init__(self, env_info: Dict[str, Any]):
        self.env_info = env_info
        self.model = None
        self.tokenizer = None
        self.model_path = ""
        self.device = "AUTO"
        self.runtime_kind = "seq2seq_translate"

    def unload(self) -> Dict[str, Any]:
        self.model = None
        self.tokenizer = None
        self.model_path = ""
        self.device = "AUTO"
        return {"unloaded": True}

    def load(self, model_path: str, device: str) -> Dict[str, Any]:
        self.unload()
        resolved_path = str(Path(model_path).resolve())
        trust_remote_code = _bool_env("OPENVINO_HF_EXPORT_TRUST_REMOTE_CODE", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        self.model = OVModelForSeq2SeqLM.from_pretrained(
            resolved_path,
            device=device,
            export=False,
            compile=False,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        compile_method = getattr(self.model, "compile", None)
        if callable(compile_method):
            compile_method()

        self.model_path = resolved_path
        self.device = str(device or "AUTO").strip() or "AUTO"
        return {
            "loaded": True,
            "modelPath": self.model_path,
            "device": self.device,
            "runtimeKind": self.runtime_kind,
        }

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Seq2Seq translation model is not loaded.")

        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("prompt is required.")

        generation_config = params.get("generationConfig") or {}
        if not isinstance(generation_config, dict):
            raise RuntimeError("generationConfig must be an object.")

        tokenizer_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(max_length, int) and 0 < max_length < 10_000_000:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max_length

        encoded = self.tokenizer(prompt, **tokenizer_kwargs)

        generate_kwargs: Dict[str, Any] = {}
        for key in (
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "num_beams",
        ):
            value = generation_config.get(key)
            if value is None:
                continue
            generate_kwargs[key] = value

        if not bool(generate_kwargs.get("do_sample", False)):
            generate_kwargs.pop("temperature", None)
            generate_kwargs.pop("top_p", None)
            generate_kwargs.pop("top_k", None)

        started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generate_kwargs)
        elapsed_ms = max(0, round((time.perf_counter() - started_at) * 1000))

        texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = str(texts[0] if texts else "").strip()
        return {
            "texts": [text] if text else [],
            "text": text,
            "device": self.device,
            "runtimeKind": self.runtime_kind,
            "timing": {"providerMs": elapsed_ms},
        }

    def health(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "modelLoaded": self.model is not None and self.tokenizer is not None,
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
        if not model_path:
            raise RuntimeError("modelPath is required.")
        return {"requestId": request_id, "ok": True, "result": state.load(model_path, device)}
    if method == "generate":
        return {"requestId": request_id, "ok": True, "result": state.generate(params)}
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
