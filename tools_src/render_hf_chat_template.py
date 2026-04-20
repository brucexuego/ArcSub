#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from python_runtime_bootstrap import ensure_python_packages

# This helper only needs tokenizer/processor chat template support.
# Avoid importing heavy ML backends like torch when the host Python has partial GPU installs.
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


ensure_python_packages(
    {
        "transformers": "transformers",
        "sentencepiece": "sentencepiece",
    },
    reason="Hugging Face chat template renderer",
)

from transformers import AutoProcessor, AutoTokenizer  # type: ignore


def can_use_tokenizer_only(messages: list[object]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or len(content) != 1:
            return False
        entry = content[0]
        if not isinstance(entry, dict):
            return False
        if str(entry.get("type") or "").strip().lower() != "text":
            return False
        if "text" not in entry:
            return False
    return True


def main() -> int:
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    model_dir = pathlib.Path(str(payload.get("modelDir") or "")).resolve()
    messages = payload.get("messages")
    if not model_dir.is_dir():
        raise RuntimeError(f"Model directory not found: {model_dir}")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("messages must be a non-empty array")

    prompt: str | None = None

    if can_use_tokenizer_only(messages):
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not expose apply_chat_template.")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        try:
            processor = AutoProcessor.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            if hasattr(processor, "apply_chat_template"):
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = None

        if not prompt:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            if not hasattr(tokenizer, "apply_chat_template"):
                raise RuntimeError("Tokenizer does not expose apply_chat_template.")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sys.stdout.write(json.dumps({"prompt": prompt}, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
