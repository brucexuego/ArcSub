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

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

ensure_python_packages(
    {
        "transformers": "transformers",
        "sentencepiece": "sentencepiece",
    },
    reason="Hugging Face input token counter",
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


def count_entry(tokenizer, processor, entry: dict[str, object]) -> int:
    messages = entry.get("messages")
    prompt = entry.get("prompt")

    if isinstance(messages, list) and messages:
        if can_use_tokenizer_only(messages):
            if not hasattr(tokenizer, "apply_chat_template"):
                raise RuntimeError("Tokenizer does not expose apply_chat_template.")
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return len(token_ids or [])

        if processor is not None and hasattr(processor, "apply_chat_template"):
            token_ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return len(token_ids or [])

        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not expose apply_chat_template.")
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(token_ids or [])

    if isinstance(prompt, str) and prompt:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        return len(input_ids or [])

    raise RuntimeError("Each entry must contain either a non-empty prompt or a non-empty messages array.")


def main() -> int:
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    model_dir = pathlib.Path(str(payload.get("modelDir") or "")).resolve()
    entries = payload.get("entries")
    if not model_dir.is_dir():
        raise RuntimeError(f"Model directory not found: {model_dir}")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("entries must be a non-empty array")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception:
        processor = None

    counts: list[int] = []
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("Each entry must be an object.")
        counts.append(count_entry(tokenizer, processor, item))

    sys.stdout.write(json.dumps({"counts": counts}, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
