#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_asr_official_support import convert_model


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-local-dir", action="store_true")
    parser.add_argument("--compression", default=None)
    args = parser.parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = convert_model(
                args.repo_id,
                args.output_dir,
                use_local_dir=args.use_local_dir,
                compression_mode=args.compression,
            )
        sys.stdout.write(json.dumps(result, ensure_ascii=True))
        sys.stdout.flush()
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.write(
            json.dumps(
                {
                    "converted": False,
                    "error": str(error) or error.__class__.__name__,
                },
                ensure_ascii=True,
            )
        )
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
