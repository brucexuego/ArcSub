from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch
from torch import nn
from pyannote.audio import Model


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subfolder", required=True, choices=["segmentation", "embedding"])
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Model.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=args.token,
        subfolder=args.subfolder,
    ).eval()

    if args.subfolder == "segmentation":
        dummy_input = torch.zeros(1, 1, 160000)
        export_model = model
        input_names = ["input_values"]
        output_names = ["logits"]
    else:
        class ResNetEmbeddingExport(nn.Module):
            def __init__(self, embedding_model: nn.Module):
                super().__init__()
                self.resnet = embedding_model.resnet

            def forward(self, fbank: torch.Tensor) -> torch.Tensor:
                return self.resnet(fbank)[1]

        dummy_input = torch.zeros(1, 298, 80)
        export_model = ResNetEmbeddingExport(model).eval()
        input_names = ["fbank"]
        output_names = ["embeddings"]

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path.as_posix(),
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
    )

    print(f"Exported {args.subfolder} -> {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
