from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.linalg import eigh


def l2_norm(vec_or_matrix: np.ndarray) -> np.ndarray:
    if len(vec_or_matrix.shape) == 1:
        norm = np.linalg.norm(vec_or_matrix)
        return vec_or_matrix if norm == 0 else vec_or_matrix / norm
    if len(vec_or_matrix.shape) == 2:
        norms = np.linalg.norm(vec_or_matrix, axis=1, ord=2, keepdims=True)
        norms[norms == 0] = 1.0
        return vec_or_matrix / norms
    raise ValueError(f"Unsupported ndim={len(vec_or_matrix.shape)}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--transform-npz", required=True)
    parser.add_argument("--plda-npz", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    x = np.load(args.transform_npz)
    mean1 = x["mean1"]
    mean2 = x["mean2"]
    lda = x["lda"]

    p = np.load(args.plda_npz)
    plda_mu = p["mu"]
    plda_tr = p["tr"]
    plda_psi = p["psi"]

    w = np.linalg.inv(plda_tr.T.dot(plda_tr))
    b = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(b, w)
    plda_phi = acvar[::-1]
    plda_tr_reordered = wccn.T[::-1]

    payload = {
        "mean1": mean1.tolist(),
        "mean2": mean2.tolist(),
        "lda": lda.tolist(),
        "plda_mu": plda_mu.tolist(),
        "plda_tr": plda_tr_reordered.tolist(),
        "phi": plda_phi.tolist(),
        "fa": 0.07,
        "fb": 0.8,
        "threshold": 0.6,
        "max_iters": 20,
        "init_smoothing": 7.0,
        "lda_dimension": int(lda.shape[1]),
    }

    output_path = pathlib.Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Prepared VBx params -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
