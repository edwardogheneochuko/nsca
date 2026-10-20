"""STRUCTURAL test fixtures only. Disease and symptom names here are meaningless placeholders
(disease_a, sym_01, ...). They exist to exercise the code paths, are never shipped as data,
and say nothing about real diseases."""
from pathlib import Path

import numpy as np

N_SYMPTOMS = 18
CORES = {
    "disease_a": [0, 1, 2, 3, 4],
    "disease_b": [5, 6, 7, 8, 9],
    "disease_c": [10, 11, 12, 13, 14],
}


def write_synthetic_csv(path: Path, per_disease: int = 70, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    lines = ["Disease," + ",".join(f"Symptom_{i}" for i in range(1, 8))]
    for disease, core in CORES.items():
        for _ in range(per_disease):
            syms = {c for c in core if rng.random() < 0.9}
            syms |= {int(k) for k in rng.choice(N_SYMPTOMS, size=rng.integers(0, 2), replace=False)}
            names = [f"sym_{k:02d}" for k in sorted(syms)][:7]
            lines.append(disease + "," + ",".join(names + [""] * (7 - len(names))))
    path.write_text("\n".join(lines) + "\n")
    return path
