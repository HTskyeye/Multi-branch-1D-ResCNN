from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from spectra_dl.data.dataset import SpectrumDataset
from spectra_dl.models.cnn import SpectralCNNIndependent


def _limit_cpu_threads(n: int = 1) -> None:
    n = max(1, int(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate(cfg: Dict[str, Any], ckpt_path: Path) -> Dict[str, float]:
    _limit_cpu_threads(int(cfg.get("cpu_threads", 1)))
    data_cfg = cfg.get("data", {})
    device = _resolve_device(str(cfg.get("device", "auto")))

    ds = SpectrumDataset(
        ai_path=Path(data_cfg["ai_path"]),
        spectrum_path=Path(data_cfg["spectrum_path"]),
        ai_sheet=data_cfg.get("ai_sheet", 0),
        spectrum_sheets=data_cfg.get("spectrum_sheets", ["ch1", "ch2", "ch3"]),
        strict=True,
    )

    model = SpectralCNNIndependent(n_channels=3, in_len=ds.info.num_features, out_len=ds.info.num_points)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=32, shuffle=False)
    criterion_l1 = nn.L1Loss(reduction="mean")
    criterion_mse = nn.MSELoss(reduction="mean")

    l1s, mses = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            l1s.append(float(criterion_l1(pred, yb).item()))
            mses.append(float(criterion_mse(pred, yb).item()))

    return {"l1": float(np.mean(l1s)), "mse": float(np.mean(mses))}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the full dataset.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics = evaluate(cfg, args.ckpt)
    print(metrics)


if __name__ == "__main__":
    main()
