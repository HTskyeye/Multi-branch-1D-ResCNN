from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import yaml

from spectra_dl.data.dataset import SpectrumDataset
from spectra_dl.models.cnn import SpectralCNNIndependent
from spectra_dl.utils.seed import set_seed


def _limit_cpu_threads(n: int = 1) -> None:
    """Avoid over-threading issues on some CPU environments.

    In some sandbox/CI setups, default OpenMP/MKL thread settings can cause
    hangs or very slow performance. We default to 1 thread unless the user
    explicitly overrides it.
    """
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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mean_loss_over_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _save_test_predictions(
    model: nn.Module,
    ds: Subset,
    out_xlsx: Path,
    device: torch.device,
    epoch: int,
) -> None:
    """Save per-sample prediction vs ground truth into one workbook.

    Each sheet = one sample. Rows = 0..(P-1), columns = true/pred for each channel.
    """
    model.eval()
    rows = []
    # materialize subset indices for naming
    subset_indices = getattr(ds, "indices", list(range(len(ds))))

    with pd.ExcelWriter(out_xlsx) as writer:
        with torch.no_grad():
            for local_i in range(len(ds)):
                x, y = ds[local_i]
                x = x.unsqueeze(0).to(device)
                pred = model(x).squeeze(0).cpu().numpy()  # (3,P)
                y_np = y.numpy()  # (3,P)
                P = y_np.shape[1]

                df = pd.DataFrame(
                    {
                        "true_ch1": y_np[0],
                        "pred_ch1": pred[0],
                        "true_ch2": y_np[1],
                        "pred_ch2": pred[1],
                        "true_ch3": y_np[2],
                        "pred_ch3": pred[2],
                    }
                )
                sheet_name = f"sample_{subset_indices[local_i]}"
                df.to_excel(writer, sheet_name=sheet_name[:31], index_label="idx")

        meta = pd.DataFrame({"epoch": [epoch], "note": ["Columns are (true/pred) for each channel; rows are spectral point index."]})
        meta.to_excel(writer, sheet_name="meta", index=False)


def train_from_config(cfg: Dict[str, Any]) -> Path:
    # ---- config ----
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    run_cfg = cfg.get("run", {})

    ai_path = Path(data_cfg["ai_path"])
    spectrum_path = Path(data_cfg["spectrum_path"])
    ai_sheet = data_cfg.get("ai_sheet", 0)
    spectrum_sheets = data_cfg.get("spectrum_sheets", ["ch1", "ch2", "ch3"])

    seed = int(train_cfg.get("seed", 42))
    _limit_cpu_threads(int(train_cfg.get("cpu_threads", 1)))
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    epochs = int(train_cfg.get("epochs", 300))
    lr = float(train_cfg.get("lr", 1e-3))
    batch_size = int(train_cfg.get("batch_size", 32))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    test_indices = train_cfg.get("test_indices", None)
    test_ratio = float(train_cfg.get("test_ratio", 0.2))
    save_epochs = train_cfg.get("save_epochs", [1, 10, 30, epochs])

    exp_name = str(run_cfg.get("exp_name", "exp"))
    out_dir = Path(run_cfg.get("out_dir", "runs"))
    run_path = _ensure_dir(out_dir / exp_name)

    # ---- reproducibility ----
    set_seed(seed)

    # ---- dataset ----
    ds_full = SpectrumDataset(
        ai_path=ai_path,
        spectrum_path=spectrum_path,
        ai_sheet=ai_sheet,
        spectrum_sheets=spectrum_sheets,
        strict=True,
    )

    N = len(ds_full)
    all_idx = np.arange(N)

    if test_indices is not None:
        test_idx = np.array([int(i) for i in test_indices], dtype=int)
        train_idx = np.array([i for i in all_idx if i not in set(test_idx.tolist())], dtype=int)
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(all_idx)
        n_test = max(1, int(round(N * test_ratio)))
        test_idx = all_idx[:n_test]
        train_idx = all_idx[n_test:]

    # save split for reproducibility
    (run_path / "split.json").write_text(
        json.dumps({"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}, indent=2),
        encoding="utf-8",
    )

    train_ds = Subset(ds_full, train_idx.tolist())
    test_ds = Subset(ds_full, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)), shuffle=False)

    # ---- model ----
    model = SpectralCNNIndependent(n_channels=3, in_len=ds_full.info.num_features, out_len=ds_full.info.num_points).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Save resolved config
    (run_path / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # ---- training loop ----
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        test_loss = _mean_loss_over_loader(model, test_loader, criterion, device)

        history.append({"epoch": float(epoch), "train_loss": train_loss, "test_loss": test_loss})

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Train L1: {train_loss:.6f} | Test L1: {test_loss:.6f}")

        if epoch in save_epochs or epoch == epochs:
            # checkpoint
            ckpt_path = run_path / f"model_epoch_{epoch}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "info": ds_full.info.__dict__}, ckpt_path)

            # predictions on test set
            pred_path = run_path / f"test_predictions_epoch_{epoch}.xlsx"
            _save_test_predictions(model, test_ds, pred_path, device=device, epoch=epoch)

    # final checkpoint symlinks / copies
    final_ckpt = run_path / "model_final.pt"
    torch.save({"epoch": epochs, "model_state": model.state_dict(), "info": ds_full.info.__dict__}, final_ckpt)

    # save loss history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_path / "loss_history.csv", index=False)
    hist_df.to_excel(run_path / "loss_history.xlsx", index=False)

    return run_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train 3-channel spectral regression model.")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_path = train_from_config(cfg)
    print(f"[OK] run outputs saved to: {run_path}")


if __name__ == "__main__":
    main()
