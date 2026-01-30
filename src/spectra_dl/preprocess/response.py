"""Preprocess raw response workbook -> model input features.

Input:
  - Excel (.xls/.xlsx) with multiple sheets. Each sheet has at least 4 columns.
  - We extract the 4th column (index 3) from each sheet.

Processing (kept consistent with the original project):
  1) Compute mean of first 200 points, and min of first 1000 points.
  2) Find the first index i>=200 such that
       |data[i] - mean_first_200| > 0.2 * |data[i] - min_value|
  3) Take a window of length 320 starting from i; if not enough points, pad by repeating the last value.
  4) Standardize (z-score) within the window, then shift to be nonnegative, apply log(1+x).

Output:
  - Excel (.xlsx) where each column is a sample (sheet name), each row is a feature index (length=320).
  - Also writes a 'meta' sheet with extraction indices for reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _process_one_series(data: np.ndarray, window: int = 320) -> Tuple[np.ndarray, Dict[str, float]]:
    data = np.asarray(data, dtype=float)

    # Basic stats
    first_200 = data[:200]
    first_1000 = data[: min(1000, len(data))]
    mean_first_200 = float(np.nanmean(first_200))
    min_value = float(np.nanmin(first_1000))

    # Find critical index i
    i = None
    for idx in range(200, len(data)):
        if abs(data[idx] - mean_first_200) > 0.2 * abs(data[idx] - min_value):
            i = idx
            break
    if i is None:
        raise ValueError("No valid critical point found in this sheet.")

    # Slice + pad
    end_idx = i + window
    if end_idx <= len(data):
        data_vec = data[i:end_idx]
    else:
        data_vec = data[i:].copy()
        if len(data_vec) == 0:
            raise ValueError("Empty slice after critical point.")
        pad_val = float(data_vec[-1])
        pad_len = end_idx - len(data)
        data_vec = np.concatenate([data_vec, np.full(pad_len, pad_val, dtype=float)], axis=0)

    # Normalize
    data_vec = data_vec - mean_first_200
    normalized = StandardScaler().fit_transform(data_vec.reshape(-1, 1)).flatten()
    normalized = normalized - float(np.min(normalized))
    normalized = np.log(normalized + 1.0)

    meta = {
        "critical_index": float(i),
        "mean_first_200": mean_first_200,
        "min_first_1000": min_value,
        "window": float(window),
    }
    return normalized.astype(np.float32), meta


def preprocess_response(input_path: Path, output_path: Path, window: int = 320) -> None:
    xls = pd.ExcelFile(input_path)
    processed: Dict[str, np.ndarray] = {}
    meta_rows: List[Dict[str, float]] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if df.shape[1] < 4:
            # Skip sheets that don't conform
            continue
        series = pd.to_numeric(df.iloc[:, 3], errors="coerce").values
        try:
            vec, meta = _process_one_series(series, window=window)
        except Exception as e:
            # Record failure for transparency
            meta_rows.append({"sheet": sheet, "status": 0.0, "error": str(e)[:120]})
            continue

        processed[sheet] = vec
        meta_rows.append({"sheet": sheet, "status": 1.0, **meta})

    if len(processed) == 0:
        raise RuntimeError("No valid sheets were processed. Please check input format.")

    # Make columns = sample IDs, rows = feature index
    out_df = pd.DataFrame(processed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        out_df.to_excel(writer, sheet_name="ai", index=False)
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="meta", index=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess AI response features from raw 100-IT workbook.")
    p.add_argument("--input", type=Path, required=True, help="Path to raw 100-IT.xls")
    p.add_argument("--output", type=Path, required=True, help="Path to write processed ai_features.xlsx")
    p.add_argument("--window", type=int, default=320, help="Feature length per sample")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    preprocess_response(args.input, args.output, window=args.window)
    print(f"[OK] AI features saved to: {args.output}")


if __name__ == "__main__":
    main()
