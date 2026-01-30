"""Preprocess raw spectrum workbook -> (3 x 120) label tensors.

Input:
  - Excel file with 3 sheets, each sheet: rows = spectral samples along wavelength/idx, columns = sample IDs.

Output:
  - Excel file with 3 sheets: ch1, ch2, ch3. Each sheet: 120 rows (uniformly sampled), columns = sample IDs.

This script performs cubic spline interpolation to resample each spectrum to a fixed number of points.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def _smooth_and_sample(df: pd.DataFrame, n_points: int, k: int = 3, s: float = 0.0) -> pd.DataFrame:
    """Resample each column to n_points using a spline.

    Args:
        df: DataFrame where each column is a spectrum.
        n_points: Number of uniformly sampled points.
        k: Spline order (3 = cubic).
        s: Smoothing factor. s=0 means interpolation.
    """
    df = df.dropna(how="all").reset_index(drop=True)
    n_rows = df.shape[0]
    if n_rows < 2:
        raise ValueError("Spectrum sheet has <2 valid rows after dropping empty rows.")

    x_orig = np.arange(n_rows, dtype=float)
    x_norm = np.linspace(0.0, 1.0, n_points, dtype=float)
    x_new = x_norm * (n_rows - 1)

    sampled = pd.DataFrame(index=np.arange(n_points))
    for col in df.columns:
        y = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        # If a column is all-NaN, keep it NaN
        if np.all(np.isnan(y)):
            sampled[col] = np.nan
            continue

        # Fill NaNs by linear interpolation to avoid spline crashing
        y_series = pd.Series(y)
        y_filled = y_series.interpolate(limit_direction="both").values

        spline = UnivariateSpline(x_orig, y_filled, k=k, s=s)
        sampled[col] = spline(x_new)

    return sampled


def preprocess_spectrum(
    input_path: Path,
    output_path: Path,
    n_points: int = 120,
    sheet_names: Optional[List[str]] = None,
    k: int = 3,
    s: float = 0.0,
) -> None:
    xls = pd.ExcelFile(input_path)
    names = xls.sheet_names

    if sheet_names is None:
        if len(names) < 3:
            raise ValueError(
                f"Expected >=3 sheets for 3-channel labels, but got {len(names)}: {names}"
            )
        sheet_names = names[:3]

    if len(sheet_names) != 3:
        raise ValueError(f"For 3-channel labels, sheet_names must have length 3, got {sheet_names}")

    dfs = [xls.parse(n) for n in sheet_names]
    processed = [_smooth_and_sample(df, n_points=n_points, k=k, s=s) for df in dfs]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        processed[0].to_excel(writer, sheet_name="ch1", index=False)
        processed[1].to_excel(writer, sheet_name="ch2", index=False)
        processed[2].to_excel(writer, sheet_name="ch3", index=False)

        meta = pd.DataFrame(
            {
                "input_file": [str(input_path)],
                "source_sheets": [", ".join(sheet_names)],
                "n_points": [n_points],
                "spline_k": [k],
                "spline_s": [s],
            }
        )
        meta.to_excel(writer, sheet_name="meta", index=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess spectrum labels (3 channels) to uniform length.")
    p.add_argument("--input", type=Path, required=True, help="Path to raw spectrum_data.xlsx")
    p.add_argument("--output", type=Path, required=True, help="Path to write processed spectrum_labels.xlsx")
    p.add_argument("--n_points", type=int, default=120, help="Number of points per spectrum channel")
    p.add_argument(
        "--sheets",
        type=str,
        default=None,
        help="Comma-separated sheet names to use as (ch1,ch2,ch3). Default: first 3 sheets.",
    )
    p.add_argument("--k", type=int, default=3, help="Spline order (3=cubic)")
    p.add_argument("--s", type=float, default=0.0, help="Spline smoothing factor (0=interpolation)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    sheet_names = args.sheets.split(",") if args.sheets else None
    preprocess_spectrum(
        input_path=args.input,
        output_path=args.output,
        n_points=args.n_points,
        sheet_names=sheet_names,
        k=args.k,
        s=args.s,
    )
    print(f"[OK] spectrum labels saved to: {args.output}")


if __name__ == "__main__":
    main()
