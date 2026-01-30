from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetInfo:
    sample_ids: List[str]
    num_features: int
    num_points: int
    num_channels: int


class SpectrumDataset(Dataset):
    """Dataset for (AI features -> 3-channel spectra)."""

    def __init__(
        self,
        ai_path: Path | str,
        spectrum_path: Path | str,
        ai_sheet: str | int = 0,
        spectrum_sheets: Sequence[str | int] = ("ch1", "ch2", "ch3"),
        strict: bool = True,
    ):
        self.ai_path = Path(ai_path)
        self.spectrum_path = Path(spectrum_path)

        df_ai = pd.read_excel(self.ai_path, sheet_name=ai_sheet)

        # Read three spectrum channels
        df_s = []
        for s in spectrum_sheets:
            df_s.append(pd.read_excel(self.spectrum_path, sheet_name=s))

        # Determine common sample ids across all
        sample_ids = [
            sid
            for sid in df_ai.columns
            if all(sid in dfi.columns for dfi in df_s)
        ]
        if strict and len(sample_ids) == 0:
            raise ValueError(
                "No common sample IDs found across AI features and all 3 spectrum channels. "
                "Please check column names and sheet selection."
            )

        # Features: columns are samples, rows are features
        X = df_ai[sample_ids].values.T.astype(np.float32)
        # Labels: each sheet columns are samples, rows are points
        Ys = [dfi[sample_ids].values.T.astype(np.float32) for dfi in df_s]

        # Ensure consistent points length
        n_points = Ys[0].shape[1]
        if strict and any(y.shape[1] != n_points for y in Ys):
            raise ValueError("Spectrum channels do not have the same number of points.")

        # Stack labels into (N, C, P)
        Y = np.stack(Ys, axis=1)

        self.X = X
        self.Y = Y
        self.info = DatasetInfo(
            sample_ids=sample_ids,
            num_features=int(self.X.shape[1]),
            num_points=int(n_points),
            num_channels=int(self.Y.shape[1]),
        )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (F,)
        y = torch.from_numpy(self.Y[idx])  # (3, P)
        return x, y
