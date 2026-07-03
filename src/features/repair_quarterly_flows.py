"""Repair the legacy quarterly flow conversion without calling DART again."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.dart_collector_v2 import normalize_quarterly_flows


FLOW_COLS = ["revenue", "op_income", "net_income", "interest_exp"]


def _restore_legacy_reported(group: pd.DataFrame) -> pd.DataFrame:
    """Invert the old consecutive-subtraction transform within one company-year."""
    restored = group.sort_values("quarter").copy()
    for col in FLOW_COLS:
        if col not in restored:
            continue
        values: list[float] = []
        previous_reported = np.nan
        for legacy_value in restored[col].tolist():
            if pd.isna(legacy_value):
                values.append(np.nan)
                previous_reported = np.nan
                continue
            reported = float(legacy_value)
            if pd.notna(previous_reported):
                reported += float(previous_reported)
            values.append(reported)
            previous_reported = reported
        restored[col] = values
    return restored


def repair_legacy_quarterly_flows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return corrected single-quarter flows and preserved DART-reported values.

    The function is idempotent: data that already contains ``*_reported`` columns
    is normalized from those values rather than inverted a second time.
    """
    required = {"corp_name", "year", "quarter"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}")

    repaired: list[pd.DataFrame] = []
    has_reported = all(f"{col}_reported" in frame.columns for col in FLOW_COLS)
    for _, group in frame.groupby(["corp_name", "year"], sort=False):
        source = group.copy()
        if has_reported:
            for col in FLOW_COLS:
                source[col] = source[f"{col}_reported"]
        else:
            source = _restore_legacy_reported(source)
        repaired.append(normalize_quarterly_flows(source))

    result = pd.concat(repaired, ignore_index=True)
    return result.sort_values(["corp_name", "year", "quarter"]).reset_index(drop=True)


def repair_file(
    path: str | Path = "data/raw/dart/financials_quarterly.parquet",
) -> pd.DataFrame:
    target = Path(path)
    repaired = repair_legacy_quarterly_flows(pd.read_parquet(target))
    repaired.to_parquet(target, index=False)
    print(f"분기 손익 교정 완료: {target} ({len(repaired)}행)")
    return repaired


if __name__ == "__main__":
    repair_file()
