from __future__ import annotations

import pickle
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel_annual_v2.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "risk_predictor_v2.pkl"


class RiskModelStore:
    """Read-only access to the validated annual panel and trained models."""

    def __init__(self, panel_path: Path = PANEL_PATH, model_path: Path = MODEL_PATH):
        self.panel_path = Path(panel_path)
        self.model_path = Path(model_path)
        self.panel = pd.read_parquet(self.panel_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.model_path.open("rb") as stream:
                saved = pickle.load(stream)
        self.models = saved["models"]
        self.features = list(saved["features"])
        self.label_encoder = saved["label_encoder"]
        self.debt_ratio_log_transformed = bool(
            saved.get("debt_ratio_log_transformed", False)
        )
        self.targets = list(self.models)
        self.companies = sorted(self.panel["corp_name"].dropna().unique().tolist())

    def resolve_company(self, corp_name: str) -> str:
        name = (corp_name or "").strip()
        if name in self.companies:
            return name
        partial = [company for company in self.companies if name and name in company]
        if len(partial) == 1:
            return partial[0]
        raise ValueError(
            f"지원하지 않는 기업입니다: {corp_name!r}. "
            f"지원 기업: {', '.join(self.companies)}"
        )

    def company_row(self, corp_name: str, year: int | None = None) -> pd.Series:
        company = self.resolve_company(corp_name)
        rows = self.panel[self.panel["corp_name"] == company]
        if year is not None:
            rows = rows[rows["year"] == int(year)]
        if rows.empty:
            raise ValueError(f"{company}의 {year or '최신'} 데이터가 없습니다.")
        return rows.sort_values("year").iloc[-1].copy()

    def feature_frame(self, row: pd.Series) -> pd.DataFrame:
        values = row.copy()
        values["corp_code_enc"] = int(
            self.label_encoder.transform([values["corp_name"]])[0]
        )
        frame = pd.DataFrame([{name: values.get(name, np.nan) for name in self.features}])
        if frame.isna().any(axis=None):
            missing = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"모델 입력값이 누락되었습니다: {', '.join(missing)}")
        return frame

    def predict(self, target: str, frame: pd.DataFrame) -> float:
        if target not in self.models:
            raise ValueError(f"지원하지 않는 리스크지표입니다: {target}")
        value = float(self.models[target].predict(frame[self.features])[0])
        if target == "debt_ratio" and self.debt_ratio_log_transformed:
            value = float(np.expm1(value))
        return value

    def macro_range(self, feature: str) -> tuple[float, float]:
        if feature not in self.panel.columns:
            raise ValueError(f"지원하지 않는 시나리오 변수입니다: {feature}")
        series = self.panel[feature].dropna().astype(float)
        return float(series.min()), float(series.max())


@lru_cache(maxsize=1)
def get_model_store() -> RiskModelStore:
    return RiskModelStore()


def finite_or_none(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value
