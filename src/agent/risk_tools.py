from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
import shap

from src.models.risk_model import finite_or_none, get_model_store


RISK_LABELS = {
    "debt_ratio": "부채비율",
    "current_ratio": "유동비율",
    "icr": "이자보상배율",
    "op_margin": "영업이익률",
    "z_score": "Z-score",
}

FEATURE_LABELS = {
    "base_rate_kr": "한국 기준금리",
    "usd_krw": "USD/KRW 환율",
    "ppi_us": "미국 PPI",
    "wti_oil": "WTI 유가",
    "iron_ore_price": "철광석 가격",
    "base_rate_kr_lag1": "전년도 한국 기준금리",
    "usd_krw_lag1": "전년도 USD/KRW 환율",
    "ppi_us_lag1": "전년도 미국 PPI",
    "wti_oil_lag1": "전년도 WTI 유가",
    "iron_ore_price_lag1": "전년도 철광석 가격",
    "corp_code_enc": "기업 고유효과",
}

SCENARIO_FEATURES = {
    "base_rate_kr",
    "usd_krw",
    "ppi_us",
    "wti_oil",
    "iron_ore_price",
}

HIGHER_IS_BETTER = {
    "debt_ratio": False,
    "current_ratio": True,
    "icr": True,
    "op_margin": True,
    "z_score": True,
}


def _risk_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if pd.notna(row.get("debt_ratio")) and row["debt_ratio"] >= 200:
        flags.append("부채비율 200% 이상")
    if pd.notna(row.get("current_ratio")) and row["current_ratio"] < 100:
        flags.append("유동비율 100% 미만")
    if pd.notna(row.get("icr")) and row["icr"] < 1:
        flags.append("이자보상배율 1배 미만")
    if pd.notna(row.get("op_margin")) and row["op_margin"] < 0:
        flags.append("영업적자")
    if row.get("z_grade") == "위험":
        flags.append("Z-score 위험 구간")
    return flags


def get_current_risk(corp_name: str, year: int | None = None) -> dict[str, Any]:
    store = get_model_store()
    row = store.company_row(corp_name, year)
    metrics = {target: finite_or_none(row.get(target)) for target in RISK_LABELS}
    return {
        "corp_name": row["corp_name"],
        "year": int(row["year"]),
        "metrics": metrics,
        "metric_labels": RISK_LABELS,
        "z_grade": row.get("z_grade", "N/A"),
        "risk_flags": _risk_flags(row),
        "data_quality": {
            "n_quarters": finite_or_none(row.get("n_quarters")),
            "complete_year": bool(row.get("n_quarters", 0) == 4),
            "missing_metrics": [name for name, value in metrics.items() if value is None],
        },
        "source": "data/processed/panel_annual_v2.parquet",
    }


def compare_peers(corp_name: str, year: int | None = None) -> dict[str, Any]:
    store = get_model_store()
    row = store.company_row(corp_name, year)
    selected_year = int(row["year"])
    peers = store.panel[store.panel["year"] == selected_year]
    comparisons: dict[str, Any] = {}
    for target in RISK_LABELS:
        series = peers[["corp_name", target]].dropna().sort_values(target)
        value = row.get(target)
        if pd.isna(value) or series.empty:
            comparisons[target] = None
            continue
        ascending = not HIGHER_IS_BETTER[target]
        health_order = series.sort_values(target, ascending=ascending).reset_index(drop=True)
        rank_index = health_order.index[health_order["corp_name"] == row["corp_name"]]
        health_rank = int(rank_index[0] + 1) if len(rank_index) else None
        percentile = float((series[target] <= float(value)).mean() * 100)
        comparisons[target] = {
            "value": float(value),
            "peer_median": float(series[target].median()),
            "raw_percentile": percentile,
            "health_rank": health_rank,
            "peer_count": int(len(series)),
            "higher_is_better": HIGHER_IS_BETTER[target],
        }
    return {
        "corp_name": row["corp_name"],
        "year": selected_year,
        "comparisons": comparisons,
        "interpretation": "health_rank 1이 동종기업 중 가장 양호합니다.",
    }


def run_scenario(corp_name: str, scenario: dict[str, float]) -> dict[str, Any]:
    store = get_model_store()
    row = store.company_row(corp_name)
    base_frame = store.feature_frame(row)
    scenario_frame = base_frame.copy()
    applied: dict[str, Any] = {}

    if not scenario:
        raise ValueError("최소 한 개의 시나리오 변수를 입력해야 합니다.")
    for feature, raw_value in scenario.items():
        if feature not in SCENARIO_FEATURES:
            raise ValueError(
                f"지원하지 않는 시나리오 변수입니다: {feature}. "
                f"지원 변수: {', '.join(sorted(SCENARIO_FEATURES))}"
            )
        value = float(raw_value)
        lower, upper = store.macro_range(feature)
        if not lower <= value <= upper:
            raise ValueError(
                f"{feature}={value}는 관측 범위 {lower:.4g}~{upper:.4g} 밖입니다."
            )
        scenario_frame.loc[0, feature] = value
        applied[feature] = {
            "label": FEATURE_LABELS[feature],
            "before": float(base_frame.loc[0, feature]),
            "after": value,
            "observed_range": [lower, upper],
        }

    predictions: dict[str, Any] = {}
    for target in store.targets:
        baseline = store.predict(target, base_frame)
        changed = store.predict(target, scenario_frame)
        predictions[target] = {
            "label": RISK_LABELS[target],
            "baseline": baseline,
            "scenario": changed,
            "delta": changed - baseline,
            "delta_pct": ((changed - baseline) / abs(baseline) * 100)
            if baseline != 0
            else None,
        }

    return {
        "corp_name": row["corp_name"],
        "base_year": int(row["year"]),
        "applied_scenario": applied,
        "predictions": predictions,
        "guardrail": {
            "within_observed_range": True,
            "causal_forecast": False,
            "message": (
                "과거 관측 범위 안에서 한 변수만 바꾼 모델 민감도입니다. "
                "거시변수의 인과효과나 실제 미래값을 의미하지 않습니다."
            ),
        },
    }


def explain_shap(
    corp_name: str, target: str = "z_score", top_n: int = 5
) -> dict[str, Any]:
    store = get_model_store()
    if target not in store.models:
        raise ValueError(f"지원하지 않는 리스크지표입니다: {target}")
    row = store.company_row(corp_name)
    frame = store.feature_frame(row)
    explainer = shap.TreeExplainer(store.models[target])
    shap_values = np.asarray(explainer.shap_values(frame[store.features]))[0]
    contributions = [
        {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "feature_value": float(frame.loc[0, feature]),
            "shap_value": float(value),
            "direction": "증가" if value > 0 else "감소" if value < 0 else "중립",
        }
        for feature, value in zip(store.features, shap_values)
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    company_effect = next(
        (item for item in contributions if item["feature"] == "corp_code_enc"), None
    )
    macro_effects = [
        item for item in contributions if item["feature"] != "corp_code_enc"
    ][: max(1, min(int(top_n), 10))]
    expected = explainer.expected_value
    if isinstance(expected, np.ndarray):
        expected = expected.ravel()[0]
    return {
        "corp_name": row["corp_name"],
        "year": int(row["year"]),
        "target": target,
        "target_label": RISK_LABELS[target],
        "prediction": store.predict(target, frame),
        "expected_value": float(expected),
        "company_effect": company_effect,
        "top_macro_effects": macro_effects,
        "unit_note": (
            "부채비율 SHAP값은 로그 학습공간 단위입니다."
            if target == "debt_ratio" and store.debt_ratio_log_transformed
            else "SHAP 부호는 모델 출력 증가·감소 방향입니다."
        ),
        "validation_note": (
            "기업 고유효과의 비중이 큰 모델이므로 거시변수 SHAP을 "
            "인과효과로 해석하지 않습니다."
        ),
    }


TOOL_FUNCTIONS = {
    "get_current_risk": get_current_risk,
    "run_scenario": run_scenario,
    "explain_shap": explain_shap,
    "compare_peers": compare_peers,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"지원하지 않는 도구입니다: {name}")
    return TOOL_FUNCTIONS[name](**arguments)
