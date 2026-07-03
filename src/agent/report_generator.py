from __future__ import annotations

from typing import Any

from .risk_tools import (
    RISK_LABELS,
    compare_peers,
    explain_shap,
    get_current_risk,
    run_scenario,
)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.{digits}f}"


def _actions(current: dict[str, Any]) -> list[str]:
    metrics = current["metrics"]
    actions: list[str] = []
    if metrics.get("current_ratio") is not None and metrics["current_ratio"] < 100:
        actions.append("단기 유동성 및 운전자본 회전 점검")
    if metrics.get("icr") is not None and metrics["icr"] < 1:
        actions.append("이자비용 부담과 차입 만기구조 점검")
    if metrics.get("debt_ratio") is not None and metrics["debt_ratio"] >= 200:
        actions.append("차입금 축소와 자본확충 가능성 검토")
    if metrics.get("op_margin") is not None and metrics["op_margin"] < 0:
        actions.append("적자 사업·원가구조 및 고객사별 수익성 점검")
    if current["z_grade"] == "위험":
        actions.append("현금흐름과 계속기업 리스크에 대한 상세 실사")
    if not actions:
        actions.append("현재 위험등급 유지 여부를 다음 결산 데이터에서 재점검")
    return actions


def build_report(
    corp_name: str, scenario: dict[str, float] | None = None
) -> dict[str, Any]:
    current = get_current_risk(corp_name)
    peers = compare_peers(corp_name, current["year"])
    explanation = explain_shap(corp_name, "z_score", top_n=5)
    scenario_result = run_scenario(corp_name, scenario) if scenario else None
    actions = _actions(current)

    metric_lines = [
        f"- {RISK_LABELS[name]}: {_fmt(value)}"
        for name, value in current["metrics"].items()
    ]
    peer_lines = []
    for name, item in peers["comparisons"].items():
        if item:
            peer_lines.append(
                f"- {RISK_LABELS[name]}: {item['health_rank']}/{item['peer_count']}위, "
                f"업계 중앙값 {_fmt(item['peer_median'])}"
            )
    cause_lines = [
        f"- {item['label']}: SHAP {item['shap_value']:+.3f} ({item['direction']})"
        for item in explanation["top_macro_effects"]
    ]
    scenario_lines = ["- 입력된 시나리오 없음"]
    if scenario_result:
        scenario_lines = [
            f"- {item['label']}: {_fmt(item['baseline'])} → {_fmt(item['scenario'])} "
            f"(Δ {item['delta']:+.2f})"
            for item in scenario_result["predictions"].values()
        ]

    markdown = "\n".join(
        [
            f"# {current['corp_name']} 리스크 분석 리포트",
            f"기준연도: {current['year']}년 | Z-score 등급: {current['z_grade']}",
            "",
            "## 1. 요약",
            *metric_lines,
            f"- 주요 경고: {', '.join(current['risk_flags']) or '없음'}",
            "",
            "## 2. 시나리오 분석",
            *scenario_lines,
            "- 주의: 시나리오는 과거 관측 범위 내 모델 민감도이며 인과예측이 아닙니다.",
            "",
            "## 3. 원인 분석",
            *cause_lines,
            "- 기업 고유효과 비중이 커 거시 SHAP을 인과효과로 해석하지 않습니다.",
            "",
            "## 4. 동종업계 비교",
            *peer_lines,
            "",
            "## 5. 점검 제안",
            *[f"- {action}" for action in actions],
            "",
            "---",
            "본 리포트는 공시 재무데이터 기반 분석 참고자료이며 투자·신용의견이 아닙니다.",
        ]
    )
    return {
        "corp_name": current["corp_name"],
        "year": current["year"],
        "sections": {
            "summary": current,
            "scenario": scenario_result,
            "causes": explanation,
            "peer_comparison": peers,
            "actions": actions,
        },
        "markdown": markdown,
    }
