from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.service import RiskAnalysisAgent
from src.models.risk_model import get_model_store


app = FastAPI(
    title="자동차 부품업계 리스크 분석 API",
    version="0.4.0",
    description=(
        "DART·거시경제 패널 기반 위험 조회, 동종 비교, SHAP 설명, "
        "관측 범위 내 민감도 분석 API"
    ),
)
agent = RiskAnalysisAgent()


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=2, description="자연어 분석 질문")
    corp_name: str | None = Field(default=None, description="질문과 별도로 지정할 기업명")
    use_llm: bool = Field(default=True, description="Gemini 사용 여부")


class ReportRequest(BaseModel):
    corp_name: str
    scenario: dict[str, float] | None = Field(
        default=None,
        description="base_rate_kr, usd_krw, ppi_us, wti_oil, iron_ore_price",
    )
    use_llm: bool = Field(
        default=False,
        description="Gemini로 정량 리포트 문체를 다듬을지 여부",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    store = get_model_store()
    return {
        "status": "ok",
        "phase3": "complete",
        "phase4": "complete",
        "phase5": "in_progress",
        "companies": len(store.companies),
        "model_targets": store.targets,
        "scenario_mode": "bounded_sensitivity_not_causal_forecast",
    }


@app.get("/companies")
def companies() -> dict[str, Any]:
    return {"companies": get_model_store().companies}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        return agent.analyze(request.question, request.corp_name, request.use_llm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/report")
def report(request: ReportRequest) -> dict[str, Any]:
    try:
        return agent.report(request.corp_name, request.scenario, request.use_llm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
