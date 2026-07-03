from __future__ import annotations

from typing import Any

from src.models.risk_model import get_model_store

from .gemini_client import GeminiClient
from .report_generator import build_report
from .risk_tools import compare_peers, execute_tool, explain_shap, get_current_risk


class RiskAnalysisAgent:
    def __init__(self, gemini_client: GeminiClient | None = None):
        self.store = get_model_store()
        self.gemini = gemini_client or GeminiClient()

    def _detect_company(self, text: str, explicit: str | None = None) -> str:
        if explicit:
            return self.store.resolve_company(explicit)
        matches = [company for company in self.store.companies if company in text]
        if len(matches) == 1:
            return matches[0]
        raise ValueError("질문에 지원 기업명을 하나 포함하거나 corp_name을 입력하세요.")

    def _build_context_prompt(
        self,
        question: str,
        corp_name: str,
        base_year: int,
        current_macro: dict[str, float],
    ) -> str:
        """
        Gemini에게 전달할 프롬프트.
        - 현재 선택된 기업·연도·거시지표값을 명시해서
          "금리 -1%"를 변화량으로 올바르게 해석하도록 유도.
        """
        rate     = current_macro.get("base_rate_kr", 3.0)
        usdkrw   = current_macro.get("usd_krw", 1300.0)
        wti      = current_macro.get("wti_oil", 70.0)

        macro_range = self.store.macro_range if hasattr(self.store, "macro_range") else None
        rate_min, rate_max = (macro_range("base_rate_kr") if macro_range else (0.5, 3.5))

        return f"""[현재 분석 컨텍스트]
- 분석 기업: {corp_name}
- 기준 연도: {base_year}년
- 현재 한국 기준금리: {rate:.2f}%  (관측 범위: {rate_min:.1f}% ~ {rate_max:.1f}%)
- 현재 USD/KRW 환율: {usdkrw:.0f}원
- 현재 WTI 유가: {wti:.1f}달러

[시나리오 해석 규칙]
- "금리 -1%", "금리 1%p 인하" 같은 표현 → 현재 금리에서 뺀 절댓값 사용
  예: 금리 -1%p → base_rate_kr = {rate - 1.0:.2f}
- "금리 +0.5%p 인상" → base_rate_kr = {rate + 0.5:.2f}
- "환율 1400원" → usd_krw = 1400 (절댓값 직접 사용)
- 반드시 관측 범위 내 값으로 run_scenario 도구를 호출할 것

[질문]
{question}"""

    def _fallback(
        self,
        question: str,
        corp_name: str | None = None,
        base_year: int | None = None,
    ) -> dict[str, Any]:
        company = self._detect_company(question, corp_name)
        year    = base_year or self.store.latest_year
        current = get_current_risk(company, year)
        tools   = [{"tool": "get_current_risk", "result": current}]
        if any(word in question for word in ["비교", "업계", "순위", "동종"]):
            tools.append({"tool": "compare_peers", "result": compare_peers(company, year)})
        if any(word in question.lower() for word in ["원인", "이유", "shap", "영향"]):
            tools.append({"tool": "explain_shap", "result": explain_shap(company)})
        metrics = current["metrics"]

        def metric_text(name: str, suffix: str = "") -> str:
            value = metrics.get(name)
            return "N/A" if value is None else f"{value:.2f}{suffix}"

        answer = (
            f"{company}의 {year}년 Z-score는 "
            f"{metric_text('z_score')}로 {current['z_grade']} 등급입니다. "
            f"부채비율 {metric_text('debt_ratio', '%')}, 유동비율 "
            f"{metric_text('current_ratio', '%')}, 영업이익률 "
            f"{metric_text('op_margin', '%')}입니다. "
            "결과는 공시 기반 분석 참고자료이며 거시변수의 인과예측이 아닙니다."
        )
        return {
            "answer": answer,
            "provider": "local_fallback",
            "tool_trace": tools,
        }

    def analyze(
        self,
        question: str,
        corp_name: str | None = None,
        base_year: int | None = None,           # ← 추가: 선택된 기준연도
        current_macro: dict[str, float] | None = None,  # ← 추가: 현재 거시지표값
        use_llm: bool = True,
    ) -> dict[str, Any]:
        company  = self._detect_company(question, corp_name)
        year     = base_year or self.store.latest_year
        macro    = current_macro or {}

        if use_llm and self.gemini.available:
            # 컨텍스트가 포함된 프롬프트 생성
            prompt = self._build_context_prompt(question, company, year, macro)
            try:
                return self.gemini.generate(prompt, execute_tool)
            except Exception as exc:
                fallback = self._fallback(question, corp_name, base_year)
                fallback["llm_error"] = str(exc)
                return fallback

        return self._fallback(question, corp_name, base_year)

    def report(
        self,
        corp_name: str,
        scenario: dict[str, float] | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        report = build_report(corp_name, scenario)
        if use_llm and self.gemini.available:
            try:
                report["llm_markdown"] = self.gemini.rewrite_report(report["markdown"])
                report["llm_provider"] = self.gemini.model
            except Exception as exc:
                report["llm_error"] = str(exc)
        return report