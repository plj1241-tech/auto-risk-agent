from __future__ import annotations

import os
from typing import Any, Callable

import requests
from dotenv import load_dotenv


load_dotenv()


SYSTEM_PROMPT = """당신은 자동차 부품업계 리스크 분석 전문 에이전트입니다.
반드시 제공된 도구를 사용하고, 도구 없이 답변을 만들지 않습니다.
현재 데이터는 최근 21개 자동차 기업의 과거 패턴에 기반합니다.
시나리오 결과는 예측이나 인과효과가 아님을 명시하고 한국어로 답변합니다.
수치와 데이터를 중심으로 구체적으로 답하고, 데이터 한계와 거시 경도를 함께 안내합니다.
"""


FUNCTION_DECLARATIONS = [
    {
        "name": "get_current_risk",
        "description": "기업의 최근 또는 특정 연도 리스크 지표를 조회합니다.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "corp_name": {"type": "STRING", "description": "기업명 (예: 현대모비스)"},
                "year": {"type": "INTEGER", "description": "기준 연도 (예: 2024)"},
            },
            "required": ["corp_name"],
        },
    },
    {
        "name": "run_scenario",
        "description": "거시변수 시나리오 변경에 따른 리스크 지표 민감도를 계산합니다. 금리·환율·유가 변경 질문에 반드시 이 도구를 사용하세요.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "corp_name": {"type": "STRING", "description": "기업명"},
                "scenario": {
                    "type": "OBJECT",
                    "description": "변경할 거시변수 절댓값. 예: 금리 -1%p면 현재금리에서 1을 뺀 값을 입력",
                    "properties": {
                        "base_rate_kr": {"type": "NUMBER", "description": "한국 기준금리 (%)"},
                        "usd_krw": {"type": "NUMBER", "description": "USD/KRW 환율 (원)"},
                        "ppi_us": {"type": "NUMBER", "description": "미국 PPI"},
                        "wti_oil": {"type": "NUMBER", "description": "WTI 유가 (달러/배럴)"},
                        "iron_ore_price": {"type": "NUMBER", "description": "철광석 가격 (달러/MT)"},
                    },
                },
            },
            "required": ["corp_name", "scenario"],
        },
    },
    {
        "name": "explain_shap",
        "description": "특정 기업과 지표의 SHAP 변수 기여도를 설명합니다.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "corp_name": {"type": "STRING", "description": "기업명"},
                "target": {
                    "type": "STRING",
                    "description": "분석할 리스크 지표",
                    "enum": ["debt_ratio", "current_ratio", "icr", "op_margin", "z_score"],
                },
                "top_n": {"type": "INTEGER", "description": "상위 변수 개수"},
            },
            "required": ["corp_name"],
        },
    },
    {
        "name": "compare_peers",
        "description": "특정 연도의 21개 동종 기업과 리스크 지표를 비교합니다.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "corp_name": {"type": "STRING", "description": "기업명"},
                "year": {"type": "INTEGER", "description": "비교 연도"},
            },
            "required": ["corp_name"],
        },
    },
]


class GeminiClient:
    """Minimal Gemini REST client with function-calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        question: str,
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_tool_rounds: int = 4,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        headers = {"x-goog-api-key": self.api_key}
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": question}]}
        ]
        tool_trace: list[dict[str, Any]] = []

        for _ in range(max_tool_rounds + 1):
            payload = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": contents,
                "tools": [{"functionDeclarations": FUNCTION_DECLARATIONS}],
                "generationConfig": {"temperature": 0.1},
            }
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini 응답에 candidate가 없습니다: {data}")
            model_content = candidates[0].get("content", {})
            parts = model_content.get("parts", [])
            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not function_calls:
                text = "\n".join(p.get("text", "") for p in parts).strip()
                return {
                    "answer": text,
                    "provider": "gemini",
                    "model": self.model,
                    "tool_trace": tool_trace,
                }

            contents.append({"role": "model", "parts": parts})
            response_parts = []
            for call in function_calls:
                name = call["name"]
                arguments = call.get("args", {})
                try:
                    result = tool_executor(name, arguments)
                except Exception as exc:
                    result = {"error": str(exc)}
                tool_trace.append({"tool": name, "arguments": arguments, "result": result})
                response_parts.append(
                    {
                        "functionResponse": {
                            "name": name,
                            "response": {"result": result},
                        }
                    }
                )
            contents.append({"role": "user", "parts": response_parts})
        raise RuntimeError("Gemini 도구 호출 횟수 한계를 초과했습니다.")

    def rewrite_report(self, markdown: str) -> str:
        """Rewrite a grounded deterministic report without changing its numbers."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        prompt = (
            "아래 분석 보고서를 전문 애널리스트 문체의 한국어 Markdown으로 다시 써라. "
            "수치, 등급, 기업명, 연도, 지표 순서를 변경하지 말고, "
            "새로운 예측이나 의견을 추가하지 말라.\n\n" + markdown
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        response = requests.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(p.get("text", "") for p in parts).strip()