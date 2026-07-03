from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
except (FileNotFoundError, KeyError):
    pass

from src.agent.pdf_report import generate_pdf
from src.agent.risk_tools import FEATURE_LABELS, RISK_LABELS, get_current_risk, run_scenario, compare_peers, explain_shap
from src.agent.service import RiskAnalysisAgent
from src.models.risk_model import get_model_store


st.set_page_config(
    page_title="자동차 부품 리스크 에이전트",
    page_icon="🚘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    h1 {font-size:2.05rem !important; line-height:1.25 !important;}
    [data-testid="stMetric"] {background:#ffffff; border:1px solid #e5e3dd; padding:14px; border-radius:12px;}
    [data-testid="stMetricValue"] {font-size:1.55rem;}
    .risk-banner {padding:14px 18px; border-radius:12px; background:#e6f1fb; border:1px solid #b5d4f4; margin-bottom:16px;}
    .small-note {font-size:.82rem; color:#666;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_agent() -> RiskAnalysisAgent:
    return RiskAnalysisAgent()


@st.cache_resource
def load_store():
    return get_model_store()


agent = load_agent()
store = load_store()
panel = store.panel.copy()

st.title("🚘 자동차 부품업계 리스크 분석 에이전트")
st.caption("DART 재무데이터 × 거시경제지표 | 21개 기업 · 2019~2024")

with st.sidebar:
    st.header("분석 설정")
    corp_name = st.selectbox("기업", store.companies, index=store.companies.index("현대모비스"))
    available_years = sorted(panel.loc[panel.corp_name.eq(corp_name), "year"].unique(), reverse=True)
    selected_year = st.selectbox("기준연도", available_years)
    use_gemini = st.toggle("Gemini 설명 사용", value=True)
    st.divider()
    st.markdown("**모델 사용 범위**")
    st.caption("현재위험·과거추이·동종비교: 사용 가능")
    st.caption("시나리오: 관측 범위 내 민감도만 제공")

current = agent.store.company_row(corp_name, int(selected_year))
current_risk = get_current_risk(corp_name, int(selected_year))

# ── 현재 거시지표값 (챗봇 컨텍스트용) ──────────────────
current_rate    = float(current.get("base_rate_kr", 3.0))
current_usdkrw  = float(current.get("usd_krw", 1300.0))
current_wti     = float(current.get("wti_oil", 70.0))

grade_color = {"안전": "#1D9E75", "주의": "#A15418", "위험": "#C0392B"}.get(
    current_risk["z_grade"], "#5B5A57"
)
st.markdown(
    f'<div class="risk-banner"><b>{corp_name} · {selected_year}년</b> '
    f'Z-score 등급: <span style="color:{grade_color};font-weight:700">{current_risk["z_grade"]}</span>'
    f'<br><span class="small-note">{", ".join(current_risk["risk_flags"]) or "주요 경고 없음"}</span></div>',
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
for column, (name, label) in zip(metric_columns, RISK_LABELS.items()):
    value = current_risk["metrics"].get(name)
    suffix = "%" if name in {"debt_ratio", "current_ratio", "op_margin"} else ""
    column.metric(label, "N/A" if value is None else f"{value:,.2f}{suffix}")

overview_tab, trend_tab, scenario_tab, peer_tab, chat_tab, report_tab = st.tabs(
    ["요약", "추이", "시나리오", "동종·SHAP", "챗봇", "리포트"]
)

with overview_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("리스크지표 추이")
        risk_choice = st.selectbox("리스크지표", list(RISK_LABELS), format_func=RISK_LABELS.get, key="overview_risk")
        history = panel[panel.corp_name.eq(corp_name)][["year", risk_choice]].dropna()
        chart = (
            alt.Chart(history)
            .mark_line(point=True, color="#185FA5", strokeWidth=3)
            .encode(x=alt.X("year:O", title="연도"), y=alt.Y(f"{risk_choice}:Q", title=RISK_LABELS[risk_choice]), tooltip=["year", risk_choice])
            .properties(height=330)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.subheader("데이터 품질")
        quality = current_risk["data_quality"]
        st.write(f"- 구성 분기: {quality['n_quarters'] or 'N/A'}개")
        st.write(f"- 완전 연도: {'예' if quality['complete_year'] else '아니오'}")
        st.write(f"- 결측 지표: {', '.join(quality['missing_metrics']) or '없음'}")
        st.info("현재 등급과 수치는 공시 기반 분석 참고자료이며 투자·신용의견이 아닙니다.")

with trend_tab:
    st.subheader("리스크지표와 거시변수 오버레이")
    col1, col2 = st.columns(2)
    risk_overlay = col1.selectbox("리스크", list(RISK_LABELS), format_func=RISK_LABELS.get, key="overlay_risk")
    macro_options = ["base_rate_kr", "usd_krw", "ppi_us", "wti_oil", "iron_ore_price"]
    macro_overlay = col2.selectbox("거시변수", macro_options, format_func=FEATURE_LABELS.get)
    overlay = panel[panel.corp_name.eq(corp_name)][["year", risk_overlay, macro_overlay]].dropna().copy()
    for name in [risk_overlay, macro_overlay]:
        first = overlay[name].iloc[0]
        overlay[FEATURE_LABELS.get(name, RISK_LABELS.get(name, name))] = overlay[name] / first * 100 if first else overlay[name]
    melted = overlay.melt(
        id_vars="year",
        value_vars=[RISK_LABELS[risk_overlay], FEATURE_LABELS[macro_overlay]],
        var_name="구분",
        value_name="지수",
    )
    overlay_chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=3)
        .encode(x=alt.X("year:O", title="연도"), y=alt.Y("지수:Q", title="최초연도=100"), color="구분:N", tooltip=["year", "구분", "지수"])
        .properties(height=420)
    )
    st.altair_chart(overlay_chart, use_container_width=True)

with scenario_tab:
    st.subheader("거시변수 시나리오 민감도")
    st.warning("관측 범위 안에서 한 변수만 바꾼 모델 민감도입니다. 실제 미래값이나 인과효과가 아닙니다.")
    scenario: dict[str, float] = {}
    scenario_cols = st.columns(2)
    scenario_names = ["base_rate_kr", "usd_krw", "wti_oil", "iron_ore_price"]
    for index, feature in enumerate(scenario_names):
        lower, upper = store.macro_range(feature)
        default = float(current.get(feature))
        with scenario_cols[index % 2]:
            enabled = st.checkbox(f"{FEATURE_LABELS[feature]} 변경", key=f"enable_{feature}")
            step = (upper - lower) / 100 if upper > lower else 0.01
            value = st.slider(
                FEATURE_LABELS[feature],
                min_value=float(lower),
                max_value=float(upper),
                value=min(max(default, lower), upper),
                step=float(step),
                disabled=not enabled,
                key=f"slider_{feature}",
            )
            if enabled:
                scenario[feature] = value
    if st.button("민감도 계산", type="primary", disabled=not bool(scenario)):
        try:
            st.session_state["scenario_result"] = run_scenario(corp_name, scenario)
            st.session_state["scenario_input"] = scenario
        except ValueError as exc:
            st.error(str(exc))
    scenario_result = st.session_state.get("scenario_result")
    if scenario_result and scenario_result["corp_name"] == corp_name:
        rows = [
            {"지표": item["label"], "기준": item["baseline"], "시나리오": item["scenario"], "변화": item["delta"]}
            for item in scenario_result["predictions"].values()
        ]
        st.dataframe(pd.DataFrame(rows).style.format({"기준": "{:.2f}", "시나리오": "{:.2f}", "변화": "{:+.2f}"}), use_container_width=True, hide_index=True)

with peer_tab:
    left, right = st.columns(2)
    peers = compare_peers(corp_name, int(selected_year))
    with left:
        st.subheader("동종업계 건전성 순위")
        peer_rows = [
            {"지표": RISK_LABELS[name], "기업값": item["value"], "중앙값": item["peer_median"], "순위": f"{item['health_rank']}/{item['peer_count']}"}
            for name, item in peers["comparisons"].items() if item
        ]
        st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Z-score SHAP 설명")
        shap_result = explain_shap(corp_name, "z_score", 5)
        shap_rows = pd.DataFrame(shap_result["top_macro_effects"])
        shap_chart = (
            alt.Chart(shap_rows)
            .mark_bar()
            .encode(x=alt.X("shap_value:Q", title="SHAP값"), y=alt.Y("label:N", sort="-x", title=""), color=alt.condition("datum.shap_value > 0", alt.value("#185FA5"), alt.value("#D85A30")), tooltip=["label", "feature_value", "shap_value"])
            .properties(height=300)
        )
        st.altair_chart(shap_chart, use_container_width=True)
        st.caption(shap_result["validation_note"])

with chat_tab:
    st.subheader("리스크 분석 챗봇")

    # ── 챗봇 컨텍스트 힌트 표시 ──────────────────────────
    st.caption(
        f"현재 컨텍스트: **{corp_name}** · **{selected_year}년** · "
        f"기준금리 {current_rate:.2f}% · 환율 {current_usdkrw:.0f}원"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기업·연도가 바뀌면 대화 초기화
    ctx_key = f"{corp_name}_{selected_year}"
    if st.session_state.get("chat_ctx") != ctx_key:
        st.session_state.messages = []
        st.session_state["chat_ctx"] = ctx_key

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(f"{corp_name}에 대해 질문하세요")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                # ── 핵심 수정: selected_year + 현재 거시값 전달 ──
                result = agent.analyze(
                    prompt,
                    corp_name=corp_name,
                    base_year=int(selected_year),
                    current_macro={
                        "base_rate_kr": current_rate,
                        "usd_krw": current_usdkrw,
                        "wti_oil": current_wti,
                    },
                    use_llm=use_gemini,
                )
            st.markdown(result["answer"])
            with st.expander("사용 도구"):
                st.json(result.get("tool_trace", []))
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

with report_tab:
    st.subheader("애널리스트 리포트")
    include_scenario = st.checkbox("현재 시나리오 결과 포함", value=bool(st.session_state.get("scenario_input")))
    llm_report = st.checkbox("Gemini 문체 변환", value=False)
    report_scenario = st.session_state.get("scenario_input") if include_scenario else None
    if st.button("리포트 생성", type="primary"):
        with st.spinner("리포트 생성 중..."):
            report = agent.report(corp_name, report_scenario, use_llm=llm_report)
            st.session_state["report"] = report
    report = st.session_state.get("report")
    if report and report["corp_name"] == corp_name:
        rendered = report.get("llm_markdown") or report["markdown"]
        st.markdown(rendered)
        pdf_bytes = generate_pdf(report)
        left, right = st.columns(2)
        left.download_button("Markdown 다운로드", report["markdown"].encode("utf-8"), file_name=f"{corp_name}_risk_report.md", mime="text/markdown", use_container_width=True)
        right.download_button("PDF 다운로드", pdf_bytes, file_name=f"{corp_name}_risk_report.pdf", mime="application/pdf", use_container_width=True)