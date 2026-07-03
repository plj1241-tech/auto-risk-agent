import pandas as pd
import numpy as np
import os

def build_annual_panel(
    quarterly_path="data/raw/dart/financials_quarterly.parquet",
    macro_quarterly_path="data/raw/macro/macro_quarterly.parquet"
):
    print("분기 데이터 → 연간 데이터 변환 시작...")

    fin_q   = pd.read_parquet(quarterly_path)
    macro_q = pd.read_parquet(macro_quarterly_path)

    # ── 1. 재무데이터 연간 집계 ──────────────────────────
    # 손익항목(flow): 4분기 합산 / 재무상태표(stock): 4분기말(Q4) 값 사용
    flow_cols  = ["revenue", "op_income", "net_income", "interest_exp"]
    stock_cols = ["total_assets", "total_liab", "total_equity",
                  "current_assets", "current_liab", "retained_earnings"]

    annual_records = []
    for (corp_name, corp_code, year), grp in fin_q.groupby(["corp_name","corp_code","year"]):
        grp = grp.sort_values("quarter")
        record = {"corp_name": corp_name, "corp_code": corp_code, "year": year}

        # 손익항목: Q4 원 보고 연간값이 있으면 우선 사용한다. 이 값은
        # *_reported 열에 보존되어 있어 일부 분기 누락 시에도 연간 합계를 유지한다.
        for col in flow_cols:
            reported_col = f"{col}_reported"
            q4 = grp[grp["quarter"].eq(4)]
            if reported_col in grp and not q4.empty and pd.notna(q4.iloc[-1][reported_col]):
                record[col] = q4.iloc[-1][reported_col]
            else:
                vals = grp[col].dropna()
                record[col] = vals.sum() if len(vals) > 0 else np.nan
        record["n_quarters"] = len(grp)  # 몇 개 분기로 구성됐는지 기록

        # 재무상태표 항목: 가장 마지막 분기(보통 Q4) 값 사용
        last_row = grp.iloc[-1]
        for col in stock_cols:
            record[col] = last_row.get(col, np.nan)

        annual_records.append(record)

    fin_annual = pd.DataFrame(annual_records)

    # 4개 분기 미만인 연도는 경고 표시 (완전치 않은 연간 데이터)
    incomplete = fin_annual[fin_annual["n_quarters"] < 4]
    if len(incomplete) > 0:
        print(f"\n[참고] 4개 분기 미충족 연도 {len(incomplete)}건 (상장 초기 등):")
        print(incomplete[["corp_name","year","n_quarters"]].to_string(index=False))

    # ── 2. 거시지표 연간 집계 (분기 평균) ────────────────
    macro_annual = macro_q.groupby("year").mean(numeric_only=True).reset_index()
    macro_annual = macro_annual.drop(columns=["quarter"], errors="ignore")

    # lag 피처
    macro_annual = macro_annual.sort_values("year").reset_index(drop=True)
    for col in ["base_rate_kr","usd_krw","fed_rate","ppi_us","wti_oil","iron_ore_price"]:
        if col in macro_annual.columns:
            macro_annual[f"{col}_lag1"] = macro_annual[col].shift(1)

    # ── 3. 리스크지표 계산 ────────────────────────────────
    d = fin_annual.copy()
    positive_equity = d["total_equity"].where(d["total_equity"] > 0)
    d["debt_ratio"]    = (d["total_liab"] / positive_equity) * 100
    d["current_ratio"] = (d["current_assets"] / d["current_liab"]) * 100
    d["icr"]           = d["op_income"] / d["interest_exp"].replace(0, np.nan)
    d["op_margin"]     = (d["op_income"] / d["revenue"]) * 100

    ta = d["total_assets"].replace(0, np.nan)
    working_capital = d["current_assets"] - d["current_liab"]
    x1 = working_capital / ta
    x2 = d["retained_earnings"] / ta
    x3 = d["op_income"] / ta
    x4 = d["total_equity"] / d["total_liab"].replace(0, np.nan)
    x5 = d["revenue"] / ta
    d["z_score"] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    def z_label(z):
        if pd.isna(z): return "N/A"
        if z > 2.6:    return "안전"
        if z > 1.1:    return "주의"
        return "위험"
    d["z_grade"] = d["z_score"].apply(z_label)

    for col in ["debt_ratio","current_ratio","icr","op_margin","z_score"]:
        d[col] = d[col].replace([np.inf,-np.inf], np.nan)

    # ── 4. 합치기 ─────────────────────────────────────────
    panel = d.merge(macro_annual, on="year", how="left")

    os.makedirs("data/processed", exist_ok=True)
    panel.to_parquet("data/processed/panel_annual_v2.parquet", index=False)

    print(f"\n연간 패널 데이터(21개사) 저장 완료: {len(panel)}행 × {len(panel.columns)}열")
    print(f"기업 수: {panel['corp_name'].nunique()}개")
    print(f"\nZ-grade 분포:\n{panel['z_grade'].value_counts()}")
    print(f"\n샘플 미리보기:")
    print(panel[["corp_name","year","debt_ratio","current_ratio","icr","z_score","z_grade"]].head(10).to_string())

    return panel

if __name__ == "__main__":
    build_annual_panel()
