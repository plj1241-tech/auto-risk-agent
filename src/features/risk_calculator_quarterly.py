import pandas as pd
import numpy as np
import os

def calc_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    positive_equity = d["total_equity"].where(d["total_equity"] > 0)
    d["debt_ratio"]    = (d["total_liab"] / positive_equity) * 100
    d["current_ratio"] = (d["current_assets"] / d["current_liab"]) * 100
    d["icr"]           = d["op_income"] / d["interest_exp"].replace(0, np.nan)
    d["op_margin"]     = (d["op_income"] / d["revenue"]) * 100

    # Z-score는 연간 기준 공식이므로 분기 손익항목(영업이익·매출)을
    # 4배 연율화(annualize)해서 사용. 재무상태표 항목(자산·자본 등)은 시점값이라 그대로 둠.
    ta = d["total_assets"].replace(0, np.nan)
    working_capital = d["current_assets"] - d["current_liab"]
    op_income_annualized = d["op_income"] * 4
    revenue_annualized   = d["revenue"] * 4

    x1 = working_capital / ta
    x2 = d["retained_earnings"] / ta
    x3 = op_income_annualized / ta
    x4 = d["total_equity"] / d["total_liab"].replace(0, np.nan)
    x5 = revenue_annualized / ta
    d["z_score"] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    def z_label(z):
        if pd.isna(z): return "N/A"
        if z > 2.6:    return "안전"
        if z > 1.1:    return "주의"
        return "위험"
    d["z_grade"] = d["z_score"].apply(z_label)

    for col in ["debt_ratio","current_ratio","icr","op_margin","z_score"]:
        d[col] = d[col].replace([np.inf,-np.inf], np.nan)

    return d

def build_panel_quarterly(
    fin_path="data/raw/dart/financials_quarterly.parquet",
    macro_path="data/raw/macro/macro_quarterly.parquet"
) -> pd.DataFrame:

    fin   = pd.read_parquet(fin_path)
    macro = pd.read_parquet(macro_path)

    fin = calc_risk_metrics(fin)

    # 거시지표 1분기 시차(lag) 피처
    macro_lag = macro.copy().sort_values(["year","quarter"]).reset_index(drop=True)
    for col in ["base_rate_kr","usd_krw","fed_rate","ppi_us","wti_oil","iron_ore_price"]:
        if col in macro_lag.columns:
            macro_lag[f"{col}_lag1"] = macro_lag[col].shift(1)

    panel = fin.merge(macro_lag, on=["year","quarter"], how="left")

    os.makedirs("data/processed", exist_ok=True)
    panel.to_parquet("data/processed/panel_quarterly.parquet", index=False)
    print(f"분기 패널 데이터 저장 완료: {len(panel)}행 × {len(panel.columns)}열")
    print(f"\n기업 수: {panel['corp_name'].nunique()}개")
    print(f"기간: {panel['year'].min()}Q1 ~ {panel['year'].max()}Q4")
    print(f"\nZ-grade 분포:\n{panel['z_grade'].value_counts()}")
    return panel

if __name__ == "__main__":
    build_panel_quarterly()
