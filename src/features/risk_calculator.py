import pandas as pd
import numpy as np
import os

def calc_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 1. 부채비율: 자본잠식은 음수 비율 대신 별도 위험 신호로 다룬다.
    positive_equity = d["total_equity"].where(d["total_equity"] > 0)
    d["debt_ratio"] = (d["total_liab"] / positive_equity) * 100

    # 2. 유동비율
    d["current_ratio"] = (d["current_assets"] / d["current_liab"]) * 100

    # 3. 이자보상배율
    d["icr"] = d["op_income"] / d["interest_exp"].replace(0, np.nan)

    # 4. 영업이익률
    d["op_margin"] = (d["op_income"] / d["revenue"]) * 100

    # 5. Altman Z-score (제조업 수정판)
    ta = d["total_assets"].replace(0, np.nan)
    working_capital = d["current_assets"] - d["current_liab"]
    x1 = working_capital / ta
    x2 = d["retained_earnings"] / ta
    x3 = d["op_income"] / ta
    x4 = d["total_equity"] / d["total_liab"].replace(0, np.nan)
    x5 = d["revenue"] / ta
    d["z_score"] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    def z_label(z):
        if pd.isna(z):  return "N/A"
        if z > 2.6:     return "안전"
        if z > 1.1:     return "주의"
        return "위험"

    d["z_grade"] = d["z_score"].apply(z_label)

    for col in ["debt_ratio","current_ratio","icr","op_margin","z_score"]:
        d[col] = d[col].replace([np.inf, -np.inf], np.nan)

    return d

def build_panel(fin_path="data/raw/dart/financials.parquet",
                macro_path="data/raw/macro/macro.parquet") -> pd.DataFrame:

    fin   = pd.read_parquet(fin_path)
    macro = pd.read_parquet(macro_path)

    # 리스크지표 계산
    fin = calc_risk_metrics(fin)

    # macro 인덱스 정리 — year 컬럼으로 변환
    macro = macro.reset_index()
    macro.rename(columns={"index": "year"}, inplace=True)
    if "year" not in macro.columns:
        macro.columns = ["year"] + list(macro.columns[1:])
    macro["year"] = macro["year"].astype(int)

    # lag 피처 (1년 시차)
    macro_lag = macro.copy()
    for col in ["base_rate_kr","usd_krw","fed_rate","ppi_us","wti_oil","iron_ore_price"]:
        if col in macro_lag.columns:
            macro_lag[f"{col}_lag1"] = macro_lag[col].shift(1)

    # 합치기
    panel = fin.merge(macro_lag, on="year", how="left")

    os.makedirs("data/processed", exist_ok=True)
    panel.to_parquet("data/processed/panel.parquet", index=False)
    print(f"패널 데이터 저장 완료: {len(panel)}행 × {len(panel.columns)}열")
    print(panel[["corp_name","year","debt_ratio","current_ratio","icr","op_margin","z_score","z_grade"]].to_string())
    return panel

if __name__ == "__main__":
    build_panel()
