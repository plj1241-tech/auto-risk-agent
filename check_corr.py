import pandas as pd

panel = pd.read_parquet("data/processed/panel.parquet")
risk  = ["debt_ratio","current_ratio","icr","op_margin","z_score"]
macro = ["base_rate_kr","usd_krw","fed_rate","ppi_us","wti_oil","iron_ore_price"]

RISK_LABELS = {
    "debt_ratio":    "부채비율",
    "current_ratio": "유동비율",
    "icr":           "이자보상배율",
    "op_margin":     "영업이익률",
    "z_score":       "Z-score",
}
MACRO_LABELS = {
    "base_rate_kr":   "한국금리",
    "usd_krw":        "환율",
    "fed_rate":       "미국금리",
    "ppi_us":         "PPI",
    "wti_oil":        "유가",
    "iron_ore_price": "철광석",
}

print("=== 전체 상관행렬 (리스크 × 거시) ===")
corr_all = panel[risk+macro].corr().loc[risk, macro]
corr_all.index   = [RISK_LABELS[c] for c in corr_all.index]
corr_all.columns = [MACRO_LABELS[c] for c in corr_all.columns]
print(corr_all.round(3).to_string())

print("\n\n=== 기업별 상관계수 TOP 3 ===")
for corp in panel["corp_name"].unique():
    sub  = panel[panel["corp_name"] == corp][risk + macro]
    corr = sub.corr().loc[risk, macro]
    top  = corr.abs().stack().sort_values(ascending=False).head(3)
    print(f"\n── {corp} ──")
    for (r, m), v in top.items():
        actual = corr.loc[r, m]
        print(f"  {RISK_LABELS[r]:10s} ↔ {MACRO_LABELS[m]:8s} : {actual:+.3f}")
