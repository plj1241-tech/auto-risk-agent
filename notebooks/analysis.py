import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

panel = pd.read_parquet("data/processed/panel.parquet")

RISK_COLS  = ["debt_ratio", "current_ratio", "icr", "op_margin", "z_score"]

# 다중공선성 줄이기 위해 대표 변수만 선택
# 금리는 한국금리만, 물가는 PPI만, 원자재는 유가만, 환율 포함
MACRO_COLS = ["base_rate_kr", "usd_krw", "ppi_us", "wti_oil"]

RISK_LABELS = {
    "debt_ratio":    "부채비율",
    "current_ratio": "유동비율",
    "icr":           "이자보상배율",
    "op_margin":     "영업이익률",
    "z_score":       "Z-score",
}
MACRO_LABELS = {
    "base_rate_kr": "한국금리",
    "usd_krw":      "환율",
    "ppi_us":       "PPI",
    "wti_oil":      "유가",
}

os.makedirs("data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════
# 1. 패널 고정효과 회귀
# ══════════════════════════════════════════════════════
print("=" * 60)
print("1. 패널 고정효과 회귀 분석")
print("=" * 60)

from linearmodels.panel import PanelOLS
import statsmodels.api as sm

panel_reg = panel.copy().set_index(["corp_name", "year"])
results_summary = []

for risk in RISK_COLS:
    try:
        y = panel_reg[risk].dropna()
        X = panel_reg[MACRO_COLS].loc[y.index].dropna()
        y = y.loc[X.index]

        model  = PanelOLS(y, X, entity_effects=True,
                          drop_absorbed=True, check_rank=False)
        result = model.fit(cov_type="clustered", cluster_entity=True)

        print(f"\n── {RISK_LABELS[risk]} (R²={result.rsquared_within:.3f}) ──")
        for var in MACRO_COLS:
            if var in result.params.index:
                coef = result.params[var]
                pval = result.pvalues[var]
                sig  = "***" if pval<0.01 else "**" if pval<0.05 else "*" if pval<0.1 else ""
                print(f"  {MACRO_LABELS[var]:8s}: β={coef:+.4f}  p={pval:.3f} {sig}")
                results_summary.append({
                    "리스크지표": RISK_LABELS[risk],
                    "거시변수":   MACRO_LABELS[var],
                    "β계수":     round(coef, 4),
                    "p값":       round(pval, 3),
                    "유의성":    sig,
                })
    except Exception as e:
        print(f"  [{risk}] 오류: {e}")

reg_df = pd.DataFrame(results_summary)
reg_df.to_csv("data/outputs/panel_regression_results.csv",
              index=False, encoding="utf-8-sig")
print("\n패널 회귀 결과 저장 완료")

# 히트맵
if not reg_df.empty:
    pivot     = reg_df.pivot(index="리스크지표", columns="거시변수", values="β계수")
    sig_pivot = reg_df.pivot(index="리스크지표", columns="거시변수", values="유의성").fillna("")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 10})
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            sig = sig_pivot.iloc[i, j]
            if sig:
                ax.text(j+0.8, i+0.25, sig, ha="center", va="center",
                        fontsize=9, fontweight="bold")
    ax.set_title("패널 고정효과 회귀 β계수\n(* p<0.1  ** p<0.05  *** p<0.01)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("data/outputs/05_panel_regression_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: data/outputs/05_panel_regression_heatmap.png")

# ══════════════════════════════════════════════════════
# 2. XGBoost + Walk-forward
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. XGBoost 예측 모델 (Walk-forward)")
print("=" * 60)

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

le = LabelEncoder()
panel["corp_code_enc"] = le.fit_transform(panel["corp_name"])

ALL_MACRO = ["base_rate_kr", "usd_krw", "ppi_us", "wti_oil", "iron_ore_price"]
FEATURE_COLS = (ALL_MACRO
    + [f"{c}_lag1" for c in ALL_MACRO if f"{c}_lag1" in panel.columns]
    + ["corp_code_enc"])
FEATURE_COLS = [c for c in FEATURE_COLS if c in panel.columns]

final_models = {}

for risk in RISK_COLS:
    df    = panel[["corp_name","year",risk] + FEATURE_COLS].dropna()
    years = sorted(df["year"].unique())
    preds_all, trues_all = [], []

    for test_year in years[3:]:
        train = df[df["year"] < test_year]
        test  = df[df["year"] == test_year]
        if len(train) < 5 or len(test) == 0:
            continue
        m = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                              learning_rate=0.1, random_state=42, verbosity=0)
        m.fit(train[FEATURE_COLS], train[risk])
        preds_all.extend(m.predict(test[FEATURE_COLS]))
        trues_all.extend(test[risk].values)

    if trues_all:
        mae = mean_absolute_error(trues_all, preds_all)
        r2  = r2_score(trues_all, preds_all) if len(trues_all) > 1 else float("nan")
        print(f"  {RISK_LABELS[risk]:12s} → MAE={mae:.3f}  R²={r2:.3f}")

    # 전체 데이터로 최종 모델
    m_final = xgb.XGBRegressor(n_estimators=200, max_depth=3,
                                learning_rate=0.05, random_state=42, verbosity=0)
    m_final.fit(df[FEATURE_COLS], df[risk])
    final_models[risk] = m_final

with open("models/risk_predictor.pkl", "wb") as f:
    pickle.dump({"models": final_models,
                 "features": FEATURE_COLS,
                 "label_encoder": le}, f)
print("모델 저장: models/risk_predictor.pkl")

# ══════════════════════════════════════════════════════
# 3. SHAP 분석
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. SHAP 분석")
print("=" * 60)

import shap

df_shap = panel[["corp_name","year"] + RISK_COLS + FEATURE_COLS].dropna()
X_shap  = df_shap[FEATURE_COLS]

feat_labels = {}
for c in FEATURE_COLS:
    base = c.replace("_lag1","")
    lag  = "(시차)" if "_lag1" in c else ""
    feat_labels[c] = MACRO_LABELS.get(base, base) + lag
feat_labels["corp_code_enc"] = "기업"
feat_labels["iron_ore_price"] = "철광석"
feat_labels["iron_ore_price_lag1"] = "철광석(시차)"

# 리스크지표별 SHAP 중요도
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, risk in enumerate(RISK_COLS):
    explainer = shap.TreeExplainer(final_models[risk])
    sv        = explainer.shap_values(X_shap)
    mean_abs  = pd.Series(np.abs(sv).mean(axis=0),
                          index=FEATURE_COLS).sort_values(ascending=True).tail(8)
    ax = axes[i]
    ax.barh([feat_labels.get(f,f) for f in mean_abs.index],
            mean_abs.values, color="#185FA5", edgecolor="white")
    ax.set_title(f"{RISK_LABELS[risk]} — SHAP 중요도",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("평균 |SHAP값|")
    ax.grid(True, alpha=0.3, axis="x")

axes[-1].set_visible(False)
plt.suptitle("거시경제지표별 리스크지표 영향도 (SHAP)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/outputs/06_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: data/outputs/06_shap_importance.png")

# 기업별 Z-score SHAP 비교
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
explainer_z = shap.TreeExplainer(final_models["z_score"])

for ax, corp in zip(axes, panel["corp_name"].unique()):
    sub  = df_shap[df_shap["corp_name"]==corp][FEATURE_COLS]
    sv   = explainer_z.shap_values(sub)
    mean_sv = pd.Series(sv.mean(axis=0), index=FEATURE_COLS)
    top  = mean_sv.abs().sort_values(ascending=False).head(6)
    vals = mean_sv[top.index][::-1]
    colors = ["#185FA5" if v>0 else "#D85A30" for v in vals.values]
    ax.barh([feat_labels.get(f,f) for f in vals.index],
            vals.values, color=colors, edgecolor="white")
    ax.set_title(corp, fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP값")
    ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("기업별 Z-score에 대한 거시변수 영향\n(파랑=긍정, 빨강=부정)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/outputs/07_shap_by_corp.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: data/outputs/07_shap_by_corp.png")

print("\n" + "=" * 60)
print("모든 분석 완료!")
print("  data/outputs/05_panel_regression_heatmap.png")
print("  data/outputs/06_shap_importance.png")
print("  data/outputs/07_shap_by_corp.png")
print("  data/outputs/panel_regression_results.csv")
print("  models/risk_predictor.pkl")
print("=" * 60)
