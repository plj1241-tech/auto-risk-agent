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

panel = pd.read_parquet("data/processed/panel_annual_v2.parquet")

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
reg_df.to_csv("data/outputs/panel_regression_results_v2.csv",
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
    plt.savefig("data/outputs/v2_05_panel_regression_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: data/outputs/v2_05_panel_regression_heatmap.png")

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
validation_rows = []

for risk in RISK_COLS:
    df    = panel[["corp_name","year",risk] + FEATURE_COLS].dropna()
    if risk == "debt_ratio":
        df = df[df[risk] >= 0].copy()
    years = sorted(df["year"].unique())
    prediction_frames = []

    for test_year in years[3:]:
        train = df[df["year"] < test_year]
        test  = df[df["year"] == test_year]
        if len(train) < 5 or len(test) == 0:
            continue
        m = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                              learning_rate=0.1, random_state=42, verbosity=0)
        train_target = np.log1p(train[risk]) if risk == "debt_ratio" else train[risk]
        m.fit(train[FEATURE_COLS], train_target)
        predictions = m.predict(test[FEATURE_COLS])
        if risk == "debt_ratio":
            predictions = np.expm1(predictions)
        prediction_frames.append(
            test[["corp_name", "year", risk]].assign(model_prediction=predictions)
        )

    if prediction_frames:
        predicted = pd.concat(prediction_frames, ignore_index=True)
        mae = mean_absolute_error(predicted[risk], predicted["model_prediction"])
        r2 = r2_score(predicted[risk], predicted["model_prediction"])

        prior = df[["corp_name", "year", risk]].copy()
        prior["year"] += 1
        prior = prior.rename(columns={risk: "persistence_prediction"})
        comparable = predicted.merge(prior, on=["corp_name", "year"], how="inner")
        model_r2_comparable = r2_score(comparable[risk], comparable["model_prediction"])
        persistence_r2 = r2_score(comparable[risk], comparable["persistence_prediction"])
        validation_rows.append({
            "target": risk,
            "label": RISK_LABELS[risk],
            "model_mae": mae,
            "model_r2": r2,
            "model_r2_comparable": model_r2_comparable,
            "persistence_r2": persistence_r2,
            "validation_n": len(predicted),
            "baseline_n": len(comparable),
        })
        print(
            f"  {RISK_LABELS[risk]:12s} → R²={r2:.3f}, "
            f"동일표본 R²={model_r2_comparable:.3f}, 직전연도 기준선={persistence_r2:.3f}"
        )

    # 전체 데이터로 최종 모델
    m_final = xgb.XGBRegressor(n_estimators=200, max_depth=3,
                                learning_rate=0.05, random_state=42, verbosity=0)
    final_target = np.log1p(df[risk]) if risk == "debt_ratio" else df[risk]
    m_final.fit(df[FEATURE_COLS], final_target)
    final_models[risk] = m_final

validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv("data/outputs/model_validation_v2.csv", index=False, encoding="utf-8-sig")

with open("models/risk_predictor_v2.pkl", "wb") as f:
    pickle.dump({"models": final_models,
                 "features": FEATURE_COLS,
                 "label_encoder": le,
                 "debt_ratio_log_transformed": True,
                 "validation": validation_rows}, f)
print("모델 저장: models/risk_predictor_v2.pkl")

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
plt.savefig("data/outputs/v2_06_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: data/outputs/v2_06_shap_importance.png")

# 기업별 Z-score SHAP 비교 — 자산규모 상위 6개사
top_corps = panel.groupby("corp_name")["total_assets"].mean().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
explainer_z = shap.TreeExplainer(final_models["z_score"])

for ax, corp in zip(axes, top_corps):
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

plt.suptitle("주요 기업별 Z-score에 대한 거시변수 영향 (자산규모 상위 6개사)\n(파랑=긍정, 빨강=부정)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/outputs/v2_07_shap_by_corp.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: data/outputs/v2_07_shap_by_corp.png")

# 관측치별 SHAP 방향과 변수값 분포를 함께 보여주는 beeswarm
beeswarm_values = explainer_z.shap_values(X_shap)
beeswarm_frame = X_shap.rename(columns=feat_labels)
plt.figure(figsize=(11, 7))
shap.summary_plot(
    beeswarm_values,
    beeswarm_frame,
    plot_type="dot",
    max_display=10,
    show=False,
)
plt.title("Z-score SHAP beeswarm — 관측치별 변수 기여")
plt.tight_layout()
plt.savefig("data/outputs/v2_08_shap_beeswarm_z_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: data/outputs/v2_08_shap_beeswarm_z_score.png")

print("\n" + "=" * 60)
print("모든 분석 완료!")
print("  data/outputs/v2_05_panel_regression_heatmap.png")
print("  data/outputs/v2_06_shap_importance.png")
print("  data/outputs/v2_07_shap_by_corp.png")
print("  data/outputs/v2_08_shap_beeswarm_z_score.png")
print("  data/outputs/panel_regression_results_v2.csv")
print("  data/outputs/model_validation_v2.csv")
print("  models/risk_predictor_v2.pkl")
print("=" * 60)
