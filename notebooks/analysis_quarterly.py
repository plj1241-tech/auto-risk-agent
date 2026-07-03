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

panel = pd.read_parquet("data/processed/panel_quarterly.parquet")
panel["period"] = panel["year"].astype(str) + "Q" + panel["quarter"].astype(str)

RISK_COLS  = ["debt_ratio", "current_ratio", "icr", "op_margin", "z_score"]
MACRO_COLS_REG = ["base_rate_kr", "usd_krw", "ppi_us", "wti_oil"]  # 회귀용 (다중공선성 줄임)
ALL_MACRO      = ["base_rate_kr", "usd_krw", "ppi_us", "wti_oil", "iron_ore_price"]

RISK_LABELS = {
    "debt_ratio": "부채비율", "current_ratio": "유동비율", "icr": "이자보상배율",
    "op_margin": "영업이익률", "z_score": "Z-score",
}
MACRO_LABELS = {
    "base_rate_kr": "한국금리", "usd_krw": "환율", "fed_rate": "미국금리",
    "ppi_us": "PPI", "wti_oil": "유가", "iron_ore_price": "철광석",
}

os.makedirs("data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════
# 1. EDA — 전체 추이 + 상관관계
# ══════════════════════════════════════════════════════
print("="*60); print("1. EDA — 분기 데이터 탐색"); print("="*60)

# 1-1. 전체 평균 리스크지표 추이 (21개사 평균)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
avg_by_period = panel.groupby(["year","quarter"])[RISK_COLS].mean().reset_index()
avg_by_period["period_num"] = avg_by_period["year"] + (avg_by_period["quarter"]-1)/4

for i, col in enumerate(RISK_COLS):
    ax = axes[i]
    ax.plot(avg_by_period["period_num"], avg_by_period[col],
            marker="o", color="#185FA5", linewidth=2, markersize=4)
    ax.fill_between(avg_by_period["period_num"], avg_by_period[col], alpha=0.1, color="#185FA5")
    ax.set_title(f"{RISK_LABELS[col]} — 21개사 평균", fontsize=12, fontweight="bold")
    ax.axvline(2020, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(2022, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
axes[-1].set_visible(False)
plt.suptitle("자동차부품 업계 평균 리스크지표 추이 (2019Q1~2024Q4)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/outputs/q01_avg_risk_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: q01_avg_risk_timeseries.png")

# 1-2. 상관행렬 히트맵
corr_cols = RISK_COLS + ALL_MACRO
corr_df = panel[corr_cols].corr()
labels = {**RISK_LABELS, **MACRO_LABELS}
corr_df.index = [labels.get(c,c) for c in corr_df.index]
corr_df.columns = [labels.get(c,c) for c in corr_df.columns]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size":8})
ax.set_title("리스크지표 ↔ 거시경제지표 상관행렬 (분기 데이터, n=492)",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("data/outputs/q02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: q02_correlation_heatmap.png")

# ══════════════════════════════════════════════════════
# 2. 패널 고정효과 회귀
# ══════════════════════════════════════════════════════
print("\n"+"="*60); print("2. 패널 고정효과 회귀 (분기, entity=기업)"); print("="*60)

from linearmodels.panel import PanelOLS
import statsmodels.api as sm

panel_reg = panel.copy()
panel_reg["period_idx"] = panel_reg["year"]*4 + panel_reg["quarter"]
panel_reg = panel_reg.set_index(["corp_name", "period_idx"])

results_summary = []
for risk in RISK_COLS:
    try:
        y = panel_reg[risk].dropna()
        X = panel_reg[MACRO_COLS_REG].loc[y.index].dropna()
        y = y.loc[X.index]

        model  = PanelOLS(y, X, entity_effects=True, drop_absorbed=True, check_rank=False)
        result = model.fit(cov_type="clustered", cluster_entity=True)

        print(f"\n── {RISK_LABELS[risk]} (n={len(y)}, R²={result.rsquared_within:.3f}) ──")
        for var in MACRO_COLS_REG:
            if var in result.params.index:
                coef, pval = result.params[var], result.pvalues[var]
                sig = "***" if pval<0.01 else "**" if pval<0.05 else "*" if pval<0.1 else ""
                print(f"  {MACRO_LABELS[var]:8s}: β={coef:+.4f}  p={pval:.3f} {sig}")
                results_summary.append({
                    "리스크지표": RISK_LABELS[risk], "거시변수": MACRO_LABELS[var],
                    "β계수": round(coef,4), "p값": round(pval,3), "유의성": sig
                })
    except Exception as e:
        print(f"  [{risk}] 오류: {e}")

reg_df = pd.DataFrame(results_summary)
reg_df.to_csv("data/outputs/q_panel_regression_results.csv", index=False, encoding="utf-8-sig")

if not reg_df.empty:
    pivot = reg_df.pivot(index="리스크지표", columns="거시변수", values="β계수")
    sig_pivot = reg_df.pivot(index="리스크지표", columns="거시변수", values="유의성").fillna("")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size":10})
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            sig = sig_pivot.iloc[i,j]
            if sig:
                ax.text(j+0.8, i+0.25, sig, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("패널 고정효과 회귀 β계수 (분기 데이터)\n(* p<0.1  ** p<0.05  *** p<0.01)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("data/outputs/q03_regression_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n저장: q03_regression_heatmap.png")

# ══════════════════════════════════════════════════════
# 3. XGBoost + Walk-forward (분기 단위)
# ══════════════════════════════════════════════════════
print("\n"+"="*60); print("3. XGBoost 예측 모델 (분기, Walk-forward)"); print("="*60)

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

le = LabelEncoder()
panel["corp_code_enc"] = le.fit_transform(panel["corp_name"])
panel["period_idx"] = panel["year"]*4 + panel["quarter"]

FEATURE_COLS = (ALL_MACRO
    + [f"{c}_lag1" for c in ALL_MACRO if f"{c}_lag1" in panel.columns]
    + ["corp_code_enc"])
FEATURE_COLS = [c for c in FEATURE_COLS if c in panel.columns]

final_models = {}
validation_rows = []
for risk in RISK_COLS:
    df = panel[["corp_name","period_idx",risk] + FEATURE_COLS].dropna()
    if risk == "debt_ratio":
        df = df[df[risk] >= 0].copy()
    periods = sorted(df["period_idx"].unique())

    prediction_frames = []
    # 처음 8개 분기(2년) 학습 → 이후 1개씩 예측
    for test_p in periods[8:]:
        train = df[df["period_idx"] < test_p]
        test  = df[df["period_idx"] == test_p]
        if len(train) < 10 or len(test) == 0:
            continue
        m = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08,
                              random_state=42, verbosity=0)
        train_target = np.log1p(train[risk]) if risk == "debt_ratio" else train[risk]
        m.fit(train[FEATURE_COLS], train_target)
        predictions = m.predict(test[FEATURE_COLS])
        if risk == "debt_ratio":
            predictions = np.expm1(predictions)
        prediction_frames.append(
            test[["corp_name", "period_idx", risk]].assign(model_prediction=predictions)
        )

    if prediction_frames:
        predicted = pd.concat(prediction_frames, ignore_index=True)
        mae = mean_absolute_error(predicted[risk], predicted["model_prediction"])
        r2 = r2_score(predicted[risk], predicted["model_prediction"])
        prior = df[["corp_name", "period_idx", risk]].copy()
        prior["period_idx"] += 1
        prior = prior.rename(columns={risk: "persistence_prediction"})
        comparable = predicted.merge(prior, on=["corp_name", "period_idx"], how="inner")
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
            f"동일표본 R²={model_r2_comparable:.3f}, 직전분기 기준선={persistence_r2:.3f}"
        )

    m_final = xgb.XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05,
                                random_state=42, verbosity=0)
    final_target = np.log1p(df[risk]) if risk == "debt_ratio" else df[risk]
    m_final.fit(df[FEATURE_COLS], final_target)
    final_models[risk] = m_final

validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv("data/outputs/q_model_validation.csv", index=False, encoding="utf-8-sig")

with open("models/risk_predictor_quarterly.pkl", "wb") as f:
    pickle.dump({
        "models": final_models,
        "features": FEATURE_COLS,
        "label_encoder": le,
        "debt_ratio_log_transformed": True,
        "validation": validation_rows,
    }, f)
print("\n모델 저장: models/risk_predictor_quarterly.pkl")

# ══════════════════════════════════════════════════════
# 4. SHAP 분석
# ══════════════════════════════════════════════════════
print("\n"+"="*60); print("4. SHAP 분석"); print("="*60)

import shap

df_shap = panel[["corp_name","period_idx"] + RISK_COLS + FEATURE_COLS].dropna()
X_shap = df_shap[FEATURE_COLS]

feat_labels = {}
for c in FEATURE_COLS:
    base = c.replace("_lag1","")
    lag = "(전분기)" if "_lag1" in c else ""
    feat_labels[c] = MACRO_LABELS.get(base, base) + lag
feat_labels["corp_code_enc"] = "기업"

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i, risk in enumerate(RISK_COLS):
    explainer = shap.TreeExplainer(final_models[risk])
    sv = explainer.shap_values(X_shap)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=True).tail(8)
    ax = axes[i]
    ax.barh([feat_labels.get(f,f) for f in mean_abs.index], mean_abs.values,
            color="#185FA5", edgecolor="white")
    ax.set_title(f"{RISK_LABELS[risk]} — SHAP 중요도", fontsize=11, fontweight="bold")
    ax.set_xlabel("평균 |SHAP값|")
    ax.grid(True, alpha=0.3, axis="x")
axes[-1].set_visible(False)
plt.suptitle("거시경제지표별 리스크지표 영향도 (SHAP, 분기 데이터)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/outputs/q04_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: q04_shap_importance.png")

# 기업별 비교 (상위 6개 기업, Z-score 기준)
top_corps = panel.groupby("corp_name")["total_assets"].mean().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
explainer_z = shap.TreeExplainer(final_models["z_score"])

for ax, corp in zip(axes, top_corps):
    sub = df_shap[df_shap["corp_name"]==corp][FEATURE_COLS]
    sv = explainer_z.shap_values(sub)
    mean_sv = pd.Series(sv.mean(axis=0), index=FEATURE_COLS)
    top = mean_sv.abs().sort_values(ascending=False).head(6)
    vals = mean_sv[top.index][::-1]
    colors = ["#185FA5" if v>0 else "#D85A30" for v in vals.values]
    ax.barh([feat_labels.get(f,f) for f in vals.index], vals.values,
            color=colors, edgecolor="white")
    ax.set_title(corp, fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("주요 기업별 Z-score에 대한 거시변수 영향 (자산규모 상위 6개사)\n(파랑=긍정, 빨강=부정)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/outputs/q05_shap_by_corp.png", dpi=150, bbox_inches="tight")
plt.close()
print("저장: q05_shap_by_corp.png")

print("\n"+"="*60)
print("분기 데이터 분석 완료!")
print("  q01_avg_risk_timeseries.png")
print("  q02_correlation_heatmap.png")
print("  q03_regression_heatmap.png")
print("  q04_shap_importance.png")
print("  q05_shap_by_corp.png")
print("  q_panel_regression_results.csv")
print("  q_model_validation.csv")
print("  models/risk_predictor_quarterly.pkl")
print("="*60)
