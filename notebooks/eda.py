import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

panel = pd.read_parquet("data/processed/panel.parquet")

RISK_COLS  = ["debt_ratio", "current_ratio", "icr", "op_margin", "z_score"]
MACRO_COLS = ["base_rate_kr", "usd_krw", "fed_rate", "ppi_us", "wti_oil", "iron_ore_price"]
CORP_COLORS = {
    "현대모비스": "#185FA5",
    "HL만도":     "#1D9E75",
    "평화산업":   "#D85A30",
    "한온시스템": "#7F77DD",
    "서연이화":   "#BA7517",
}
RISK_LABELS = {
    "debt_ratio":    "부채비율 (%)",
    "current_ratio": "유동비율 (%)",
    "icr":           "이자보상배율",
    "op_margin":     "영업이익률 (%)",
    "z_score":       "Altman Z-score",
}
MACRO_LABELS = {
    "base_rate_kr":   "한국 기준금리 (%)",
    "usd_krw":        "USD/KRW 환율",
    "fed_rate":       "미국 기준금리 (%)",
    "ppi_us":         "미국 PPI",
    "wti_oil":        "WTI 유가 ($/bbl)",
    "iron_ore_price": "철광석 가격 ($/MT)",
}

os.makedirs("data/outputs", exist_ok=True)

# ── 1. 기업별 리스크지표 시계열 ───────────────────────────
print("1. 기업별 리스크지표 시계열 차트 생성중...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(RISK_COLS):
    ax = axes[i]
    for corp, color in CORP_COLORS.items():
        data = panel[panel["corp_name"] == corp].sort_values("year")
        ax.plot(data["year"], data[col], marker="o", label=corp,
                color=color, linewidth=2, markersize=5)
    ax.set_title(RISK_LABELS[col], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("연도")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # 이벤트 표시
    ax.axvline(2020, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(2022, color="red",  linestyle="--", alpha=0.5, linewidth=1)
    ax.text(2020.05, ax.get_ylim()[1]*0.95, "COVID", fontsize=7, color="gray")
    ax.text(2022.05, ax.get_ylim()[1]*0.95, "금리급등", fontsize=7, color="red")

axes[-1].set_visible(False)
plt.suptitle("자동차 부품업계 기업별 리스크지표 추이 (2019~2024)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/outputs/01_risk_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("   저장: data/outputs/01_risk_timeseries.png")

# ── 2. 거시변수 시계열 ────────────────────────────────────
print("2. 거시경제지표 시계열 차트 생성중...")
macro_annual = panel[["year"] + MACRO_COLS].drop_duplicates("year").sort_values("year")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(MACRO_COLS):
    ax = axes[i]
    ax.plot(macro_annual["year"], macro_annual[col],
            marker="o", color="#185FA5", linewidth=2, markersize=6)
    ax.fill_between(macro_annual["year"], macro_annual[col],
                    alpha=0.1, color="#185FA5")
    ax.set_title(MACRO_LABELS[col], fontsize=12, fontweight="bold")
    ax.set_xlabel("연도")
    ax.grid(True, alpha=0.3)
    ax.axvline(2020, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(2022, color="red",  linestyle="--", alpha=0.5, linewidth=1)

plt.suptitle("거시경제지표 추이 (2019~2024)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/outputs/02_macro_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("   저장: data/outputs/02_macro_timeseries.png")

# ── 3. 상관행렬 히트맵 ───────────────────────────────────
print("3. 상관행렬 히트맵 생성중...")
corr_cols = RISK_COLS + MACRO_COLS
corr_df   = panel[corr_cols].corr()

# 라벨 한글화
all_labels = {**RISK_LABELS, **MACRO_LABELS}
corr_df.index   = [all_labels.get(c, c) for c in corr_df.index]
corr_df.columns = [all_labels.get(c, c) for c in corr_df.columns]

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)  # 상삼각 마스크
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("리스크지표 ↔ 거시경제지표 상관행렬", fontsize=14, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("data/outputs/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   저장: data/outputs/03_correlation_heatmap.png")

# ── 4. COVID·금리급등 전후 비교 (부채비율) ───────────────
print("4. 이벤트 전후 비교 차트 생성중...")
events = {
    "COVID 전후\n(2019 vs 2020)": (2019, 2020),
    "금리급등 전후\n(2021 vs 2022)": (2021, 2022),
}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (event_name, (y1, y2)) in zip(axes, events.items()):
    corps = list(CORP_COLORS.keys())
    x     = np.arange(len(corps))
    w     = 0.35

    vals1 = [panel[(panel["corp_name"]==c) & (panel["year"]==y1)]["debt_ratio"].values[0]
             for c in corps]
    vals2 = [panel[(panel["corp_name"]==c) & (panel["year"]==y2)]["debt_ratio"].values[0]
             for c in corps]

    bars1 = ax.bar(x - w/2, vals1, w, label=f"{y1}년", color="#B5D4F4", edgecolor="white")
    bars2 = ax.bar(x + w/2, vals2, w, label=f"{y2}년", color="#185FA5", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(corps, fontsize=10)
    ax.set_ylabel("부채비율 (%)")
    ax.set_title(f"부채비율 변화 — {event_name}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("주요 거시 이벤트 전후 부채비율 변화", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/outputs/04_event_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   저장: data/outputs/04_event_comparison.png")

# ── 5. 주요 상관관계 수치 출력 ───────────────────────────
print("\n=== 거시변수 → 리스크지표 주요 상관계수 ===")
raw_corr = panel[RISK_COLS + MACRO_COLS].corr()
for macro in MACRO_COLS:
    for risk in RISK_COLS:
        val = raw_corr.loc[risk, macro]
        if abs(val) >= 0.4:
            print(f"  {MACRO_LABELS[macro]:20s} ↔ {RISK_LABELS[risk]:15s} : {val:+.3f}")

print("\n모든 차트 생성 완료! data/outputs/ 폴더를 확인하세요.")
