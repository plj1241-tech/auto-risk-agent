import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")

panel = pd.read_parquet("data/processed/panel_annual_v2.parquet")

ALL_MACRO = ["base_rate_kr", "usd_krw", "ppi_us", "wti_oil", "iron_ore_price"]

le = LabelEncoder()
panel["corp_code_enc"] = le.fit_transform(panel["corp_name"])

FEATURE_COLS = (ALL_MACRO
    + [f"{c}_lag1" for c in ALL_MACRO if f"{c}_lag1" in panel.columns]
    + ["corp_code_enc"])
FEATURE_COLS = [c for c in FEATURE_COLS if c in panel.columns]

# ── 부채비율 로그변환 ────────────────────────────────────
# 부채비율은 항상 양수이고 기업간 스케일 차이가 커서 로그변환이 적합
df = panel[["corp_name","year","debt_ratio"] + FEATURE_COLS].copy()
df["log_debt_ratio"] = np.log1p(df["debt_ratio"])  # log(1+x), 0 방지
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["log_debt_ratio"] + FEATURE_COLS)

years = sorted(df["year"].unique())
preds_all, trues_all = [], []

for test_year in years[3:]:
    train = df[df["year"] < test_year]
    test  = df[df["year"] == test_year]
    if len(train) < 5 or len(test) == 0:
        continue
    m = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.08,
                          random_state=42, verbosity=0)
    m.fit(train[FEATURE_COLS], train["log_debt_ratio"])

    log_pred = m.predict(test[FEATURE_COLS])
    pred = np.expm1(log_pred)  # 역변환

    preds_all.extend(pred)
    trues_all.extend(test["debt_ratio"].values)

mae = mean_absolute_error(trues_all, preds_all)
r2  = r2_score(trues_all, preds_all)
print(f"부채비율 (로그변환) → MAE={mae:.3f}  R²={r2:.3f}")
print(f"(기존 로그변환 전: MAE=208.870  R²=-1.542)")

# 최종 모델 학습 및 저장
m_final = xgb.XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                            random_state=42, verbosity=0)
m_final.fit(df[FEATURE_COLS], df["log_debt_ratio"])

# 기존 pkl 불러와서 부채비율만 교체
with open("models/risk_predictor_v2.pkl", "rb") as f:
    saved = pickle.load(f)

saved["models"]["debt_ratio"] = m_final
saved["debt_ratio_log_transformed"] = True  # 예측시 np.expm1() 역변환 필요함을 표시

with open("models/risk_predictor_v2.pkl", "wb") as f:
    pickle.dump(saved, f)

print("\n모델 업데이트 완료: models/risk_predictor_v2.pkl")
print("주의: 부채비율 예측시 np.expm1(model.predict(X)) 로 역변환 필요")
