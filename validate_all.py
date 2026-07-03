"""
프로젝트 전체 정합성 검증 스크립트
실행: python validate_all.py
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

PASS = "✓ PASS"
FAIL = "✗ FAIL"
WARN = "△ WARN"

results = []

def check(name, condition, detail="", warn_only=False):
    status = (WARN if warn_only else FAIL) if not condition else PASS
    results.append({"검증항목": name, "결과": status, "상세": detail})
    print(f"{status}  {name}")
    if detail:
        print(f"      → {detail}")

print("=" * 60)
print("레이어 1: 데이터 정합성")
print("=" * 60)

# 1-1. 파일 존재 확인
import os
files = {
    "DART 분기 원본":     "data/raw/dart/financials_quarterly.parquet",
    "거시지표 원본":       "data/raw/macro/macro_quarterly.parquet",
    "연간 패널(최종)":    "data/processed/panel_annual_v2.parquet",
    "XGBoost 모델":       "models/risk_predictor_v2.pkl",
}
for name, path in files.items():
    exists = os.path.exists(path)
    check(f"파일 존재: {name}", exists, path if not exists else "")

# 1-2. 패널 기본 구조
panel = pd.read_parquet("data/processed/panel_annual_v2.parquet")
check("패널 행 수 (100행 이상)", len(panel) >= 100, f"현재 {len(panel)}행")
check("기업 수 (20개 이상)", panel['corp_name'].nunique() >= 20,
      f"현재 {panel['corp_name'].nunique()}개사")
check("연도 범위 (2019~2024)", set(panel['year'].unique()) >= {2019,2020,2021,2022,2023,2024},
      f"현재 {sorted(panel['year'].unique())}")
check("필수 컬럼 존재",
      all(c in panel.columns for c in ['debt_ratio','current_ratio','icr','op_margin','z_score']),
      "")

# 1-3. NaN 현황
nan_counts = panel[['debt_ratio','current_ratio','icr','op_margin','z_score']].isnull().sum()
total_nan = nan_counts.sum()
check("리스크지표 NaN 비율 30% 미만",
      total_nan / (len(panel)*5) < 0.3,
      f"NaN 총 {total_nan}개: {nan_counts.to_dict()}")

# 1-4. 현대모비스 2024년 부채비율 검증 (DART 공시 기준 약 44%)
mob_2024 = panel[(panel['corp_name']=='현대모비스') & (panel['year']==2024)]
if not mob_2024.empty:
    dr = mob_2024['debt_ratio'].values[0]
    check("현대모비스 2024 부채비율 합리적 (30~60%)", 30 <= dr <= 60,
          f"현재 {dr:.2f}% (DART 공시 기준 약 44%)")

# 1-5. 코다코 자본잠식 확인 (2023년 자본 음수)
kodako_2023 = panel[(panel['corp_name']=='코다코') & (panel['year']==2023)]
if not kodako_2023.empty:
    eq = kodako_2023['total_equity'].values[0]
    check("코다코 2023 완전자본잠식 확인", eq < 0,
          f"자본총계 {eq/1e8:.1f}억원 (음수=완전자본잠식)")

print()
print("=" * 60)
print("레이어 2: 지표 계산 정합성")
print("=" * 60)

# 2-1. 부채비율 재계산 검증
panel['debt_ratio_check'] = panel['total_liab'] / panel['total_equity'] * 100
diff = (panel['debt_ratio'] - panel['debt_ratio_check']).abs()
check("부채비율 재계산 일치 (오차 0.01% 미만)",
      diff.dropna().max() < 0.01,
      f"최대 오차: {diff.dropna().max():.6f}%")

# 2-2. 유동비율 재계산
panel['current_ratio_check'] = panel['current_assets'] / panel['current_liab'] * 100
diff2 = (panel['current_ratio'] - panel['current_ratio_check']).abs()
check("유동비율 재계산 일치",
      diff2.dropna().max() < 0.01,
      f"최대 오차: {diff2.dropna().max():.6f}%")

# 2-3. Z-score 등급 일관성
grade_map = panel['z_score'].apply(
    lambda z: "안전" if z > 2.6 else ("주의" if z > 1.1 else ("위험" if pd.notna(z) else "N/A"))
)
mismatch = (grade_map != panel['z_grade']).sum()
check("Z-score 등급 라벨 일관성",
      mismatch == 0,
      f"불일치 {mismatch}건")

# 2-4. 현대모비스 전 기간 안전 등급 확인
mob = panel[panel['corp_name']=='현대모비스']
mob_safe = (mob['z_grade'] == '안전').all()
check("현대모비스 전 기간 안전 등급",
      mob_safe,
      f"등급: {mob[['year','z_grade']].set_index('year')['z_grade'].to_dict()}")

# 2-5. 평화산업 누적 결손 확인 (이익잉여금 음수)
pyh = panel[panel['corp_name']=='평화산업']
neg_retained = (pyh['retained_earnings'] < 0).any()
check("평화산업 누적 결손 확인 (이익잉여금 음수 존재)",
      neg_retained, warn_only=True,
      detail=f"음수 연도: {pyh[pyh['retained_earnings']<0]['year'].tolist()}")

print()
print("=" * 60)
print("레이어 3: 모델 정합성")
print("=" * 60)

# 3-1. 모델 파일 로드
try:
    with open("models/risk_predictor_v2.pkl", "rb") as f:
        saved = pickle.load(f)
    models = saved["models"]
    features = saved["features"]
    le = saved["label_encoder"]
    check("모델 pkl 로드 성공", True, f"모델 수: {len(models)}개")
except Exception as e:
    check("모델 pkl 로드 성공", False, str(e))
    models = {}

# 3-2. 피처 컬럼 존재 확인
if models:
    feat_cols = [f for f in features if f != 'corp_code_enc']
    missing_feats = [f for f in feat_cols if f not in panel.columns]
    check("모델 피처 컬럼 존재", len(missing_feats)==0,
          f"누락 피처: {missing_feats}" if missing_feats else "")

# 3-3. 시나리오 방향성 검증 — 금리 상승 시 Z-score 방향
# 금리가 높으면 Z-score가 올라가는 경향 (회귀에서 β>0 확인됨)
if models and 'z_score' in models:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    # corp_code_enc는 패널에 없으므로 직접 생성
    df_test = panel[['corp_name'] + [f for f in features if f != 'corp_code_enc']].dropna().copy()
    df_test['corp_code_enc'] = le.transform(df_test['corp_name'])

    # 현대모비스 기준으로 금리 변화 테스트
    mob_row = df_test[df_test['corp_name']=='현대모비스'].iloc[-1:].copy()
    if not mob_row.empty and 'base_rate_kr' in mob_row.columns:
        base_pred = models['z_score'].predict(mob_row[features])[0]
        mob_high = mob_row.copy()
        mob_high['base_rate_kr'] = mob_row['base_rate_kr'].values[0] + 1.0
        high_pred = models['z_score'].predict(mob_high[features])[0]
        delta = high_pred - base_pred
        check("금리 +1% → Z-score 방향 합리적 (양수 또는 미미한 변화)",
              delta > -0.5,
              f"금리 +1%p시 Z-score 변화: {delta:+.4f}")

# 3-4. 부채비율 로그변환 플래그 확인
log_flag = saved.get("debt_ratio_log_transformed", False)
check("부채비율 로그변환 플래그 존재", log_flag,
      "predict 시 np.expm1() 역변환 필요" if log_flag else "플래그 없음 → 역변환 누락 위험")

print()
print("=" * 60)
print("레이어 4: 거시지표 정합성")
print("=" * 60)

# 4-1. 거시지표 범위 확인
macro_checks = {
    "base_rate_kr": (0.0, 5.0, "한국 기준금리(%)"),
    "usd_krw":      (1000, 1500, "USD/KRW 환율"),
    "ppi_us":       (150, 320, "미국 PPI"),
    "wti_oil":      (20, 120, "WTI 유가"),
    "iron_ore_price": (50, 250, "철광석 가격"),
}
for col, (lo, hi, label) in macro_checks.items():
    if col in panel.columns:
        vals = panel[col].dropna()
        in_range = vals.between(lo, hi).all()
        check(f"{label} 범위 합리적 ({lo}~{hi})",
              in_range,
              f"실제 범위: {vals.min():.1f} ~ {vals.max():.1f}")

# 4-2. 2022년 금리 급등 확인
rate_2022 = panel[panel['year']==2022]['base_rate_kr'].iloc[0] if len(panel[panel['year']==2022]) > 0 else None
rate_2020 = panel[panel['year']==2020]['base_rate_kr'].iloc[0] if len(panel[panel['year']==2020]) > 0 else None
if rate_2022 and rate_2020:
    check("2022년 금리 > 2020년 금리 (금리급등 이벤트 반영)",
          rate_2022 > rate_2020,
          f"2020년: {rate_2020:.2f}% → 2022년: {rate_2022:.2f}%")

print()
print("=" * 60)
print("최종 요약")
print("=" * 60)
df_results = pd.DataFrame(results)
pass_cnt = (df_results['결과'] == PASS).sum()
fail_cnt = (df_results['결과'] == FAIL).sum()
warn_cnt = (df_results['결과'] == WARN).sum()
total = len(df_results)
print(f"PASS: {pass_cnt}/{total}  |  FAIL: {fail_cnt}  |  WARN: {warn_cnt}")
print()
if fail_cnt > 0:
    print("[수정 필요]")
    for _, row in df_results[df_results['결과']==FAIL].iterrows():
        print(f"  - {row['검증항목']}: {row['상세']}")
if warn_cnt > 0:
    print("[참고 사항]")
    for _, row in df_results[df_results['결과']==WARN].iterrows():
        print(f"  - {row['검증항목']}: {row['상세']}")

df_results.to_csv("data/outputs/validation_report.csv", index=False, encoding="utf-8-sig")
print()
print("상세 결과 저장: data/outputs/validation_report.csv")
