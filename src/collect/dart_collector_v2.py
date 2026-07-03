import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
DART_API_KEY = os.getenv("DART_API_KEY")

# ── 20개 자동차부품 기업 ──────────────────────────────
COMPANIES = {
    "현대모비스":   "00164788",
    "HL만도":       "01042775",
    "현대위아":     "00106623",
    "한온시스템":   "00161125",
    "서연이화":     "01036446",
    "평화산업":     "00573579",
    "에스엘":       "00125521",
    "세원정공":     "00134316",
    "동원금속":     "00118008",
    "명신산업":     "00173078",
    "화신":         "00166315",
    "화신정공":     "00825223",
    "화신테크":     "00166333",
    "성우하이텍":   "00132992",
    "인지컨트롤스": "00103510",
    "디와이":       "00117179",
    "디와이파워":   "01059605",
    "엠에스오토텍": "00259545",
    "캐프":         "00876643",
    "코다코":       "00295857",
    "대원강업":     "00111847",
}

# 분기보고서 코드: 11013=1Q, 11012=반기(2Q), 11014=3Q, 11011=사업보고서(4Q=연간)
REPORT_CODES = {
    1: "11013",  # 1분기
    2: "11012",  # 반기
    3: "11014",  # 3분기
    4: "11011",  # 사업보고서(연간 누적)
}

ACCOUNT_MAP = {
    "자산총계":           "total_assets",
    "부채총계":           "total_liab",
    "자본총계":           "total_equity",
    "유동자산":           "current_assets",
    "유동부채":           "current_liab",
    "매출액":             "revenue",
    "수익(매출액)":       "revenue",
    "영업이익":           "op_income",
    "영업이익(손실)":     "op_income",
    "영업손익":           "op_income",
    "당기순이익":         "net_income",
    "당기순이익(손실)":   "net_income",
    "당기순손익":         "net_income",
    "분기순이익":         "net_income",
    "분기순이익(손실)":   "net_income",
    "분기순손익":         "net_income",
    "반기순이익":         "net_income",
    "반기순이익(손실)":   "net_income",
    "반기순손익":         "net_income",
    "이자비용":           "interest_exp",
    "금융원가":           "interest_exp",
    "금융비용":           "interest_exp",
    "이익잉여금":         "retained_earnings",
    "이익잉여금(결손금)": "retained_earnings",
    "결손금":             "retained_earnings",
}

def fetch_financials(corp_code: str, year: int, quarter: int) -> pd.DataFrame:
    report_code = REPORT_CODES[quarter]
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": report_code,
        "fs_div": "CFS",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()

    if data.get("status") != "000":
        params["fs_div"] = "OFS"
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

    if data.get("status") != "000" or not data.get("list"):
        return pd.DataFrame()

    rows = []
    for item in data["list"]:
        nm = item.get("account_nm", "").strip()
        if nm in ACCOUNT_MAP:
            val_str = item.get("thstrm_amount", "0").replace(",", "").replace(" ", "")
            try:
                val = float(val_str) if val_str else None
            except ValueError:
                val = None
            rows.append({"account": ACCOUNT_MAP[nm], "value": val})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates("account").set_index("account")

def normalize_quarterly_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    fnlttSinglAcntAll의 thstrm_amount는 Q1~Q3 손익계산서에서 해당
    3개월 값이고, 사업보고서(Q4)에서는 연간 값이다. 따라서 Q1~Q3은
    그대로 두고 Q4만 연간 값에서 앞선 세 분기를 차감한다.

    원 보고값은 ``*_reported`` 열에 보존한다. 앞선 분기가 빠진 연도는
    Q4 단일분기 값을 안전하게 계산할 수 없으므로 결측으로 둔다.
    """
    flow_cols = ["revenue", "op_income", "net_income", "interest_exp"]
    df = df.sort_values("quarter").reset_index(drop=True)

    for col in flow_cols:
        if col not in df.columns:
            continue
        reported_col = f"{col}_reported"
        df[reported_col] = df[col]
        q4_mask = df["quarter"].eq(4)
        if not q4_mask.any():
            continue
        prior = df[df["quarter"].isin([1, 2, 3])].set_index("quarter")[col]
        has_complete_prior = set(prior.dropna().index) == {1, 2, 3}
        if has_complete_prior:
            df.loc[q4_mask, col] = df.loc[q4_mask, reported_col] - prior.sum()
        else:
            df.loc[q4_mask, col] = None
    return df

def collect_all(years=None, companies=None):
    if years is None:
        years = list(range(2019, 2026))
    if companies is None:
        companies = COMPANIES

    records = []
    total = len(companies) * len(years) * 4
    done = 0

    for corp_name, corp_code in companies.items():
        corp_records = []
        for year in years:
            for q in [1, 2, 3, 4]:
                done += 1
                df = fetch_financials(corp_code, year, q)
                if df.empty:
                    print(f"  [{done}/{total}] {corp_name} {year}Q{q} - 데이터 없음")
                    continue

                record = {"corp_name": corp_name, "corp_code": corp_code,
                          "year": year, "quarter": q}
                for col in set(ACCOUNT_MAP.values()):
                    record[col] = df.loc[col, "value"] if col in df.index else None
                corp_records.append(record)
                print(f"  [{done}/{total}] {corp_name} {year}Q{q} - OK")
                time.sleep(0.25)

        if corp_records:
            corp_df = pd.DataFrame(corp_records)
            # 손익항목 누적값 → 단일분기값 변환 (연도별로)
            fixed = []
            for yr in corp_df["year"].unique():
                yr_df = corp_df[corp_df["year"] == yr].copy()
                yr_df = normalize_quarterly_flows(yr_df)
                fixed.append(yr_df)
            corp_df = pd.concat(fixed, ignore_index=True)
            records.extend(corp_df.to_dict("records"))

    result = pd.DataFrame(records)
    os.makedirs("data/raw/dart", exist_ok=True)
    result.to_parquet("data/raw/dart/financials_quarterly.parquet", index=False)
    print(f"\n저장 완료: data/raw/dart/financials_quarterly.parquet ({len(result)}행)")
    return result

if __name__ == "__main__":
    df = collect_all()
    print(f"\n기업별 행 수:\n{df['corp_name'].value_counts()}")
