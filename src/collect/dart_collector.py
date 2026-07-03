import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
DART_API_KEY = os.getenv("DART_API_KEY")

COMPANIES = {
    "현대모비스": "00164788",
    "HL만도":     "01042775",
    "평화산업":   "00573579",
    "한온시스템": "00161125",
    "서연이화":   "01036446",
}
ACCOUNT_MAP = {
    "자산총계":           "total_assets",
    "부채총계":           "total_liab",
    "자본총계":           "total_equity",
    "유동자산":           "current_assets",
    "유동부채":           "current_liab",
    "매출액":             "revenue",
    "수익(매출액)":       "revenue",        # 한온시스템 일부연도
    "영업이익":           "op_income",
    "영업이익(손실)":     "op_income",
    "영업손익":           "op_income",      # 한온시스템 일부연도
    "당기순이익":         "net_income",
    "당기순이익(손실)":   "net_income",
    "당기순손익":         "net_income",     # 한온시스템 일부연도
    "이자비용":           "interest_exp",
    "금융원가":           "interest_exp",
    "금융비용":           "interest_exp",
    "이익잉여금":         "retained_earnings",
    "이익잉여금(결손금)": "retained_earnings",
    "결손금":             "retained_earnings",
}
def fetch_financials(corp_code: str, year: int, report_code: str = "11011") -> pd.DataFrame:
    """
    report_code:
      11011 = 사업보고서(연간)
      11012 = 반기보고서
      11013 = 1분기보고서
      11014 = 3분기보고서
    """
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": report_code,
        "fs_div": "CFS",  # 연결재무제표 (없으면 OFS 개별로 fallback)
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()

    if data.get("status") != "000":
        # 연결재무제표 없는 경우 개별재무제표로 재시도
        params["fs_div"] = "OFS"
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

    if data.get("status") != "000" or not data.get("list"):
        print(f"  [경고] {corp_code} {year}년 데이터 없음: {data.get('message','')}")
        return pd.DataFrame()

    rows = []
    for item in data["list"]:
        account_nm = item.get("account_nm", "").strip()
        if account_nm in ACCOUNT_MAP:
            val_str = item.get("thstrm_amount", "0").replace(",", "").replace(" ", "")
            try:
                val = float(val_str) if val_str else None
            except ValueError:
                val = None
            rows.append({"account": ACCOUNT_MAP[account_nm], "value": val})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("account").set_index("account")
    return df

def collect_all(years: list = None) -> pd.DataFrame:
    if years is None:
        years = list(range(2019, 2025))

    records = []
    for corp_name, corp_code in COMPANIES.items():
        for year in years:
            print(f"수집중: {corp_name} {year}...")
            df = fetch_financials(corp_code, year)
            if df.empty:
                continue

            record = {"corp_name": corp_name, "corp_code": corp_code, "year": year}
            for col in ACCOUNT_MAP.values():
                record[col] = df.loc[col, "value"] if col in df.index else None
            records.append(record)
            time.sleep(0.3)  # API 호출 간격

    result = pd.DataFrame(records)
    os.makedirs("data/raw/dart", exist_ok=True)
    result.to_parquet("data/raw/dart/financials.parquet", index=False)
    print(f"\n저장 완료: data/raw/dart/financials.parquet ({len(result)}행)")
    return result

if __name__ == "__main__":
    df = collect_all()
    print(df.head(10).to_string())
