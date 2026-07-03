import os
import requests
import pandas as pd
import fredapi
from dotenv import load_dotenv

load_dotenv()
ECOS_KEY = os.getenv("ECOS_API_KEY")
FRED_KEY = os.getenv("FRED_API_KEY")

def fetch_ecos_daily_quarterly(stat_code, item_code, label,
                                start="20190101", end="20251231"):
    """일간 ECOS 데이터 → 분기 평균"""
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr"
        f"/1/5000/{stat_code}/D/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=15).json()
    rows = resp.get("StatisticSearch", {}).get("row", [])
    if not rows:
        print(f"  [경고] ECOS {label} 데이터 없음")
        return pd.DataFrame(columns=["year","quarter",label])

    records = []
    for r in rows:
        try:
            val  = float(r["DATA_VALUE"].replace(",","").strip())
            date = r["TIME"]
            year = int(date[:4])
            month = int(date[4:6])
            quarter = (month - 1) // 3 + 1
            records.append({"year": year, "quarter": quarter, label: val})
        except (ValueError, AttributeError):
            continue

    df = pd.DataFrame(records)
    quarterly = df.groupby(["year","quarter"])[label].mean().reset_index()
    return quarterly

def fetch_ecos_annual_to_quarterly(stat_code, item_code, label,
                                   start="2019", end="2025"):
    """연간 ECOS 데이터를 4분기에 동일값 적용 (기준금리처럼 자주 안변하는 지표)"""
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr"
        f"/1/100/{stat_code}/A/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=10).json()
    rows = resp.get("StatisticSearch", {}).get("row", [])
    if not rows:
        print(f"  [경고] ECOS {label} 데이터 없음")
        return pd.DataFrame(columns=["year","quarter",label])

    records = []
    for r in rows:
        year = int(r["TIME"])
        val  = float(r["DATA_VALUE"].replace(",", "") or 0)
        for q in [1,2,3,4]:
            records.append({"year": year, "quarter": q, label: val})
    return pd.DataFrame(records)

def fetch_fred_quarterly(series_id, label, start="2019-01-01", end="2025-12-31"):
    fred = fredapi.Fred(api_key=FRED_KEY)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    q = s.resample("QE").mean()
    df = pd.DataFrame({label: q})
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter
    return df[["year","quarter",label]].reset_index(drop=True)

def collect_macro_quarterly():
    print("거시경제지표(분기) 수집 시작...")

    rate_kr = fetch_ecos_annual_to_quarterly("722Y001", "0101000", "base_rate_kr")
    usdkrw  = fetch_ecos_daily_quarterly("731Y001", "0000001", "usd_krw")
    rate_us = fetch_fred_quarterly("FEDFUNDS",     "fed_rate")
    ppi     = fetch_fred_quarterly("PPIACO",       "ppi_us")
    wti     = fetch_fred_quarterly("DCOILWTICO",   "wti_oil")
    steel   = fetch_fred_quarterly("PIORECRUSDM",  "iron_ore_price")

    # 병합
    dfs = [rate_kr, usdkrw, rate_us, ppi, wti, steel]
    macro = dfs[0]
    for df in dfs[1:]:
        macro = macro.merge(df, on=["year","quarter"], how="outer")

    macro = macro.sort_values(["year","quarter"]).reset_index(drop=True)
    macro = macro[(macro["year"]>=2019) & (macro["year"]<=2025)]
    macro = macro.interpolate(method="linear").ffill().bfill()

    os.makedirs("data/raw/macro", exist_ok=True)
    macro.to_parquet("data/raw/macro/macro_quarterly.parquet", index=False)
    print(f"\n저장 완료: data/raw/macro/macro_quarterly.parquet ({len(macro)}행)")
    print(macro.head(10).to_string())
    return macro

if __name__ == "__main__":
    collect_macro_quarterly()
