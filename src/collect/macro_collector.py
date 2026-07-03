import os
import requests
import pandas as pd
import fredapi
from dotenv import load_dotenv

load_dotenv()
ECOS_KEY = os.getenv("ECOS_API_KEY")
FRED_KEY = os.getenv("FRED_API_KEY")

def fetch_ecos_annual(stat_code, item_code, label, start="2019", end="2024"):
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr"
        f"/1/100/{stat_code}/A/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=10).json()
    rows = resp.get("StatisticSearch", {}).get("row", [])
    if not rows:
        print(f"  [경고] ECOS {label} 데이터 없음")
        return pd.Series(dtype=float, name=label)
    records = [{"year": int(r["TIME"]), label: float(r["DATA_VALUE"].replace(",", "") or 0)} for r in rows]
    return pd.DataFrame(records).set_index("year")[label]

def fetch_ecos_daily(stat_code, item_code, label, start="20190101", end="20241231"):
    """일간 ECOS 데이터 → 연간 평균"""
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr"
        f"/1/3000/{stat_code}/D/{start}/{end}/{item_code}"
    )
    resp = requests.get(url, timeout=10).json()
    rows = resp.get("StatisticSearch", {}).get("row", [])
    if not rows:
        print(f"  [경고] ECOS {label} 데이터 없음")
        return pd.Series(dtype=float, name=label)

    records = []
    for r in rows:
        try:
            val = float(r["DATA_VALUE"].replace(",", "").strip())
            year = int(r["TIME"][:4])
            records.append({"year": year, label: val})
        except (ValueError, AttributeError):
            continue

    annual = pd.DataFrame(records).groupby("year")[label].mean()
    return annual

def fetch_fred_annual(series_id, label, start="2019-01-01", end="2024-12-31"):
    fred = fredapi.Fred(api_key=FRED_KEY)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    annual = s.resample("YE").mean()
    annual.index = annual.index.year
    annual.index.name = "year"
    annual.name = label
    return annual

def collect_macro():
    print("거시경제지표 수집 시작...")

    # ECOS 기준금리 (연간)
    rate_kr = fetch_ecos_annual("722Y001", "0101000", "base_rate_kr")

    # ECOS 환율 USD/KRW (일간 → 연간 평균)
    usdkrw = fetch_ecos_daily("731Y001", "0000001", "usd_krw")

    # FRED (연간 평균)
    rate_us = fetch_fred_annual("FEDFUNDS",    "fed_rate")
    ppi     = fetch_fred_annual("PPIACO",      "ppi_us")
    wti     = fetch_fred_annual("DCOILWTICO",  "wti_oil")
    steel   = fetch_fred_annual("PIORECRUSDM", "iron_ore_price")

    macro = pd.DataFrame({
        "base_rate_kr":   rate_kr,
        "usd_krw":        usdkrw,
        "fed_rate":       rate_us,
        "ppi_us":         ppi,
        "wti_oil":        wti,
        "iron_ore_price": steel,
    })
    macro.index.name = "year"
    macro = macro.loc[2019:2024]
    macro = macro.interpolate(method="linear").ffill().bfill()

    os.makedirs("data/raw/macro", exist_ok=True)
    macro.to_parquet("data/raw/macro/macro.parquet")
    print(f"\n저장 완료: data/raw/macro/macro.parquet ({len(macro)}행)")
    print(macro.to_string())
    return macro

if __name__ == "__main__":
    collect_macro()
