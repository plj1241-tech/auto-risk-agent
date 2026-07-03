import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("ECOS_API_KEY")

# 월간(M), 일간(D), 분기(Q) 모두 시도
for cycle, start, end in [
    ("M", "201901", "201903"),
    ("D", "20190101", "20190110"),
    ("Q", "2019Q1", "2019Q2"),
    ("A", "2019", "2020"),
]:
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr"
        f"/1/5/731Y001/{cycle}/{start}/{end}/0000001"
    )
    resp = requests.get(url, timeout=10).json()
    rows = resp.get("StatisticSearch", {}).get("row", [])
    status = resp.get("StatisticSearch", {}).get("list_total_count", 0)
    print(f"주기={cycle} → {status}건: {rows[:1]}")
