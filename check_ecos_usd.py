import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("ECOS_API_KEY")

# 환율 관련 통계표 검색
print("=== 환율 통계표 검색 ===")
url = f"https://ecos.bok.or.kr/api/StatisticTableList/{key}/json/kr/1/20/환율"
resp = requests.get(url, timeout=10).json()
rows = resp.get("StatisticTableList", {}).get("row", [])
for r in rows:
    print(r.get("STAT_CODE"), "|", r.get("STAT_NAME"), "|", r.get("CYCLE"))

# 731Y001 항목코드 확인
print("\n=== 731Y001 항목 확인 ===")
url2 = f"https://ecos.bok.or.kr/api/StatisticItemList/{key}/json/kr/1/20/731Y001"
resp2 = requests.get(url2, timeout=10).json()
rows2 = resp2.get("StatisticItemList", {}).get("row", [])
for r in rows2[:10]:
    print(r.get("ITEM_CODE"), "|", r.get("ITEM_NAME"))
