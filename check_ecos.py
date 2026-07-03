import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("ECOS_API_KEY")

# 기준금리 통계표 검색
print("=== 기준금리 검색 ===")
url = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/10/722Y001/A/2023/2023"
resp = requests.get(url, timeout=10).json()
print(resp)

print("\n=== 통계목록에서 기준금리 찾기 ===")
url2 = f"https://ecos.bok.or.kr/api/StatisticTableList/{key}/json/kr/1/20/기준금리"
resp2 = requests.get(url2, timeout=10).json()
rows = resp2.get("StatisticTableList", {}).get("row", [])
for r in rows:
    print(r.get("STAT_CODE"), r.get("STAT_NAME"), r.get("CYCLE"))
