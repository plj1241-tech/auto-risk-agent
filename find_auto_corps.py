import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("DART_API_KEY")

print("DART 전체 기업 목록 다운로드중...")
resp = requests.get(
    f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}"
)
z = zipfile.ZipFile(io.BytesIO(resp.content))
xml_data = z.read("CORPCODE.xml")
root = ET.fromstring(xml_data)

# 상장사만 (stock_code 있는 것)
corps = []
for corp in root.findall("list"):
    stock = corp.findtext("stock_code", "").strip()
    name  = corp.findtext("corp_name", "").strip()
    code  = corp.findtext("corp_code", "").strip()
    if stock:
        corps.append({"corp_name": name, "corp_code": code, "stock_code": stock})

df_all = pd.DataFrame(corps)
print(f"전체 상장사: {len(df_all)}개")

# 자동차부품 관련 키워드 필터링
keywords = [
    "모비스", "만도", "한온", "서연", "평화", "화신", "성우",
    "대원강업", "에스엘", "SL", "덕양산업", "세원정공",
    "현대위아", "현대다이모스", "인지컨트롤스", "동원금속",
    "명신산업", "삼기", "엠에스오토텍", "오스템",
    "코다코", "동성화인텍", "NVH코리아", "세종공업",
    "한국프랜지", "동양피스톤", "캐프", "일진",
    "성창오토텍", "디와이", "넥센", "한국타이어",
    "금호타이어", "효성첨단소재", "HL", "자동차부품"
]

mask = df_all["corp_name"].apply(
    lambda x: any(k in x for k in keywords)
)
df_auto = df_all[mask].copy()
print(f"\n자동차부품 관련 상장사 후보: {len(df_auto)}개")
print(df_auto[["corp_name", "corp_code", "stock_code"]].to_string(index=False))

# 저장
df_auto.to_csv("data/raw/dart/auto_corps_candidates.csv",
               index=False, encoding="utf-8-sig")
print(f"\n저장: data/raw/dart/auto_corps_candidates.csv")
