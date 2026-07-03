import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("DART_API_KEY")

for year in ["2019", "2020", "2023"]:
    print(f"\n== 한온시스템 {year} ==")
    resp = requests.get(
        "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
        params={
            "crtfc_key": key,
            "corp_code": "00161125",
            "bsns_year": year,
            "reprt_code": "11011",
            "fs_div": "CFS"
        }
    ).json()
    for r in resp.get("list", []):
        nm = r["account_nm"]
        if any(k in nm for k in ["매출", "영업", "순이익", "당기"]):
            print(nm, "|", r["thstrm_amount"])
