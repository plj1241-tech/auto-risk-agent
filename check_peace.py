import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("DART_API_KEY")

for year in ["2019", "2020", "2021", "2022"]:
    print(f"\n== 평화산업 {year} ==")
    resp = requests.get(
        "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
        params={
            "crtfc_key": key,
            "corp_code": "00573579",
            "bsns_year": year,
            "reprt_code": "11011",
            "fs_div": "CFS"
        }
    ).json()

    if resp.get("status") != "000":
        print(f"  CFS 없음 ({resp.get('message','')}), OFS 시도...")
        resp = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": key,
                "corp_code": "00573579",
                "bsns_year": year,
                "reprt_code": "11011",
                "fs_div": "OFS"
            }
        ).json()

    for r in resp.get("list", []):
        nm = r["account_nm"]
        if any(k in nm for k in ["잉여", "결손", "자본"]):
            print(nm, "|", r["thstrm_amount"])
