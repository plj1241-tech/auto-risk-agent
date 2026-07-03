import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("DART_API_KEY")

corps = {
    "현대모비스": "00164788",
    "평화산업":   "00573579",
    "한온시스템": "00161125",
    "서연이화":   "01036446",
}

for name, code in corps.items():
    print(f"\n== {name} ==")
    resp = requests.get(
        "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
        params={
            "crtfc_key": key,
            "corp_code": code,
            "bsns_year": "2023",
            "reprt_code": "11011",
            "fs_div": "CFS"
        }
    ).json()
    for r in resp.get("list", []):
        nm = r["account_nm"]
        if any(k in nm for k in ["자본", "잉여", "결손", "이익"]):
            print(nm, "|", r["thstrm_amount"])
