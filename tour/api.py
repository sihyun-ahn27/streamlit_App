import requests
import pandas as pd
import streamlit as st

SERVICE_KEY = st.secrets["TOUR_API"]

AREA = {
    "전체":"",
    "서울":"1",
    "인천":"2",
    "대전":"3",
    "대구":"4",
    "광주":"5",
    "부산":"6",
    "울산":"7",
    "세종":"8",
    "경기":"31",
    "강원":"32",
    "충북":"33",
    "충남":"34",
    "경북":"35",
    "경남":"36",
    "전북":"37",
    "전남":"38",
    "제주":"39"
}

def get_festival(area, keyword):

    url="https://apis.data.go.kr/B551011/KorService1/searchFestival1"

    params={
        "serviceKey":SERVICE_KEY,
        "MobileOS":"ETC",
        "MobileApp":"Festival",
        "_type":"json",
        "numOfRows":100,
        "arrange":"A",
        "eventStartDate":"20260101"
    }

    if AREA[area]!="":
        params["areaCode"]=AREA[area]

    if keyword!="":
        params["keyword"]=keyword

    r=requests.get(url,params=params)

    items=r.json()["response"]["body"]["items"]["item"]

    rows=[]

    for i in items:

        rows.append({

            "title":i.get("title",""),

            "addr":i.get("addr1",""),

            "date":f'{i.get("eventstartdate","")} ~ {i.get("eventenddate","")}',

            "image":i.get("firstimage",""),

            "overview":i.get("overview",""),

            "homepage":f'https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid={i.get("contentid")}'

        })

    return pd.DataFrame(rows)
