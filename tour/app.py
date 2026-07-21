from pathlib import Path
import textwrap, zipfile, os, json

base = Path("/mnt/data/korea_festival_streamlit")
base.mkdir(parents=True, exist_ok=True)

app_py = r'''
from __future__ import annotations

import html
import random
import re
from datetime import date, datetime, timedelta
from urllib.parse import unquote

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="축제콕",
    page_icon="🎪",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = (
    "https://apis.data.go.kr/B551011/KorService2/searchFestival2"
)

AREA_CODES = {
    "전국": "",
    "서울": "1",
    "인천": "2",
    "대전": "3",
    "대구": "4",
    "광주": "5",
    "부산": "6",
    "울산": "7",
    "세종": "8",
    "경기": "31",
    "강원": "32",
    "충북": "33",
    "충남": "34",
    "경북": "35",
    "경남": "36",
    "전북": "37",
    "전남": "38",
    "제주": "39",
}

DEFAULT_IMAGE = (
    "https://images.unsplash.com/photo-1492684223066-81342ee5ff30"
    "?auto=format&fit=crop&w=1200&q=80"
)


# -----------------------------
# 스타일
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {
        border-radius: 24px;
        padding: 28px 30px;
        background: linear-gradient(135deg, #fff4e8 0%, #fff 48%, #eef7ff 100%);
        border: 1px solid rgba(0,0,0,.06);
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 6px 0; font-size: 2.35rem;}
    .hero p {margin: 0; color: #555; font-size: 1.02rem;}
    .festival-card {
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 18px;
        padding: 15px;
        background: white;
        box-shadow: 0 5px 16px rgba(0,0,0,.04);
        min-height: 185px;
    }
    .festival-title {
        font-weight: 750;
        font-size: 1.08rem;
        line-height: 1.35;
        margin-bottom: 8px;
    }
    .muted {color: #6b7280; font-size: .92rem;}
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 9px;
        margin-right: 5px;
        font-size: .78rem;
        background: #f3f4f6;
    }
    .status-now {background: #dcfce7; color: #166534;}
    .status-soon {background: #ffedd5; color: #9a3412;}
    .status-end {background: #f3f4f6; color: #4b5563;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 유틸리티
# -----------------------------
def get_api_key() -> str:
    """Streamlit Secrets에서 API 키를 읽는다."""
    try:
        key = str(st.secrets["TOUR_API_KEY"]).strip()
    except Exception:
        st.error(
            "API 키가 설정되지 않았습니다. Streamlit Cloud의 "
            "App settings → Secrets에 `TOUR_API_KEY = \"발급받은키\"`를 입력하세요."
        )
        st.stop()

    # data.go.kr에서 '인코딩된 인증키'를 복사한 경우도 처리
    return unquote(key)


def yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def parse_api_date(value: object) -> date | None:
    raw = re.sub(r"\D", "", str(value or ""))
    if len(raw) < 8:
        return None
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def safe_float(value: object) -> float | None:
    try:
        number = float(value)
        return number if number != 0 else None
    except (TypeError, ValueError):
        return None


def festival_status(start: date | None, end: date | None) -> tuple[str, str]:
    today = date.today()
    if not start or not end:
        return "일정 확인 필요", "end"
    if start <= today <= end:
        return "진행 중", "now"
    if today < start:
        days = (start - today).days
        return (f"D-{days}" if days > 0 else "오늘 시작"), "soon"
    return "종료", "end"


def date_label(start: date | None, end: date | None) -> str:
    if not start and not end:
        return "일정 정보 없음"
    if start and end:
        return f"{start:%Y.%m.%d} ~ {end:%Y.%m.%d}"
    value = start or end
    return f"{value:%Y.%m.%d}"


def normalize_items(payload: dict) -> list[dict]:
    """TourAPI의 item이 dict/list/빈 값인 경우를 모두 처리."""
    response = payload.get("response", {})
    header = response.get("header", {})

    result_code = str(header.get("resultCode", ""))
    if result_code and result_code not in {"0000", "0"}:
        message = header.get("resultMsg", "알 수 없는 API 오류")
        raise RuntimeError(f"TourAPI 오류: {result_code} / {message}")

    body = response.get("body", {})
    items_box = body.get("items") or {}
    items = items_box.get("item") if isinstance(items_box, dict) else []

    if not items:
        return []
    if isinstance(items, dict):
        return [items]
    return items


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_festivals(
    api_key: str,
    start_date: str,
    end_date: str,
    area_code: str,
    page_no: int = 1,
    num_rows: int = 100,
) -> tuple[list[dict], int]:
    params = {
        "serviceKey": api_key,
        "MobileOS": "ETC",
        "MobileApp": "FestivalKkok",
        "_type": "json",
        "eventStartDate": start_date,
        "eventEndDate": end_date,
        "arrange": "A",
        "numOfRows": num_rows,
        "pageNo": page_no,
    }
    if area_code:
        params["areaCode"] = area_code

    response = requests.get(API_BASE_URL, params=params, timeout=20)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        preview = response.text[:300]
        raise RuntimeError(f"API가 JSON이 아닌 응답을 보냈습니다: {preview}") from exc

    items = normalize_items(payload)
    total_count = int(
        payload.get("response", {}).get("body", {}).get("totalCount", len(items)) or 0
    )
    return items, total_count


def build_dataframe(items: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for item in items:
        start = parse_api_date(item.get("eventstartdate"))
        end = parse_api_date(item.get("eventenddate"))
        status, status_type = festival_status(start, end)

        rows.append(
            {
                "contentid": str(item.get("contentid", "")),
                "축제명": clean_text(item.get("title")) or "이름 없는 축제",
                "주소": clean_text(item.get("addr1")),
                "상세주소": clean_text(item.get("addr2")),
                "시작일": start,
                "종료일": end,
                "기간": date_label(start, end),
                "상태": status,
                "status_type": status_type,
                "전화": clean_text(item.get("tel")),
                "대표이미지": item.get("firstimage") or item.get("firstimage2") or DEFAULT_IMAGE,
                "위도": safe_float(item.get("mapy")),
                "경도": safe_float(item.get("mapx")),
                "수정일": clean_text(item.get("modifiedtime")),
            }
        )

    return pd.DataFrame(rows)


def make_map(df: pd.DataFrame) -> pdk.Deck:
    map_df = df.dropna(subset=["위도", "경도"]).copy()

    if map_df.empty:
        view_state = pdk.ViewState(latitude=36.35, longitude=127.8, zoom=6)
    else:
        view_state = pdk.ViewState(
            latitude=float(map_df["위도"].mean()),
            longitude=float(map_df["경도"].mean()),
            zoom=6.3 if len(map_df) > 1 else 11,
        )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[경도, 위도]",
        get_radius=650,
        radius_min_pixels=5,
        radius_max_pixels=14,
        pickable=True,
        auto_highlight=True,
        get_fill_color=[255, 112, 67, 190],
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": "<b>{축제명}</b><br>{기간}<br>{주소}",
        "style": {"backgroundColor": "#1f2937", "color": "white"},
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,
    )


def favorite_ids() -> set[str]:
    if "favorite_ids" not in st.session_state:
        st.session_state.favorite_ids = set()
    return st.session_state.favorite_ids


def render_card(row: pd.Series, key_prefix: str) -> None:
    favorite = row["contentid"] in favorite_ids()
    badge_class = {
        "now": "status-now",
        "soon": "status-soon",
        "end": "status-end",
    }.get(row["status_type"], "status-end")

    st.image(row["대표이미지"], use_container_width=True)
    st.markdown(
        f"""
        <div class="festival-card">
            <div class="festival-title">{html.escape(row["축제명"])}</div>
            <span class="badge {badge_class}">{html.escape(row["상태"])}</span>
            <span class="badge">{html.escape(row["기간"])}</span>
            <p class="muted">📍 {html.escape(row["주소"] or "주소 정보 없음")}</p>
            <p class="muted">☎️ {html.escape(row["전화"] or "문의처 정보 없음")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    label = "💔 찜 해제" if favorite else "⭐ 찜하기"
    if st.button(label, key=f"{key_prefix}_{row['contentid']}", use_container_width=True):
        if favorite:
            st.session_state.favorite_ids.remove(row["contentid"])
        else:
            st.session_state.favorite_ids.add(row["contentid"])
        st.rerun()


# -----------------------------
# 화면
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎪 축제콕</h1>
      <p>한국관광공사 TourAPI로 찾는 전국 축제 지도 · 랜덤 추천 · 나만의 축제 일정표</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = get_api_key()

today = date.today()
default_end = today + timedelta(days=90)

with st.sidebar:
    st.header("🔎 축제 찾기")
    region_name = st.selectbox("지역", AREA_CODES.keys())
    search_range = st.date_input(
        "축제 기간",
        value=(today, default_end),
        min_value=today - timedelta(days=365),
        max_value=today + timedelta(days=730),
    )
    keyword = st.text_input("축제명·주소 검색", placeholder="예: 불꽃, 서울, 벚꽃")
    only_ongoing = st.checkbox("현재 진행 중인 축제만")
    sort_option = st.selectbox(
        "정렬",
        ["가까운 시작일순", "축제명순", "종료 임박순"],
    )
    search_clicked = st.button("축제 검색", type="primary", use_container_width=True)

    st.divider()
    st.caption("API 키는 GitHub에 직접 올리지 말고 Streamlit Secrets에 저장하세요.")

if isinstance(search_range, (tuple, list)) and len(search_range) == 2:
    selected_start, selected_end = search_range
else:
    selected_start = selected_end = search_range

if selected_start > selected_end:
    st.warning("시작일이 종료일보다 늦습니다.")
    st.stop()

with st.spinner("전국의 축제 정보를 불러오는 중입니다..."):
    try:
        items, total_count = fetch_festivals(
            api_key=api_key,
            start_date=yyyymmdd(selected_start),
            end_date=yyyymmdd(selected_end),
            area_code=AREA_CODES[region_name],
        )
    except requests.Timeout:
        st.error("TourAPI 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요.")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"TourAPI 연결에 실패했습니다: {exc}")
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

df = build_dataframe(items)

if not df.empty and keyword.strip():
    query = keyword.strip()
    mask = (
        df["축제명"].str.contains(query, case=False, na=False)
        | df["주소"].str.contains(query, case=False, na=False)
    )
    df = df[mask]

if not df.empty and only_ongoing:
    df = df[df["status_type"] == "now"]

if not df.empty:
    if sort_option == "축제명순":
        df = df.sort_values("축제명")
    elif sort_option == "종료 임박순":
        df = df.sort_values(["종료일", "시작일"], na_position="last")
    else:
        df = df.sort_values(["시작일", "축제명"], na_position="last")

df = df.reset_index(drop=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("검색 결과", f"{len(df)}개")
m2.metric("API 전체 결과", f"{total_count}개")
m3.metric("진행 중", f"{(df['status_type'] == 'now').sum() if not df.empty else 0}개")
m4.metric("찜한 축제", f"{len(favorite_ids())}개")

if df.empty:
    st.info("조건에 맞는 축제가 없습니다. 지역이나 날짜 범위를 넓혀 보세요.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["✨ 추천", "🗺️ 지도", "📋 전체 목록", "⭐ 나의 축제"]
)

with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("오늘의 축제 룰렛")
        if "roulette_id" not in st.session_state or search_clicked:
            st.session_state.roulette_id = random.choice(df["contentid"].tolist())

        if st.button("🎲 다른 축제 뽑기", use_container_width=True):
            st.session_state.roulette_id = random.choice(df["contentid"].tolist())

        roulette_rows = df[df["contentid"] == st.session_state.roulette_id]
        if roulette_rows.empty:
            st.session_state.roulette_id = random.choice(df["contentid"].tolist())
            roulette_rows = df[df["contentid"] == st.session_state.roulette_id]

        render_card(roulette_rows.iloc[0], "roulette")

    with right:
        st.subheader("곧 시작하는 축제")
        upcoming = df[df["시작일"].notna() & (df["시작일"] >= today)].head(5)
        if upcoming.empty:
            st.caption("선택한 조건에서 예정된 축제가 없습니다.")
        else:
            for _, row in upcoming.iterrows():
                st.markdown(
                    f"**{row['축제명']}**  \n"
                    f"{row['상태']} · {row['기간']}  \n"
                    f"📍 {row['주소'] or '주소 정보 없음'}"
                )
                st.divider()

with tab2:
    mapped = df.dropna(subset=["위도", "경도"])
    if mapped.empty:
        st.info("검색 결과에 지도 좌표가 없습니다.")
    else:
        st.pydeck_chart(make_map(mapped), use_container_width=True)
        st.caption(f"지도에 좌표가 있는 축제 {len(mapped)}개를 표시했습니다.")

with tab3:
    display_columns = ["축제명", "상태", "기간", "주소", "전화"]
    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "축제명": st.column_config.TextColumn(width="large"),
            "주소": st.column_config.TextColumn(width="large"),
        },
    )

    st.subheader("축제 카드")
    max_cards = min(len(df), 24)
    for start_idx in range(0, max_cards, 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, df.iloc[start_idx:start_idx + 3].iterrows()):
            with col:
                render_card(row, f"list_{start_idx}")

    if len(df) > max_cards:
        st.caption(
            f"화면 속도를 위해 카드에는 앞의 {max_cards}개만 표시했습니다. "
            "전체 결과는 위 표에서 확인할 수 있습니다."
        )

with tab4:
    favorite_df = df[df["contentid"].isin(favorite_ids())].copy()

    if favorite_df.empty:
        st.info("찜한 축제가 없습니다. 축제 카드에서 ⭐ 찜하기를 눌러 보세요.")
    else:
        st.subheader("나만의 축제 일정표")
        st.dataframe(
            favorite_df[["축제명", "기간", "주소", "전화"]],
            use_container_width=True,
            hide_index=True,
        )

        export_df = favorite_df[
            ["축제명", "시작일", "종료일", "주소", "전화"]
        ].copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "📥 나의 축제 일정표 CSV 다운로드",
            data=csv_data,
            file_name=f"나의_축제_일정표_{today:%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "축제 정보는 한국관광공사 TourAPI 제공 자료를 바탕으로 하며, "
    "일정과 운영 내용은 주최 측 사정에 따라 바뀔 수 있습니다."
)
'''

requirements = '''
streamlit>=1.42,<2.0
requests>=2.32,<3.0
pandas>=2.2,<3.0
pydeck>=0.9,<1.0
'''

readme = r'''
# 축제콕 🎪

한국관광공사 TourAPI를 이용해 전국 축제를 검색하고 지도에서 확인하는 Streamlit 앱입니다.

## 주요 기능

- 지역·날짜·키워드 검색
- 진행 중 / D-day 자동 표시
- 전국 축제 지도
- 축제 룰렛 랜덤 추천
- 찜 목록과 일정표 CSV 다운로드

## 1. 파일 구성

```text
.
├── app.py
├── requirements.txt
└── .gitignore
