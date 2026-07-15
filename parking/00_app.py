from pathlib import Path
import textwrap
import zipfile

base = Path("/mnt/data/public_parking_streamlit")
base.mkdir(parents=True, exist_ok=True)

app_code = r'''
import html
import io
import re

import pandas as pd
import pydeck as pdk
import streamlit as st


# ---------------------------------------------------------
# 1. 기본 화면 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="공영주차장 요금 안내",
    page_icon="🅿️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #fffdf9;
        }

        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            color: #666666;
            margin-bottom: 1.5rem;
        }

        .info-box {
            padding: 1rem 1.2rem;
            border: 1px solid #ffd7ad;
            border-radius: 14px;
            background: #fff7ee;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #eeeeee;
            padding: 0.8rem;
            border-radius: 12px;
        }

        .parking-card {
            background: white;
            border: 1px solid #eeeeee;
            border-left: 5px solid #ff7a00;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }

        .parking-name {
            font-weight: 800;
            font-size: 1.05rem;
        }

        .parking-address {
            color: #555555;
            margin-top: 0.25rem;
        }

        .parking-fee {
            color: #e76900;
            font-weight: 700;
            margin-top: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🅿️ 공영주차장 정보 안내</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">CSV 파일을 업로드하면 주소별 주차요금을 검색하고 지도에서 확인할 수 있습니다.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. CSV 읽기 및 열 이름 자동 인식 함수
# ---------------------------------------------------------
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """UTF-8, CP949, EUC-KR 순서로 CSV 읽기를 시도합니다."""
    raw_data = uploaded_file.getvalue()
    last_error = None

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
        except pd.errors.ParserError as error:
            last_error = error

    raise ValueError(
        "CSV 파일을 읽지 못했습니다. 쉼표로 구분된 CSV인지 확인해 주세요."
    ) from last_error


def normalize_column_name(column_name: str) -> str:
    """열 이름 비교를 위해 공백, 특수문자, 대소문자 차이를 제거합니다."""
    return re.sub(r"[\s_\-()/]", "", str(column_name)).lower()


COLUMN_ALIASES = {
    "name": [
        "주차장명",
        "공영주차장명",
        "주차장 이름",
        "주차장",
        "name",
        "parking_name",
    ],
    "address": [
        "주소",
        "도로명주소",
        "소재지도로명주소",
        "지번주소",
        "소재지지번주소",
        "address",
    ],
    "fee": [
        "주차요금",
        "요금",
        "요금정보",
        "기본주차요금",
        "주차기본요금",
        "기본요금",
        "fee",
        "parking_fee",
    ],
    "latitude": [
        "위도",
        "lat",
        "latitude",
        "y",
        "위도좌표",
    ],
    "longitude": [
        "경도",
        "lon",
        "lng",
        "longitude",
        "x",
        "경도좌표",
    ],
}


def detect_column(columns, key):
    """별칭 목록을 이용해 적절한 열을 자동 탐색합니다."""
    normalized_columns = {
        normalize_column_name(column): column for column in columns
    }

    for alias in COLUMN_ALIASES[key]:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    return None


def select_column(label, columns, detected_column=None, optional=False):
    """자동 인식 결과를 기본값으로 보여 주는 열 선택 상자입니다."""
    options = list(columns)

    if optional:
        options = ["선택 안 함"] + options

    if detected_column in options:
        default_index = options.index(detected_column)
    else:
        default_index = 0

    selected = st.selectbox(label, options, index=default_index)

    if optional and selected == "선택 안 함":
        return None

    return selected


def format_fee(value) -> str:
    """숫자 요금에는 천 단위 쉼표와 '원'을 붙이고, 문자는 그대로 표시합니다."""
    if pd.isna(value):
        return "요금 정보 없음"

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return "요금 정보 없음"

    numeric_text = text.replace(",", "").replace("원", "").strip()

    if re.fullmatch(r"-?\d+(\.\d+)?", numeric_text):
        number = float(numeric_text)

        if number.is_integer():
            return f"{int(number):,}원"

        return f"{number:,.1f}원"

    return text


def clean_text(value, fallback="정보 없음") -> str:
    if pd.isna(value):
        return fallback

    text = str(value).strip()
    return text if text else fallback


# ---------------------------------------------------------
# 3. CSV 업로드
# ---------------------------------------------------------
with st.sidebar:
    st.header("📁 데이터 설정")

    uploaded_file = st.file_uploader(
        "공영주차장 CSV 파일 업로드",
        type=["csv"],
        help="주소와 주차요금 열이 필요합니다. 지도 표시를 위해 위도·경도 열도 함께 넣어 주세요.",
    )

    st.markdown(
        """
        **권장 CSV 열**

        - 주차장명
        - 주소
        - 주차요금
        - 위도
        - 경도
        """
    )

if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
            <b>사용 방법</b><br>
            1. 왼쪽에서 CSV 파일을 업로드합니다.<br>
            2. 주차장명·주소·요금·위도·경도 열을 확인합니다.<br>
            3. 주소나 주차장명을 검색하면 요금과 위치가 표시됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_data = pd.DataFrame(
        {
            "주차장명": ["샘플 공영주차장 A", "샘플 공영주차장 B"],
            "주소": ["서울특별시 중구 샘플로 1", "서울특별시 종로구 예시로 2"],
            "주차요금": ["30분 1,000원", "1시간 2,000원"],
            "위도": [37.5663, 37.5720],
            "경도": [126.9779, 126.9794],
        }
    )

    st.subheader("CSV 예시")
    st.dataframe(sample_data, use_container_width=True, hide_index=True)
    st.stop()


try:
    original_df = read_csv_safely(uploaded_file)
except Exception as error:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()

if original_df.empty:
    st.warning("업로드한 CSV에 데이터가 없습니다.")
    st.stop()

original_df.columns = [str(column).strip() for column in original_df.columns]


# ---------------------------------------------------------
# 4. 사용할 열 지정
# ---------------------------------------------------------
detected_name = detect_column(original_df.columns, "name")
detected_address = detect_column(original_df.columns, "address")
detected_fee = detect_column(original_df.columns, "fee")
detected_latitude = detect_column(original_df.columns, "latitude")
detected_longitude = detect_column(original_df.columns, "longitude")

with st.sidebar.expander("열 연결 설정", expanded=False):
    name_column = select_column(
        "주차장명 열",
        original_df.columns,
        detected_name,
        optional=True,
    )
    address_column = select_column(
        "주소 열",
        original_df.columns,
        detected_address,
    )
    fee_column = select_column(
        "주차요금 열",
        original_df.columns,
        detected_fee,
    )
    latitude_column = select_column(
        "위도 열",
        original_df.columns,
        detected_latitude,
        optional=True,
    )
    longitude_column = select_column(
        "경도 열",
        original_df.columns,
        detected_longitude,
        optional=True,
    )

if not address_column or not fee_column:
    st.error("주소 열과 주차요금 열을 선택해 주세요.")
    st.stop()


# ---------------------------------------------------------
# 5. 내부에서 사용할 표준 데이터 만들기
# ---------------------------------------------------------
parking_df = pd.DataFrame()

if name_column:
    parking_df["parking_name"] = original_df[name_column].apply(
        lambda value: clean_text(value, "이름 없는 주차장")
    )
else:
    parking_df["parking_name"] = "공영주차장"

parking_df["address"] = original_df[address_column].apply(clean_text)
parking_df["fee"] = original_df[fee_column].apply(format_fee)

if latitude_column and longitude_column:
    parking_df["latitude"] = pd.to_numeric(
        original_df[latitude_column], errors="coerce"
    )
    parking_df["longitude"] = pd.to_numeric(
        original_df[longitude_column], errors="coerce"
    )
else:
    parking_df["latitude"] = pd.NA
    parking_df["longitude"] = pd.NA

parking_df = parking_df[
    (parking_df["address"] != "정보 없음")
    | (parking_df["parking_name"] != "이름 없는 주차장")
].copy()

parking_df["region"] = (
    parking_df["address"]
    .str.split()
    .apply(lambda words: " ".join(words[:2]) if isinstance(words, list) else "기타")
)


# ---------------------------------------------------------
# 6. 검색 및 필터
# ---------------------------------------------------------
st.subheader("🔎 주소 또는 주차장 검색")

search_col, region_col = st.columns([2, 1])

with search_col:
    search_keyword = st.text_input(
        "주소나 주차장명을 입력하세요",
        placeholder="예: 강남구, 세종대로, 시청 주차장",
    ).strip()

with region_col:
    region_options = ["전체"] + sorted(
        region
        for region in parking_df["region"].dropna().unique()
        if region and region != "정보 없음"
    )
    selected_region = st.selectbox("지역 선택", region_options)

filtered_df = parking_df.copy()

if search_keyword:
    keyword_mask = (
        filtered_df["address"].str.contains(
            search_keyword, case=False, na=False, regex=False
        )
        | filtered_df["parking_name"].str.contains(
            search_keyword, case=False, na=False, regex=False
        )
    )
    filtered_df = filtered_df[keyword_mask]

if selected_region != "전체":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]


# ---------------------------------------------------------
# 7. 검색 결과 요약 및 요금 안내
# ---------------------------------------------------------
valid_map_df = filtered_df.dropna(
    subset=["latitude", "longitude"]
).copy()

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("전체 주차장", f"{len(parking_df):,}곳")
metric_col2.metric("검색 결과", f"{len(filtered_df):,}곳")
metric_col3.metric("지도 표시 가능", f"{len(valid_map_df):,}곳")

if filtered_df.empty:
    st.warning("검색 조건에 맞는 주차장이 없습니다.")
    st.stop()

st.subheader("💳 주차요금 안내")

# 결과가 너무 많으면 카드 대신 표를 먼저 보여 줍니다.
if len(filtered_df) <= 20:
    for _, row in filtered_df.iterrows():
        safe_name = html.escape(row["parking_name"])
        safe_address = html.escape(row["address"])
        safe_fee = html.escape(row["fee"])

        st.markdown(
            f"""
            <div class="parking-card">
                <div class="parking-name">{safe_name}</div>
                <div class="parking-address">📍 {safe_address}</div>
                <div class="parking-fee">💰 {safe_fee}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("검색 결과가 많아 표 형식으로 표시합니다.")

display_df = filtered_df[
    ["parking_name", "address", "fee", "latitude", "longitude"]
].rename(
    columns={
        "parking_name": "주차장명",
        "address": "주소",
        "fee": "주차요금",
        "latitude": "위도",
        "longitude": "경도",
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

csv_download = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "검색 결과 CSV 다운로드",
    data=csv_download,
    file_name="공영주차장_검색결과.csv",
    mime="text/csv",
)


# ---------------------------------------------------------
# 8. 지도 시각화
# ---------------------------------------------------------
st.subheader("🗺️ 공영주차장 지도")

if valid_map_df.empty:
    st.warning(
        "지도에 표시할 좌표가 없습니다. CSV에 위도와 경도 열을 추가하거나 "
        "왼쪽의 열 연결 설정에서 올바른 열을 선택해 주세요."
    )
    st.stop()

# 툴팁 안에서 CSV의 HTML이 실행되지 않도록 특수문자를 변환합니다.
valid_map_df["tooltip_name"] = valid_map_df["parking_name"].apply(html.escape)
valid_map_df["tooltip_address"] = valid_map_df["address"].apply(html.escape)
valid_map_df["tooltip_fee"] = valid_map_df["fee"].apply(html.escape)

center_latitude = valid_map_df["latitude"].mean()
center_longitude = valid_map_df["longitude"].mean()
zoom_level = 14 if len(valid_map_df) == 1 else 11

layer = pdk.Layer(
    "ScatterplotLayer",
    data=valid_map_df,
    get_position="[longitude, latitude]",
    get_radius=70,
    radius_min_pixels=6,
    radius_max_pixels=18,
    get_fill_color=[255, 122, 0, 190],
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
    stroked=True,
    filled=True,
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=center_latitude,
    longitude=center_longitude,
    zoom=zoom_level,
    pitch=0,
)

tooltip = {
    "html": """
        <div style="max-width: 310px;">
            <b style="font-size: 15px;">{tooltip_name}</b><br/>
            <span>📍 {tooltip_address}</span><br/>
            <span style="color: #ffb066;">💰 {tooltip_fee}</span>
        </div>
    """,
    "style": {
        "backgroundColor": "#222222",
        "color": "white",
        "fontSize": "13px",
        "padding": "10px",
        "borderRadius": "8px",
    },
}

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style=None,
)

st.pydeck_chart(deck, use_container_width=True, height=560)

st.caption(
    "지도 위의 주황색 점에 마우스를 올리면 주차장명, 주소, 주차요금을 확인할 수 있습니다."
)
'''.strip()

requirements = '''
streamlit>=1.45,<2
pandas>=2.2,<4
pydeck>=0.9,<1
'''.strip()

sample_csv = '''주차장명,주소,주차요금,위도,경도
샘플 공영주차장 A,서울특별시 중구 샘플로 1,"30분 1,000원",37.5663,126.9779
샘플 공영주차장 B,서울특별시 종로구 예시로 2,"1시간 2,000원",37.5720,126.9794
샘플 공영주차장 C,서울특별시 용산구 테스트로 3,"10분당 500원",37.5326,126.9900
'''.strip()
