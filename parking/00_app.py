from pathlib import Path
import zipfile

out_dir = Path("/mnt/data/parking_streamlit_fixed")
out_dir.mkdir(parents=True, exist_ok=True)

app_code = r'''
import html
import io
import re

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="공영주차장 정보 안내",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ 공영주차장 정보 안내")
st.caption("CSV 파일을 업로드하면 주소별 주차요금을 검색하고 지도에서 확인할 수 있습니다.")


def read_csv_safely(uploaded_file):
    raw_data = uploaded_file.getvalue()

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError("CSV 파일을 읽지 못했습니다. 파일 형식과 인코딩을 확인해 주세요.")


def normalize_column_name(name):
    return re.sub(r"[\s_\-()/]", "", str(name)).lower()


COLUMN_ALIASES = {
    "name": ["주차장명", "공영주차장명", "주차장이름", "주차장", "name", "parkingname"],
    "address": ["주소", "도로명주소", "소재지도로명주소", "지번주소", "소재지지번주소", "address"],
    "fee": ["주차요금", "요금", "요금정보", "기본주차요금", "주차기본요금", "기본요금", "fee"],
    "latitude": ["위도", "lat", "latitude", "y", "위도좌표"],
    "longitude": ["경도", "lon", "lng", "longitude", "x", "경도좌표"],
}


def detect_column(columns, key):
    normalized = {normalize_column_name(col): col for col in columns}

    for alias in COLUMN_ALIASES[key]:
        alias_normalized = normalize_column_name(alias)
        if alias_normalized in normalized:
            return normalized[alias_normalized]

    return None


def select_column(label, columns, detected=None, optional=False):
    options = list(columns)

    if optional:
        options = ["선택 안 함"] + options

    index = options.index(detected) if detected in options else 0
    selected = st.selectbox(label, options, index=index)

    if optional and selected == "선택 안 함":
        return None

    return selected


def clean_text(value, fallback="정보 없음"):
    if pd.isna(value):
        return fallback

    text = str(value).strip()
    return text if text else fallback


def format_fee(value):
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


with st.sidebar:
    st.header("📁 데이터 업로드")

    uploaded_file = st.file_uploader(
        "공영주차장 CSV 파일",
        type=["csv"],
        help="주소와 주차요금 열은 필수이며 지도 표시에는 위도·경도 열이 필요합니다.",
    )

    st.markdown(
        """
        **권장 열 이름**

        - 주차장명
        - 주소
        - 주차요금
        - 위도
        - 경도
        """
    )


if uploaded_file is None:
    st.info("왼쪽에서 CSV 파일을 업로드해 주세요.")

    sample_df = pd.DataFrame(
        {
            "주차장명": ["시청 공영주차장", "종로 공영주차장"],
            "주소": ["서울특별시 중구 세종대로 1", "서울특별시 종로구 종로 1"],
            "주차요금": ["30분 1,000원", "1시간 2,000원"],
            "위도": [37.5663, 37.5720],
            "경도": [126.9779, 126.9794],
        }
    )

    st.subheader("CSV 형식 예시")
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    st.stop()


try:
    original_df = read_csv_safely(uploaded_file)
except Exception as error:
    st.error(str(error))
    st.stop()


if original_df.empty:
    st.warning("업로드한 CSV에 데이터가 없습니다.")
    st.stop()


original_df.columns = [str(column).strip() for column in original_df.columns]

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
        original_df[latitude_column],
        errors="coerce",
    )
    parking_df["longitude"] = pd.to_numeric(
        original_df[longitude_column],
        errors="coerce",
    )
else:
    parking_df["latitude"] = pd.NA
    parking_df["longitude"] = pd.NA


st.subheader("🔎 주차장 검색")

search_keyword = st.text_input(
    "주소 또는 주차장명을 입력하세요",
    placeholder="예: 강남구, 세종대로, 시청 주차장",
).strip()

filtered_df = parking_df.copy()

if search_keyword:
    mask = (
        filtered_df["address"].str.contains(
            search_keyword,
            case=False,
            na=False,
            regex=False,
        )
        | filtered_df["parking_name"].str.contains(
            search_keyword,
            case=False,
            na=False,
            regex=False,
        )
    )
    filtered_df = filtered_df[mask]


summary_col1, summary_col2, summary_col3 = st.columns(3)

map_df = filtered_df.dropna(subset=["latitude", "longitude"]).copy()

summary_col1.metric("전체 주차장", f"{len(parking_df):,}곳")
summary_col2.metric("검색 결과", f"{len(filtered_df):,}곳")
summary_col3.metric("지도 표시 가능", f"{len(map_df):,}곳")


if filtered_df.empty:
    st.warning("검색 조건에 맞는 주차장이 없습니다.")
    st.stop()


st.subheader("💳 주소별 주차요금")

result_df = filtered_df[
    ["parking_name", "address", "fee"]
].rename(
    columns={
        "parking_name": "주차장명",
        "address": "주소",
        "fee": "주차요금",
    }
)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
)

download_data = result_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "검색 결과 CSV 다운로드",
    data=download_data,
    file_name="공영주차장_검색결과.csv",
    mime="text/csv",
)


st.subheader("🗺️ 공영주차장 지도")

if map_df.empty:
    st.warning(
        "지도에 표시할 위도·경도 데이터가 없습니다. "
        "CSV에 위도와 경도 열을 추가해 주세요."
    )
    st.stop()


map_df["tooltip_name"] = map_df["parking_name"].apply(html.escape)
map_df["tooltip_address"] = map_df["address"].apply(html.escape)
map_df["tooltip_fee"] = map_df["fee"].apply(html.escape)

center_latitude = map_df["latitude"].mean()
center_longitude = map_df["longitude"].mean()
zoom_level = 14 if len(map_df) == 1 else 11

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
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
        <b>{tooltip_name}</b><br/>
        📍 {tooltip_address}<br/>
        💰 {tooltip_fee}
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

st.pydeck_chart(
    deck,
    use_container_width=True,
    height=560,
)

st.caption(
    "지도 위 점에 마우스를 올리면 주차장명, 주소, 주차요금을 확인할 수 있습니다."
)
'''.strip()

requirements = """streamlit>=1.45,<2
pandas>=2.2,<4
pydeck>=0.9,<1
"""

sample_csv = """주차장명,주소,주차요금,위도,경도
시청 공영주차장,서울특별시 중구 세종대로 1,"30분 1,000원",37.5663,126.9779
종로 공영주차장,서울특별시 종로구 종로 1,"1시간 2,000원",37.5720,126.9794
용산 공영주차장,서울특별시 용산구 한강대로 1,"10분당 500원",37.5326,126.9900
"""

app_path = out_dir / "00_app.py"
req_path = out_dir / "requirements.txt"
sample_path = out_dir / "sample_parking.csv"

app_path.write_text(app_code, encoding="utf-8")
req_path.write_text(requirements, encoding="utf-8")
sample_path.write_text(sample_csv, encoding="utf-8-sig")

compile(app_code, str(app_path), "exec")

zip_path = Path("/mnt/data/parking_streamlit_fixed.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(app_path, "00_app.py")
    archive.write(req_path, "requirements.txt")
    archive.write(sample_path, "sample_parking.csv")

print("문법 검사 완료")
print(zip_path)
