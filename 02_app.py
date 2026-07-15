from __future__ import annotations

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Snack Match | 매점 가격 비교",
    page_icon="🍪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ORANGE = "#F47A1F"
DARK_ORANGE = "#D85E0C"
PALE_ORANGE = "#FFF4E8"
TEXT = "#202124"
MUTED = "#6B6F76"
BORDER = "#ECE6DF"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "products.csv"

FALLBACK_CSV = """상품명,가격,열량
초코송이,1150,190
칸쵸,2000,190
CU 구운감자쌀과자,3000,190
감자칩,1800,210
초코칩쿠키,1500,180
버터와플,2000,230
에너지바,1800,195
크래커,1300,175
웨하스,1600,200
젤리,1200,170
"""


def apply_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: Inter, Pretendard, "Noto Sans KR", Arial, sans-serif;
            color: {TEXT};
        }}
        .stApp {{ background: #FFFFFF; }}
        .block-container {{
            max-width: 900px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }}
        .hero {{
            padding: 2rem 2.1rem;
            border: 1px solid {BORDER};
            border-radius: 24px;
            background: linear-gradient(135deg, #FFFFFF 0%, {PALE_ORANGE} 100%);
            box-shadow: 0 12px 32px rgba(82, 50, 20, 0.06);
            margin-bottom: 1.25rem;
        }}
        .brand {{
            font-size: .82rem;
            color: {DARK_ORANGE};
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }}
        .hero-title {{
            font-size: clamp(2rem, 5vw, 3.25rem);
            line-height: 1.08;
            letter-spacing: -.055em;
            font-weight: 900;
            margin: 0 0 .7rem 0;
        }}
        .hero-title span {{ color: {ORANGE}; }}
        .hero-desc {{
            color: {MUTED};
            line-height: 1.72;
            margin: 0;
        }}
        .result-box {{
            padding: 1.15rem 1.25rem;
            border-radius: 16px;
            background: {PALE_ORANGE};
            border: 1px solid #FFD8B6;
            margin: .7rem 0 1rem 0;
            line-height: 1.65;
        }}
        div[data-testid="stForm"] {{
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1.1rem 1.15rem 1.2rem 1.15rem;
            background: #FFFFFF;
            box-shadow: 0 8px 24px rgba(70, 40, 10, 0.045);
        }}
        div[data-testid="stFormSubmitButton"] > button {{
            width: 100%;
            min-height: 48px;
            border-radius: 13px;
            border: 1px solid {ORANGE};
            background: {ORANGE};
            color: #FFFFFF;
            font-weight: 800;
        }}
        div[data-testid="stFormSubmitButton"] > button:hover {{
            background: {DARK_ORANGE};
            border-color: {DARK_ORANGE};
            color: #FFFFFF;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 16px;
            overflow: hidden;
        }}
        .tiny-note {{
            color: {MUTED};
            font-size: .84rem;
            line-height: 1.65;
            margin-top: .75rem;
        }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_products() -> pd.DataFrame:
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    else:
        data = pd.read_csv(io.StringIO(FALLBACK_CSV))

    rename_map = {
        "총열량": "열량",
        "price": "가격",
        "calories": "열량",
        "product": "상품명",
    }
    data = data.rename(columns=rename_map)
    required = ["상품명", "가격", "열량"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("products.csv에 필요한 열이 없습니다: " + ", ".join(missing))

    data = data[required].copy()
    data["상품명"] = data["상품명"].astype(str).str.strip()
    for column in ["가격", "열량"]:
        data[column] = (
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        )
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required)
    data = data[(data["가격"] > 0) & (data["열량"] > 0)]
    data = data.drop_duplicates(subset=["상품명"], keep="last").reset_index(drop=True)
    return data


def standardized_knn(
    data: pd.DataFrame,
    product_name: str,
    price: float,
    calories: float,
    k: int = 8,
) -> tuple[pd.DataFrame, int]:
    """비슷한 열량 상품을 먼저 거르고, 표준화된 가격·열량 거리로 KNN 순위를 계산한다."""
    candidates = data.copy()
    candidates = candidates[
        candidates["상품명"].str.casefold() != product_name.strip().casefold()
    ].copy()

    if candidates.empty:
        return candidates, 15

    # 기본 범위는 ±15kcal. 결과가 너무 적으면 화면 조작 없이 자동으로 조금 넓힌다.
    used_range = 15
    filtered = candidates[(candidates["열량"] - calories).abs() <= used_range].copy()
    for next_range in [25, 40]:
        if len(filtered) >= min(5, len(candidates)):
            break
        used_range = next_range
        filtered = candidates[(candidates["열량"] - calories).abs() <= used_range].copy()

    if filtered.empty:
        filtered = candidates.nsmallest(min(k, len(candidates)), columns="열량").copy()

    # 입력 상품을 포함해 평균과 표준편차를 계산한다.
    matrix = pd.concat(
        [
            filtered[["가격", "열량"]],
            pd.DataFrame([{"가격": price, "열량": calories}]),
        ],
        ignore_index=True,
    ).astype(float)

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0, 1.0)
    z = (matrix - means) / stds

    query = z.iloc[-1].to_numpy(dtype=float)
    candidate_z = z.iloc[:-1].to_numpy(dtype=float)

    # 열량 유사성을 더 중요하게 반영한다.
    weights = np.array([1.0, 4.0], dtype=float)  # 가격, 열량
    distances = np.sqrt(np.sum(weights * (candidate_z - query) ** 2, axis=1))

    filtered["KNN거리"] = distances
    filtered["열량차이"] = (filtered["열량"] - calories).abs()
    filtered["가격차이"] = filtered["가격"] - price
    filtered["원/100kcal"] = filtered["가격"] / filtered["열량"] * 100

    result = (
        filtered
        .sort_values(["KNN거리", "열량차이", "가격"], ascending=[True, True, True])
        .head(k)
        .reset_index(drop=True)
    )
    return result, used_range


def main() -> None:
    apply_css()

    st.markdown(
        """
        <div class="hero">
            <div class="brand">Snack Match</div>
            <div class="hero-title">비슷한 열량의 상품을<br><span>가격으로 바로 비교</span>하세요.</div>
            <p class="hero-desc">
                상품명, 가격, 열량만 입력하면 학교 매점 데이터에서
                열량이 비슷한 상품을 찾아 한눈에 보여 줍니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        products = load_products()
    except Exception as error:
        st.error(f"상품 데이터를 읽지 못했습니다: {error}")
        st.stop()

    with st.form("simple_search", clear_on_submit=False):
        product_name = st.text_input(
            "상품명",
            placeholder="예: 초코송이",
        )
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(
                "가격(원)",
                min_value=1,
                value=1150,
                step=50,
            )
        with c2:
            calories = st.number_input(
                "열량(kcal)",
                min_value=1,
                value=190,
                step=5,
            )
        submitted = st.form_submit_button("비슷한 상품 찾아보기")

    if submitted:
        if not product_name.strip():
            st.warning("상품명을 입력해 주세요.")
            st.stop()

        result, used_range = standardized_knn(
            products,
            product_name=product_name,
            price=float(price),
            calories=float(calories),
            k=8,
        )

        if result.empty:
            st.info("비교할 상품이 없습니다. products.csv에 상품을 추가해 주세요.")
            st.stop()

        st.markdown(
            f"""
            <div class="result-box">
                <b>{product_name}</b>({int(calories):,}kcal, {int(price):,}원)과
                열량이 가까운 상품 <b>{len(result)}개</b>를 찾았습니다.<br>
                적용된 열량 범위: 기준 상품에서 ±{used_range}kcal
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 학생에게는 가장 필요한 세 열만 보여 준다.
        simple_table = result[["상품명", "열량", "가격"]].copy()
        simple_table["열량"] = simple_table["열량"].round(0).astype(int)
        simple_table["가격"] = simple_table["가격"].round(0).astype(int)

        st.dataframe(
            simple_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "상품명": st.column_config.TextColumn("상품명"),
                "열량": st.column_config.NumberColumn("열량", format="%d kcal"),
                "가격": st.column_config.NumberColumn("가격", format="%d원"),
            },
        )

        cheapest = result.loc[result["가격"].idxmin()]
        st.caption(
            f"표에 나온 상품 중 가격이 가장 낮은 상품은 "
            f"'{cheapest['상품명']}'({int(cheapest['가격']):,}원)입니다. "
            "가격 정보만 비교한 결과이며, 건강성이나 만족도를 판단한 것은 아닙니다."
        )

        with st.expander("분석 원리 보기 · 수업용"):
            st.markdown(
                """
                **1. 열량 범위 검색**  
                먼저 입력한 열량과 가까운 상품만 남깁니다. 기본 기준은 ±15kcal이며,
                후보가 적으면 ±25kcal, ±40kcal까지 자동으로 넓힙니다.

                **2. 표준화**  
                가격과 열량은 숫자의 단위와 크기가 다르므로 각각
                `z = (x - 평균) / 표준편차`로 변환합니다.

                **3. K-최근접 이웃(KNN)**  
                표준화된 가격과 열량 사이의 거리를 계산하여 가까운 상품을 찾습니다.
                열량 비교가 중심이므로 열량 차이에 더 큰 가중치를 줍니다.

                **4. 원/100kcal**  
                이 지표는 수업용 세부 분석에만 사용합니다. 낮다고 해서 반드시 더 건강하거나
                더 좋은 상품이라는 뜻은 아니므로 학생용 기본 표에는 노출하지 않았습니다.
                """
            )

            detail = result[[
                "상품명", "열량", "가격", "열량차이", "가격차이", "원/100kcal"
            ]].copy()
            detail["원/100kcal"] = detail["원/100kcal"].round(1)
            st.dataframe(detail, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="tiny-note">
            현재 상품 데이터는 기능 확인용 예시입니다. 실제 사용 전 학교 매점에서
            상품명·판매가격·제품 전체 열량을 조사해 <b>products.csv</b>를 교체하세요.<br>
            이 앱은 가격 비교 정보를 제공할 뿐 특정 식품의 구매나 섭취를 권하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
