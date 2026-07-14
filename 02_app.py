from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (
    DERIVED_COLUMNS,
    REQUIRED_COLUMNS,
    coefficient_table,
    display_columns,
    mode_feature_weights,
    pareto_front_mask,
    pareto_objectives,
    prepare_data,
    run_hedonic_regression,
    similar_products,
    weighted_knn_recommend,
)


# =========================================================
# 1. 기본 설정과 디자인
# =========================================================
st.set_page_config(
    page_title="SnackLens | 매점 상품 비교",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#F47A1F"
DARK_ORANGE = "#D85F0D"
PALE_ORANGE = "#FFF3E8"
CREAM = "#FFF9F4"
TEXT = "#1F2328"
MUTED = "#6E737B"
BORDER = "#EEE7E0"
GREEN = "#2E8B57"
RED = "#D94B3D"
GRAY = "#B8B2AC"

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample_data.csv"


def apply_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --orange: {ORANGE};
            --dark-orange: {DARK_ORANGE};
            --pale-orange: {PALE_ORANGE};
            --cream: {CREAM};
            --text: {TEXT};
            --muted: {MUTED};
            --border: {BORDER};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, Pretendard, "Noto Sans KR", Arial, sans-serif;
            color: var(--text);
        }}

        .stApp {{ background: #FFFFFF; }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFF8F1 0%, #FFFFFF 100%);
            border-right: 1px solid var(--border);
        }}

        .block-container {{
            padding-top: 1.45rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        h1, h2, h3 {{ letter-spacing: -0.035em; }}

        .hero {{
            padding: 2.25rem 2.45rem;
            border: 1px solid var(--border);
            border-radius: 26px;
            background:
                radial-gradient(circle at 90% 12%, rgba(244,122,31,.16), transparent 25%),
                linear-gradient(135deg, #FFFFFF 0%, #FFF8F1 100%);
            box-shadow: 0 14px 40px rgba(78, 48, 20, .06);
            margin-bottom: 1.15rem;
        }}

        .eyebrow {{
            display: inline-flex;
            color: var(--dark-orange);
            background: var(--pale-orange);
            border: 1px solid #FFD7B6;
            padding: .38rem .72rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: .82rem;
            margin-bottom: .9rem;
        }}

        .hero-title {{
            font-size: clamp(2.15rem, 4vw, 4rem);
            line-height: 1.03;
            font-weight: 900;
            letter-spacing: -0.058em;
            margin: 0 0 .9rem 0;
            color: #171717;
        }}

        .hero-title span {{ color: var(--orange); }}

        .hero-subtitle {{
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.75;
            max-width: 920px;
            margin: 0;
        }}

        .section-kicker {{
            color: var(--orange);
            font-weight: 850;
            font-size: .78rem;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }}

        .section-title {{
            font-size: 1.62rem;
            font-weight: 880;
            margin: 0 0 .35rem 0;
            color: #191919;
        }}

        .section-desc {{
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: .9rem;
        }}

        .metric-card {{
            min-height: 122px;
            padding: 1.12rem 1.18rem;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: #FFFFFF;
            box-shadow: 0 8px 26px rgba(65, 39, 15, .045);
        }}

        .metric-label {{
            color: var(--muted);
            font-size: .85rem;
            font-weight: 680;
            margin-bottom: .48rem;
        }}

        .metric-value {{
            font-size: 1.62rem;
            font-weight: 900;
            color: #171717;
            letter-spacing: -0.04em;
        }}

        .metric-note {{
            margin-top: .35rem;
            color: var(--muted);
            font-size: .8rem;
            line-height: 1.45;
        }}

        .soft-card, .orange-card {{
            padding: 1.18rem 1.25rem;
            border-radius: 18px;
            height: 100%;
        }}

        .soft-card {{ border: 1px solid var(--border); background: #FFFFFF; }}
        .orange-card {{ border: 1px solid #FFD6B4; background: var(--pale-orange); }}

        .insight {{
            border-left: 4px solid var(--orange);
            background: #FFF9F4;
            padding: 1rem 1.15rem;
            border-radius: 0 14px 14px 0;
            line-height: 1.72;
            color: #343434;
        }}

        .tag {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .38rem .72rem;
            font-weight: 800;
            font-size: .82rem;
            color: var(--dark-orange);
            background: var(--pale-orange);
            border: 1px solid #FFD8B8;
        }}

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
            border-radius: 12px;
            border: 1px solid var(--orange);
            background: var(--orange);
            color: white;
            font-weight: 780;
            min-height: 42px;
        }}

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            border-color: var(--dark-orange);
            background: var(--dark-orange);
            color: white;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: .45rem; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 12px 12px 0 0; padding: .55rem .9rem; }}
        .stTabs [aria-selected="true"] {{ color: var(--orange); }}

        hr {{ border-color: var(--border); }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div style="margin:1.15rem 0 .75rem 0;">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=58, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, Noto Sans KR, sans-serif", color=TEXT),
        title_font=dict(size=17, color=TEXT),
        hoverlabel=dict(bgcolor="#FFFFFF"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#F1ECE7", zerolinecolor="#E8DFD7")
    fig.update_yaxes(gridcolor="#F1ECE7", zerolinecolor="#E8DFD7")
    return fig


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rounded = [
        "가격",
        "총열량",
        "총내용량",
        "당류",
        "나트륨",
        "단백질",
        "원/100kcal",
        "kcal/1000원",
        "원/100단위",
        "100단위당열량",
        "당류/100kcal",
        "나트륨/100kcal",
        "단백질/100kcal",
        "열량차이",
        "가격차이",
        "열량차이율(%)",
        "가격차이율(%)",
        "KNN거리",
        "유사도점수",
    ]
    for c in rounded:
        if c in result.columns:
            digits = 0 if c in {"가격", "총열량", "나트륨", "가격차이", "열량차이"} else 2
            result[c] = result[c].round(digits)
    return result


def reference_radar(reference: pd.Series, top: pd.Series, features: list[str], source_df: pd.DataFrame) -> go.Figure:
    labels = {
        "가격": "가격",
        "총열량": "열량",
        "총내용량": "내용량",
        "당류": "당류",
        "나트륨": "나트륨",
        "단백질": "단백질",
        "원/100kcal": "원/100kcal",
        "원/100단위": "원/100단위",
    }
    usable = [f for f in features if f in labels][:6]
    if len(usable) < 3:
        usable = ["가격", "총열량", "총내용량"]

    mins = source_df[usable].min()
    maxs = source_df[usable].max()
    denom = (maxs - mins).replace(0, 1)
    ref_values = ((reference[usable] - mins) / denom * 100).tolist()
    top_values = ((top[usable] - mins) / denom * 100).tolist()
    theta = [labels[f] for f in usable]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=ref_values + [ref_values[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name=str(reference["상품명"]),
            line=dict(color=ORANGE, width=3),
            fillcolor="rgba(244,122,31,.17)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=top_values + [top_values[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name=str(top["상품명"]),
            line=dict(color="#60656B", width=2),
            fillcolor="rgba(96,101,107,.08)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER)),
        title="선택 상품과 1순위 유사 상품",
    )
    return style_fig(fig, 470)


# =========================================================
# 2. 데이터 불러오기와 세션 상태
# =========================================================
apply_css()

if "added_products" not in st.session_state:
    st.session_state.added_products = []

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:.55rem .1rem .95rem .1rem;">
            <div style="font-size:1.55rem;font-weight:920;letter-spacing:-.05em;">
                Snack<span style="color:{ORANGE};">Lens</span>
            </div>
            <div style="color:{MUTED};font-size:.82rem;margin-top:.25rem;">
                school store decision support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "메뉴",
        ["홈", "조건 비교", "KNN 추천", "파레토 분석", "심화 분석", "데이터 관리"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### 데이터 선택")
    data_mode = st.radio("데이터", ["샘플 데이터", "CSV 업로드"], label_visibility="collapsed")
    uploaded = None
    if data_mode == "CSV 업로드":
        uploaded = st.file_uploader("매점 상품 CSV", type=["csv"])

    template = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig").head(5)
    st.download_button(
        "CSV 템플릿 받기",
        data=csv_bytes(template),
        file_name="snacklens_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("가격·열량·내용량은 제품 전체 기준으로 입력하세요. 1회 제공량과 혼동하지 않도록 주의합니다.")

if uploaded is not None:
    try:
        raw_df = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:
        st.error(f"CSV를 읽지 못했습니다: {exc}")
        st.stop()
else:
    raw_df = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    if data_mode == "CSV 업로드":
        st.info("CSV를 업로드하기 전까지 샘플 데이터로 기능을 보여줍니다.")

if st.session_state.added_products:
    raw_df = pd.concat([raw_df, pd.DataFrame(st.session_state.added_products)], ignore_index=True)

df, cleaning_summary, missing_required = prepare_data(raw_df)
if missing_required:
    st.error("필수 열이 없습니다: " + ", ".join(missing_required))
    st.stop()

if len(df) < 10:
    st.warning("비교와 추천의 안정성을 위해 상품 30개 이상을 권장합니다.")

product_names = sorted(df["상품명"].unique().tolist())
categories = sorted(df["카테고리"].unique().tolist())


# =========================================================
# 3. 페이지
# =========================================================
if page == "홈":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">분산된 매점 정보를 한눈에</div>
            <div class="hero-title">가격표를 넘어,<br><span>목적에 맞는 비교</span>를 합니다.</div>
            <p class="hero-subtitle">
                SnackLens는 가격·열량·내용량·영양성분을 같은 표 안에서 비교하고,
                표준화된 K-최근접 이웃 알고리즘으로 유사 상품을 찾습니다.
                가장 많은 열량이나 가장 싼 상품을 정답으로 강요하지 않고,
                학생이 자신의 소비 목적을 고른 뒤 필요한 정보를 투명하게 확인하도록 돕습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("등록 상품", f"{len(df):,}개", "분석 가능한 상품 수")
    with c2:
        metric_card("카테고리", f"{df['카테고리'].nunique()}개", "같은 유형 비교 가능")
    with c3:
        metric_card("평균 가격", f"{df['가격'].mean():,.0f}원", "샘플 또는 업로드 자료 기준")
    with c4:
        metric_card("원/100kcal 범위", f"{df['원/100kcal'].min():,.0f}~{df['원/100kcal'].max():,.0f}", "단위가격 격차")

    section_header(
        "Core Design",
        "하나의 정답 대신 세 가지 소비 목적",
        "합리성은 소비자가 무엇을 중요하게 생각하는지에 따라 달라집니다.",
    )
    cols = st.columns(3)
    cards = [
        ("가성비", "에너지·양 대비 가격을 비교하되, 영양적 우수성과 동일시하지 않습니다."),
        ("저열량", "비슷한 가격대에서 총열량과 당류가 낮은 상품을 탐색합니다."),
        ("영양균형", "가격과 함께 당류·나트륨·단백질의 균형을 비교합니다."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="tag">{title}</div>
                    <div style="margin-top:.75rem;color:{MUTED};line-height:1.68;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header(
        "Workflow",
        "데이터과학의 전 과정을 앱으로 연결",
    )
    flow = st.columns(4)
    steps = [
        ("01", "수집", "영양성분표와 가격표를 직접 조사"),
        ("02", "정제", "총제품 기준과 단위를 통일"),
        ("03", "모델링", "표준화 후 가중 KNN 거리 계산"),
        ("04", "의사결정", "조건 비교·파레토·회귀 결과 제공"),
    ]
    for col, (num, title, desc) in zip(flow, steps):
        with col:
            st.markdown(
                f"""
                <div class="soft-card">
                    <div style="font-size:.8rem;color:{ORANGE};font-weight:900;">{num}</div>
                    <div style="font-size:1.08rem;font-weight:880;margin:.32rem 0;">{title}</div>
                    <div style="color:{MUTED};font-size:.9rem;line-height:1.55;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="insight" style="margin-top:1rem;">
            <b>해석 원칙:</b> 원/100kcal이 낮다는 것은 에너지 단위당 가격이 낮다는 뜻일 뿐,
            건강·맛·포만감·만족도가 더 높다는 뜻은 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "조건 비교":
    section_header(
        "Rule-based Comparison",
        "비슷한 조건의 상품을 표로 비교합니다",
        "절대 범위와 상대 범위를 명시적으로 선택하여 ‘비슷함’의 기준을 투명하게 만듭니다.",
    )

    top1, top2, top3 = st.columns([1.5, 1, 1])
    with top1:
        reference_name = st.selectbox("기준 상품", product_names)
    with top2:
        basis_label = st.selectbox("비교 기준", ["열량", "가격"])
    with top3:
        same_category = st.checkbox("같은 카테고리만", value=True)

    basis = "총열량" if basis_label == "열량" else "가격"
    ref = df.loc[df["상품명"] == reference_name].iloc[0]

    mode_col, tol_col, sort_col = st.columns(3)
    with mode_col:
        tolerance_mode = st.radio("범위 방식", ["절대값", "상대비율"], horizontal=True)
    with tol_col:
        if tolerance_mode == "절대값":
            if basis == "총열량":
                tolerance_value = st.slider("허용 열량 차이(kcal)", 0, 150, 20, 5)
            else:
                tolerance_value = st.slider("허용 가격 차이(원)", 0, 2000, 300, 50)
        else:
            tolerance_value = st.slider("허용 차이율(%)", 0, 50, 10, 1)
    with sort_col:
        sort_label = st.selectbox(
            "정렬 기준",
            ["기준과 가까운 순", "원/100kcal 낮은 순", "가격 낮은 순", "열량 낮은 순"],
        )

    result = similar_products(
        df,
        reference_name,
        basis,
        tolerance_mode,
        float(tolerance_value),
        same_category=same_category,
        include_reference=True,
    )

    sort_map = {
        "기준과 가까운 순": "열량차이율(%)" if basis == "총열량" else "가격차이율(%)",
        "원/100kcal 낮은 순": "원/100kcal",
        "가격 낮은 순": "가격",
        "열량 낮은 순": "총열량",
    }
    result = result.sort_values(sort_map[sort_label]).reset_index(drop=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("기준 상품 가격", f"{ref['가격']:,.0f}원", reference_name)
    with m2:
        metric_card("기준 상품 열량", f"{ref['총열량']:,.0f}kcal", f"{ref['총내용량']:g}{ref['단위']}")
    with m3:
        metric_card("원/100kcal", f"{ref['원/100kcal']:,.0f}원", "에너지 단위가격")
    with m4:
        metric_card("조건 충족 상품", f"{len(result):,}개", "기준 상품 포함")

    table_cols = [
        "기준상품",
        "상품명",
        "카테고리",
        "가격",
        "총열량",
        "총내용량",
        "단위",
        "원/100kcal",
        "kcal/1000원",
        "원/100단위",
        "당류",
        "나트륨",
        "단백질",
        "열량차이",
        "가격차이",
        "열량차이율(%)",
        "가격차이율(%)",
    ]
    if result.empty:
        st.warning("설정한 범위 안에 상품이 없습니다. 범위를 넓혀 보세요.")
    else:
        st.dataframe(format_table(result[table_cols]), use_container_width=True, hide_index=True, height=410)

        plot = result.copy()
        plot["표시"] = np.where(plot["상품명"] == reference_name, "선택 상품", "비교 상품")
        fig = px.scatter(
            plot,
            x="총열량",
            y="가격",
            color="표시",
            size="총내용량",
            hover_name="상품명",
            hover_data=["카테고리", "원/100kcal", "당류", "나트륨", "단백질"],
            color_discrete_map={"선택 상품": ORANGE, "비교 상품": "#9D9A96"},
            title="가격-열량 비교 지도",
        )
        fig = style_fig(fig, 470)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="insight">
            <b>설계 결정:</b> 절대 범위는 이해하기 쉽고, 상대 범위는 상품 규모에 따라 비교 폭이 조정됩니다.
            예를 들어 10kcal 차이는 100kcal 상품에서는 10%지만 500kcal 상품에서는 2%입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "KNN 추천":
    section_header(
        "K-Nearest Neighbors",
        "표준화된 거리로 여러 조건이 비슷한 상품을 찾습니다",
        "가격·열량·영양성분의 단위가 다르므로 평균 0, 표준편차 1의 표준점수로 바꾼 뒤 가중 거리를 계산합니다.",
    )

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        reference_name = st.selectbox("기준 상품", product_names, key="knn_reference")
    with c2:
        mode = st.selectbox("소비 목적", ["가성비", "저열량", "영양균형", "직접 설정"])
    with c3:
        k = st.slider("추천 상품 수 K", 2, min(10, max(2, len(df) - 1)), 5)

    same_category = st.checkbox("같은 카테고리 안에서 추천", value=True, key="knn_category")

    all_features = ["가격", "총열량", "총내용량", "당류", "나트륨", "단백질", "원/100kcal", "원/100단위"]
    if mode == "직접 설정":
        selected_features = st.multiselect(
            "거리 계산에 사용할 변수",
            all_features,
            default=["가격", "총열량", "당류", "나트륨", "단백질"],
        )
        feature_weights: Dict[str, float] = {}
        if selected_features:
            weight_cols = st.columns(min(4, len(selected_features)))
            for i, feature in enumerate(selected_features):
                with weight_cols[i % len(weight_cols)]:
                    feature_weights[feature] = st.slider(
                        f"{feature} 가중치",
                        0.0,
                        3.0,
                        1.0,
                        0.1,
                        key=f"w_{feature}",
                    )
        else:
            feature_weights = {}
    else:
        feature_weights = mode_feature_weights(mode)
        st.markdown(
            "사용 변수: "
            + " · ".join([f"**{name}**({weight:g})" for name, weight in feature_weights.items()])
        )

    try:
        knn = weighted_knn_recommend(
            df,
            reference_name,
            feature_weights,
            k=k,
            same_category=same_category,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    ref = df.loc[df["상품명"] == reference_name].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("후보 데이터", f"{knn.candidate_count}개", "결측값·카테고리 필터 후")
    with m2:
        metric_card("사용 변수", f"{len(knn.used_features)}개", "표준화 후 거리 계산")
    with m3:
        metric_card("K", f"{k}", "가까운 상품 수")
    with m4:
        top_score = knn.recommendations["유사도점수"].iloc[0] if not knn.recommendations.empty else 0
        metric_card("최고 유사도", f"{top_score:.1f}점", "확률이 아닌 상대 점수")

    tab1, tab2, tab3 = st.tabs(["추천 결과", "표준화 확인", "KNN 원리"])
    with tab1:
        rec_cols = [
            "상품명",
            "카테고리",
            "가격",
            "총열량",
            "총내용량",
            "단위",
            "당류",
            "나트륨",
            "단백질",
            "원/100kcal",
            "원/100단위",
            "가격차이",
            "열량차이",
            "KNN거리",
            "유사도점수",
        ]
        st.dataframe(
            format_table(knn.recommendations[rec_cols]),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        if not knn.recommendations.empty:
            top = knn.recommendations.iloc[0]
            fig = reference_radar(ref, top, knn.used_features, df)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 기준 상품의 표준점수")
        st.dataframe(format_table(knn.standardized_reference), use_container_width=True, hide_index=True)
        st.markdown("#### 후보 데이터의 변수 범위")
        st.dataframe(format_table(knn.feature_summary), use_container_width=True, hide_index=True)
        st.caption("표준화는 각 변수의 단위를 없애지만, 어떤 변수를 더 중요하게 볼지는 가중치가 결정합니다.")

    with tab3:
        st.latex(r"z=\frac{x-\mu}{\sigma}")
        st.latex(r"d(i,j)=\sqrt{\sum_m w_m(z_{im}-z_{jm})^2}")
        st.markdown(
            """
            1. 각 변수를 평균과 표준편차로 표준화합니다.  
            2. 선택 목적에 따라 변수별 가중치를 적용합니다.  
            3. 가중 유클리드 거리가 가장 작은 K개 상품을 추천합니다.  
            4. 유사도점수는 `100/(1+거리)`로 만든 설명용 상대 점수이며 선택 확률이 아닙니다.
            """
        )

    st.markdown(
        """
        <div class="insight">
            <b>확률과 통계 연결:</b> 가격은 천 원 단위, 나트륨은 mg 단위이므로 원자료를 그대로 사용하면 가격이 거리를 지배합니다.
            표준점수는 각 값이 평균에서 표준편차 몇 배만큼 떨어져 있는지를 나타내어 서로 다른 단위의 변수를 공통 척도로 비교하게 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "파레토 분석":
    section_header(
        "Pareto Efficiency",
        "여러 기준에서 동시에 뒤처지는 상품을 구분합니다",
        "다른 상품보다 비싸면서 목적에 맞는 특성도 나쁘다면 지배된 상품으로 볼 수 있습니다. 파레토 상품은 정답이 아니라 효율적 후보 집합입니다.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("파레토 기준", ["가성비", "저열량", "영양균형"])
    with c2:
        category = st.selectbox("카테고리", ["전체"] + categories)
    with c3:
        selected_name = st.selectbox("강조 상품", product_names)

    subset = df.copy() if category == "전체" else df[df["카테고리"] == category].copy()
    objectives = pareto_objectives(mode)
    mask = pareto_front_mask(subset, objectives)
    subset["파레토여부"] = np.where(mask, "파레토 후보", "지배됨")

    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("분석 상품", f"{len(subset)}개", category)
    with p2:
        metric_card("파레토 후보", f"{int(mask.sum())}개", "다른 상품에 완전히 지배되지 않음")
    with p3:
        ratio = mask.mean() * 100 if len(mask) else 0
        metric_card("후보 비율", f"{ratio:.1f}%", "선택 기준에 따라 변함")

    if mode == "가성비":
        x, y, size = "총열량", "가격", "총내용량"
        title = "가성비 파레토: 낮은 가격·높은 열량·높은 내용량"
    elif mode == "저열량":
        x, y, size = "총열량", "가격", "당류"
        title = "저열량 파레토: 낮은 가격·낮은 열량·낮은 당류"
    else:
        x, y, size = "당류", "가격", "단백질"
        title = "영양균형 파레토: 낮은 가격·당류·나트륨, 높은 단백질"

    subset["강조"] = np.where(subset["상품명"] == selected_name, "선택 상품", subset["파레토여부"])
    color_map = {"선택 상품": ORANGE, "파레토 후보": "#50545A", "지배됨": "#D8D4D0"}
    fig = px.scatter(
        subset,
        x=x,
        y=y,
        size=size,
        color="강조",
        hover_name="상품명",
        hover_data=["카테고리", "총열량", "가격", "당류", "나트륨", "단백질", "원/100kcal"],
        color_discrete_map=color_map,
        title=title,
    )
    fig = style_fig(fig, 520)
    st.plotly_chart(fig, use_container_width=True)

    pareto_table = subset[subset["파레토여부"] == "파레토 후보"].copy()
    st.markdown("#### 파레토 후보 표")
    st.dataframe(
        format_table(pareto_table[display_columns(pareto_table, ["파레토여부"])]),
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    st.markdown(
        """
        <div class="insight">
            파레토 후보끼리도 최종 선택은 달라질 수 있습니다. 어떤 학생은 가격을, 다른 학생은 당류나 단백질을 더 중요하게 보기 때문입니다.
            따라서 파레토 분석은 선택지를 줄이는 도구이지 하나의 상품을 강제하는 순위표가 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "심화 분석":
    section_header(
        "Advanced Analytics",
        "단위가격 분포와 헤도닉 가격 회귀를 탐색합니다",
        "가격이 열량·중량·영양성분·카테고리로 얼마나 설명되는지 분석하고, 설명되지 않은 차이를 잔차로 확인합니다.",
    )

    tab1, tab2 = st.tabs(["단위가격 분석", "헤도닉 회귀"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            metric = st.selectbox("분석 지표", ["원/100kcal", "kcal/1000원", "원/100단위", "100단위당열량"])
        with c2:
            cat_filter = st.selectbox("카테고리 필터", ["전체"] + categories, key="unit_category")
        unit_df = df if cat_filter == "전체" else df[df["카테고리"] == cat_filter]

        left, right = st.columns([1.15, 1])
        with left:
            fig = px.histogram(
                unit_df,
                x=metric,
                color_discrete_sequence=[ORANGE],
                nbins=22,
                marginal="box",
                title=f"{metric} 분포",
            )
            fig = style_fig(fig, 450)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            ranked = unit_df.sort_values(metric).head(12)
            fig = px.bar(
                ranked.sort_values(metric, ascending=False),
                x=metric,
                y="상품명",
                orientation="h",
                color_discrete_sequence=[ORANGE],
                title=f"{metric} 하위 12개 상품",
                text_auto=".3s",
            )
            fig = style_fig(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

        st.caption("원/100kcal이 낮다고 건강성이 높다는 뜻은 아닙니다. 단위가격은 여러 비교 지표 중 하나입니다.")

    with tab2:
        category = st.selectbox("회귀 범위", ["전체"] + categories, key="reg_category")
        try:
            model, reg = run_hedonic_regression(df, category)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                metric_card("분석 표본", f"{int(model.nobs)}개", "결측값 없는 상품")
            with r2:
                metric_card("R²", f"{model.rsquared:.3f}", "표본 내 설명력")
            with r3:
                metric_card("조정 R²", f"{model.rsquared_adj:.3f}", "변수 수를 고려")
            with r4:
                metric_card("평균 절대 잔차", f"{reg['residual'].abs().mean():,.0f}원", "실제-예측 가격 차이")

            coef = coefficient_table(model)
            st.markdown("#### 회귀계수")
            st.dataframe(format_table(coef), use_container_width=True, hide_index=True, height=330)

            fig = px.scatter(
                reg,
                x="predicted_price",
                y="price",
                hover_name="product",
                color="residual",
                color_continuous_scale=["#59636E", "#FFFFFF", ORANGE],
                title="실제 가격과 회귀 예측 가격",
                labels={"predicted_price": "예측 가격", "price": "실제 가격", "residual": "잔차"},
            )
            lo = min(reg["predicted_price"].min(), reg["price"].min())
            hi = max(reg["predicted_price"].max(), reg["price"].max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#999", dash="dash"))
            fig = style_fig(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

            reg_view = reg.rename(
                columns={
                    "product": "상품명",
                    "category": "카테고리",
                    "price": "실제가격",
                    "predicted_price": "예측가격",
                    "residual": "잔차",
                }
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 예상보다 가격이 높은 상품")
                st.dataframe(
                    format_table(reg_view.sort_values("잔차", ascending=False)[["상품명", "카테고리", "실제가격", "예측가격", "잔차"]].head(10)),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_b:
                st.markdown("#### 예상보다 가격이 낮은 상품")
                st.dataframe(
                    format_table(reg_view.sort_values("잔차")[["상품명", "카테고리", "실제가격", "예측가격", "잔차"]].head(10)),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                """
                <div class="insight">
                    <b>해석 주의:</b> 양의 잔차를 곧바로 ‘브랜드 프리미엄’, 음의 잔차를 ‘저평가’라고 확정할 수 없습니다.
                    맛·브랜드 인지도·포장·유통비처럼 수집하지 않은 요인이 잔차에 함께 들어갑니다.
                </div>
                """,
                unsafe_allow_html=True,
            )


elif page == "데이터 관리":
    section_header(
        "Data Management",
        "데이터 수집과 정제가 탐구의 출발점입니다",
        "제품 전체 기준의 열량과 중량을 기록하고, 사용자 제보는 세션에만 임시 추가한 뒤 CSV로 내려받을 수 있습니다.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("입력 행", f"{cleaning_summary.input_rows}개", "업로드·샘플·제보 합계")
    with m2:
        metric_card("정제 후", f"{cleaning_summary.output_rows}개", "필수값 검증 후")
    with m3:
        metric_card("중복 제거", f"{cleaning_summary.duplicate_rows_removed}개", "상품명+카테고리 기준")
    with m4:
        metric_card("유효하지 않은 행", f"{cleaning_summary.invalid_rows_removed}개", "가격·내용량·열량 0 이하 등")

    tab1, tab2, tab3 = st.tabs(["데이터 표", "상품 제보", "조사 지침"])

    with tab1:
        st.dataframe(
            format_table(df[display_columns(df)]),
            use_container_width=True,
            hide_index=True,
            height=520,
        )
        st.download_button(
            "정제·파생지표 포함 CSV 받기",
            data=csv_bytes(df),
            file_name="snacklens_cleaned_products.csv",
            mime="text/csv",
        )

    with tab2:
        st.caption("제보 데이터는 현재 브라우저 세션에만 저장됩니다. 앱을 새로 시작하면 사라질 수 있으므로 CSV를 내려받아 합쳐 주세요.")
        with st.form("add_product_form", clear_on_submit=True):
            a, b, c = st.columns(3)
            with a:
                product = st.text_input("상품명")
                category = st.selectbox("카테고리", categories + ["기타"])
                price = st.number_input("가격(원)", min_value=1, value=1500, step=100)
            with b:
                amount = st.number_input("총내용량", min_value=0.1, value=60.0, step=1.0)
                unit = st.selectbox("단위", ["g", "mL"])
                calories = st.number_input("총열량(kcal)", min_value=1.0, value=200.0, step=5.0)
            with c:
                sugar = st.number_input("당류(g)", min_value=0.0, value=10.0, step=0.5)
                sodium = st.number_input("나트륨(mg)", min_value=0.0, value=150.0, step=10.0)
                protein = st.number_input("단백질(g)", min_value=0.0, value=3.0, step=0.5)
            note = st.text_input("비고(선택)")
            submitted = st.form_submit_button("세션 데이터에 추가", use_container_width=True)

        if submitted:
            if not product.strip():
                st.error("상품명을 입력하세요.")
            else:
                st.session_state.added_products.append(
                    {
                        "상품명": product.strip(),
                        "카테고리": category,
                        "가격": price,
                        "총내용량": amount,
                        "단위": unit,
                        "총열량": calories,
                        "당류": sugar,
                        "나트륨": sodium,
                        "단백질": protein,
                        "데이터출처": "사용자 제보",
                        "비고": note,
                    }
                )
                st.success("상품을 추가했습니다. 페이지가 다시 실행됩니다.")
                st.rerun()

    with tab3:
        st.markdown(
            """
            **현장 조사 권장 절차**

            1. 매점 가격표와 제품 영양성분표를 함께 촬영한다.  
            2. ‘1회 제공량’이 아니라 **제품 전체의 총열량**인지 확인한다.  
            3. 총내용량은 g 또는 mL로 기록하고 단위를 별도 열에 보존한다.  
            4. 당류·나트륨·단백질은 가능하면 제품 전체 기준으로 환산한다.  
            5. 품절·가격 변경 날짜를 비고에 기록하여 데이터의 시점을 남긴다.  
            6. 같은 제품이 중복되면 가장 최근 조사값을 사용한다.
            """
        )
        missing_table = pd.DataFrame(
            {"열": cleaning_summary.missing_by_column.keys(), "결측값 수": cleaning_summary.missing_by_column.values()}
        )
        st.markdown("#### 원자료 결측값")
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight">
            실제 제출본에서는 샘플 데이터를 그대로 사용하지 말고 학교 매점의 실측 자료로 교체해야 합니다.
            앱의 분석 결과는 데이터의 정확도와 최신성에 직접 의존합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
