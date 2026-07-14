import warnings
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Market Cap Top 10",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 기업 정보
# 시가총액 순위는 변동될 수 있으므로 필요할 때 수정
# ---------------------------------------------------------
COMPANIES = {
    "NVIDIA": {
        "ticker": "NVDA",
        "country": "미국",
        "sector": "반도체",
    },
    "Apple": {
        "ticker": "AAPL",
        "country": "미국",
        "sector": "IT 하드웨어",
    },
    "Alphabet": {
        "ticker": "GOOG",
        "country": "미국",
        "sector": "인터넷·광고",
    },
    "Microsoft": {
        "ticker": "MSFT",
        "country": "미국",
        "sector": "소프트웨어·클라우드",
    },
    "Amazon": {
        "ticker": "AMZN",
        "country": "미국",
        "sector": "전자상거래·클라우드",
    },
    "TSMC": {
        "ticker": "TSM",
        "country": "대만",
        "sector": "반도체 파운드리",
    },
    "SpaceX": {
        # Yahoo Finance에서 지원되지 않으면 자동 제외됨
        "ticker": "SPCX",
        "country": "미국",
        "sector": "우주·통신",
    },
    "Broadcom": {
        "ticker": "AVGO",
        "country": "미국",
        "sector": "반도체·소프트웨어",
    },
    "Saudi Aramco": {
        "ticker": "2222.SR",
        "country": "사우디아라비아",
        "sector": "에너지",
    },
    "Meta Platforms": {
        "ticker": "META",
        "country": "미국",
        "sector": "소셜미디어·광고",
    },
}


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #FAFAFA;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            color: #666666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #E8E8E8;
            border-radius: 12px;
            padding: 14px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
        }

        .notice-box {
            background-color: #FFF7ED;
            border-left: 5px solid #F97316;
            border-radius: 8px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 18px;
        }

        .footer {
            text-align: center;
            color: #888888;
            font-size: 0.85rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 데이터 다운로드 함수
# ---------------------------------------------------------
@st.cache_data(ttl=60 * 60)
def download_stock_data(ticker_list, start_date, end_date):
    """
    Yahoo Finance에서 여러 종목의 주가를 다운로드한다.

    auto_adjust=True:
    주식 분할과 배당 등을 반영한 가격을 가져온다.
    """
    if not ticker_list:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=ticker_list,
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        if data.empty:
            return pd.DataFrame()

        # 여러 종목인 경우 MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):
                return pd.DataFrame()

            close_data = data["Close"].copy()

        # 한 종목만 선택한 경우
        else:
            if "Close" not in data.columns:
                return pd.DataFrame()

            close_data = data[["Close"]].copy()
            close_data.columns = ticker_list[:1]

        if isinstance(close_data, pd.Series):
            close_data = close_data.to_frame()

        close_data.index = pd.to_datetime(close_data.index)
        close_data = close_data.sort_index()

        # 데이터가 전부 없는 종목 제거
        close_data = close_data.dropna(axis=1, how="all")

        return close_data

    except Exception:
        return pd.DataFrame()


def calculate_statistics(price_data, ticker_to_name):
    """기업별 수익률 및 변동성 통계를 계산한다."""
    statistics = []

    for ticker in price_data.columns:
        series = price_data[ticker].dropna()

        if len(series) < 2:
            continue

        first_price = series.iloc[0]
        last_price = series.iloc[-1]

        period_return = (last_price / first_price - 1) * 100

        daily_returns = series.pct_change().dropna()
        annual_volatility = daily_returns.std() * (252**0.5) * 100

        statistics.append(
            {
                "기업": ticker_to_name.get(ticker, ticker),
                "티커": ticker,
                "시작 가격": first_price,
                "현재 가격": last_price,
                "기간 수익률(%)": period_return,
                "연환산 변동성(%)": annual_volatility,
                "기간 최고가": series.max(),
                "기간 최저가": series.min(),
                "거래일 수": len(series),
            }
        )

    stats_df = pd.DataFrame(statistics)

    if not stats_df.empty:
        stats_df = stats_df.sort_values(
            "기간 수익률(%)",
            ascending=False,
        ).reset_index(drop=True)

        stats_df.index = stats_df.index + 1
        stats_df.index.name = "순위"

    return stats_df


def create_normalized_data(price_data, ticker_to_name):
    """
    각 기업의 첫 거래일 가격을 100으로 환산한다.

    서로 가격 수준이 다른 주식의 변화율을
    한 그래프에서 비교할 수 있다.
    """
    normalized = pd.DataFrame(index=price_data.index)

    for ticker in price_data.columns:
        series = price_data[ticker].dropna()

        if series.empty:
            continue

        first_price = series.iloc[0]
        normalized[ticker_to_name.get(ticker, ticker)] = (
            price_data[ticker] / first_price * 100
        )

    return normalized


# ---------------------------------------------------------
# 제목
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🌎 글로벌 시가총액 Top 10 주식 대시보드</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        글로벌 시가총액 상위 기업의 주가 변화와 투자 지표를 비교합니다.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
st.sidebar.header("⚙️ 대시보드 설정")

today = datetime.today().date()
default_start = today - timedelta(days=365)

period_option = st.sidebar.selectbox(
    "조회 기간",
    options=["최근 1개월", "최근 3개월", "최근 6개월", "최근 1년", "직접 설정"],
    index=3,
)

if period_option == "최근 1개월":
    start_date = today - timedelta(days=30)
    end_date = today

elif period_option == "최근 3개월":
    start_date = today - timedelta(days=90)
    end_date = today

elif period_option == "최근 6개월":
    start_date = today - timedelta(days=180)
    end_date = today

elif period_option == "최근 1년":
    start_date = default_start
    end_date = today

else:
    date_range = st.sidebar.date_input(
        "날짜 범위",
        value=(default_start, today),
        max_value=today,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = default_start
        end_date = today


selected_companies = st.sidebar.multiselect(
    "비교할 기업",
    options=list(COMPANIES.keys()),
    default=list(COMPANIES.keys()),
)

chart_mode = st.sidebar.radio(
    "기본 비교 방식",
    options=["정규화 주가", "실제 종가"],
    index=0,
)

show_moving_average = st.sidebar.checkbox(
    "개별 차트에 이동평균선 표시",
    value=True,
)

moving_average_period = st.sidebar.slider(
    "이동평균 기간",
    min_value=5,
    max_value=60,
    value=20,
    step=5,
    disabled=not show_moving_average,
)

st.sidebar.divider()

st.sidebar.caption(
    "주가 데이터: Yahoo Finance\n\n"
    "시가총액 순위는 시장 상황에 따라 변할 수 있습니다."
)


# ---------------------------------------------------------
# 입력값 검사
# ---------------------------------------------------------
if not selected_companies:
    st.warning("사이드바에서 한 개 이상의 기업을 선택해 주세요.")
    st.stop()

if start_date >= end_date:
    st.error("시작 날짜는 종료 날짜보다 앞서야 합니다.")
    st.stop()


selected_tickers = [
    COMPANIES[company]["ticker"]
    for company in selected_companies
]

ticker_to_name = {
    information["ticker"]: company
    for company, information in COMPANIES.items()
}


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
with st.spinner("Yahoo Finance에서 주가 데이터를 불러오는 중입니다..."):
    price_data = download_stock_data(
        selected_tickers,
        start_date,
        end_date,
    )

if price_data.empty:
    st.error(
        "주가 데이터를 불러오지 못했습니다. "
        "잠시 후 다시 실행하거나 조회 기간을 변경해 주세요."
    )
    st.stop()


available_tickers = list(price_data.columns)
unavailable_tickers = [
    ticker
    for ticker in selected_tickers
    if ticker not in available_tickers
]

if unavailable_tickers:
    unavailable_names = [
        ticker_to_name.get(ticker, ticker)
        for ticker in unavailable_tickers
    ]

    st.markdown(
        f"""
        <div class="notice-box">
            <b>일부 종목의 데이터를 가져오지 못했습니다.</b><br>
            제외된 기업: {", ".join(unavailable_names)}<br>
            비상장 기업이거나 Yahoo Finance에서 해당 티커를 지원하지 않을 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 통계 계산
# ---------------------------------------------------------
statistics_df = calculate_statistics(
    price_data,
    ticker_to_name,
)

normalized_data = create_normalized_data(
    price_data,
    ticker_to_name,
)


# ---------------------------------------------------------
# 상단 핵심 지표
# ---------------------------------------------------------
if not statistics_df.empty:
    best_stock = statistics_df.iloc[0]
    worst_stock = statistics_df.iloc[-1]

    average_return = statistics_df["기간 수익률(%)"].mean()
    average_volatility = statistics_df["연환산 변동성(%)"].mean()

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        label="수익률 1위",
        value=best_stock["기업"],
        delta=f'{best_stock["기간 수익률(%)"]:.2f}%',
    )

    metric2.metric(
        label="수익률 최하위",
        value=worst_stock["기업"],
        delta=f'{worst_stock["기간 수익률(%)"]:.2f}%',
    )

    metric3.metric(
        label="평균 수익률",
        value=f"{average_return:.2f}%",
    )

    metric4.metric(
        label="평균 연환산 변동성",
        value=f"{average_volatility:.2f}%",
    )


# ---------------------------------------------------------
# 탭 구성
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 종합 비교",
        "🏢 기업별 분석",
        "🏆 수익률 순위",
        "📋 데이터",
    ]
)


# ---------------------------------------------------------
# 탭 1: 종합 비교
# ---------------------------------------------------------
with tab1:
    st.subheader("기업별 주가 변화 비교")

    if chart_mode == "정규화 주가":
        long_normalized = (
            normalized_data
            .reset_index()
            .rename(columns={"Date": "날짜", "index": "날짜"})
            .melt(
                id_vars="날짜",
                var_name="기업",
                value_name="정규화 주가",
            )
            .dropna()
        )

        comparison_figure = px.line(
            long_normalized,
            x="날짜",
            y="정규화 주가",
            color="기업",
            title="주가 변화 비교: 시작일 = 100",
            labels={
                "날짜": "날짜",
                "정규화 주가": "정규화 주가",
            },
        )

        comparison_figure.add_hline(
            y=100,
            line_dash="dash",
            line_color="gray",
            annotation_text="시작 기준",
        )

        comparison_figure.update_layout(
            hovermode="x unified",
            height=600,
            legend_title_text="기업",
            margin=dict(l=20, r=20, t=70, b=20),
        )

        comparison_figure.update_traces(
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "날짜: %{x|%Y-%m-%d}<br>"
                "정규화 주가: %{y:.2f}"
                "<extra></extra>"
            )
        )

    else:
        renamed_price_data = price_data.rename(
            columns=ticker_to_name
        )

        long_price = (
            renamed_price_data
            .reset_index()
            .rename(columns={"Date": "날짜", "index": "날짜"})
            .melt(
                id_vars="날짜",
                var_name="기업",
                value_name="종가",
            )
            .dropna()
        )

        comparison_figure = px.line(
            long_price,
            x="날짜",
            y="종가",
            color="기업",
            title="기업별 실제 종가 변화",
            labels={
                "날짜": "날짜",
                "종가": "종가",
            },
        )

        comparison_figure.update_layout(
            hovermode="x unified",
            height=600,
            legend_title_text="기업",
            margin=dict(l=20, r=20, t=70, b=20),
        )

    st.plotly_chart(
        comparison_figure,
        use_container_width=True,
    )

    st.info(
        "정규화 주가는 각 기업의 조회 시작일 가격을 100으로 환산한 값입니다. "
        "주식마다 실제 가격이 달라도 상승률과 하락률을 쉽게 비교할 수 있습니다."
    )

    st.subheader("일간 수익률 분포")

    daily_returns = (
        price_data
        .pct_change(fill_method=None)
        .rename(columns=ticker_to_name)
        * 100
    )

    daily_returns_long = (
        daily_returns
        .reset_index()
        .melt(
            id_vars=daily_returns.index.name or "Date",
            var_name="기업",
            value_name="일간 수익률",
        )
        .dropna()
    )

    date_column = daily_returns.index.name or "Date"
    daily_returns_long = daily_returns_long.rename(
        columns={date_column: "날짜"}
    )

    box_figure = px.box(
        daily_returns_long,
        x="기업",
        y="일간 수익률",
        color="기업",
        points=False,
        title="기업별 일간 수익률 분포",
        labels={
            "기업": "기업",
            "일간 수익률": "일간 수익률(%)",
        },
    )

    box_figure.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(
        box_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# 탭 2: 기업별 분석
# ---------------------------------------------------------
with tab2:
    available_company_names = [
        ticker_to_name[ticker]
        for ticker in available_tickers
    ]

    selected_detail_company = st.selectbox(
        "상세 분석할 기업",
        options=available_company_names,
    )

    detail_ticker = COMPANIES[selected_detail_company]["ticker"]
    detail_series = price_data[detail_ticker].dropna()

    detail_return = (
        detail_series.iloc[-1] / detail_series.iloc[0] - 1
    ) * 100

    detail_daily_return = detail_series.pct_change().dropna()
    detail_volatility = (
        detail_daily_return.std() * (252**0.5) * 100
    )

    company_info = COMPANIES[selected_detail_company]

    detail1, detail2, detail3, detail4 = st.columns(4)

    detail1.metric(
        "현재 가격",
        f"${detail_series.iloc[-1]:,.2f}",
    )

    detail2.metric(
        "기간 수익률",
        f"{detail_return:.2f}%",
    )

    detail3.metric(
        "연환산 변동성",
        f"{detail_volatility:.2f}%",
    )

    detail4.metric(
        "기간 최고가",
        f"${detail_series.max():,.2f}",
    )

    st.caption(
        f"티커: {detail_ticker} · "
        f"국가: {company_info['country']} · "
        f"산업: {company_info['sector']}"
    )

    detail_df = detail_series.to_frame(name="종가")

    if show_moving_average:
        moving_average_column = f"{moving_average_period}일 이동평균"
        detail_df[moving_average_column] = (
            detail_df["종가"]
            .rolling(moving_average_period)
            .mean()
        )

    detail_figure = go.Figure()

    detail_figure.add_trace(
        go.Scatter(
            x=detail_df.index,
            y=detail_df["종가"],
            mode="lines",
            name="종가",
            line=dict(width=2),
            hovertemplate=(
                "날짜: %{x|%Y-%m-%d}<br>"
                "종가: $%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    if show_moving_average:
        detail_figure.add_trace(
            go.Scatter(
                x=detail_df.index,
                y=detail_df[moving_average_column],
                mode="lines",
                name=moving_average_column,
                line=dict(width=2, dash="dash"),
                hovertemplate=(
                    "날짜: %{x|%Y-%m-%d}<br>"
                    f"{moving_average_column}: "
                    "$%{y:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

    detail_figure.update_layout(
        title=f"{selected_detail_company} 주가 변화",
        xaxis_title="날짜",
        yaxis_title="주가",
        hovermode="x unified",
        height=550,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    detail_figure.update_xaxes(
        rangeslider_visible=True
    )

    st.plotly_chart(
        detail_figure,
        use_container_width=True,
    )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        return_histogram = px.histogram(
            detail_daily_return * 100,
            nbins=35,
            title=f"{selected_detail_company} 일간 수익률 히스토그램",
            labels={
                "value": "일간 수익률(%)",
                "count": "빈도",
            },
        )

        return_histogram.update_layout(
            showlegend=False,
            height=420,
        )

        st.plotly_chart(
            return_histogram,
            use_container_width=True,
        )

    with right_chart:
        cumulative_return = (
            (1 + detail_daily_return).cumprod() - 1
        ) * 100

        cumulative_figure = px.area(
            x=cumulative_return.index,
            y=cumulative_return.values,
            title=f"{selected_detail_company} 누적 수익률",
            labels={
                "x": "날짜",
                "y": "누적 수익률(%)",
            },
        )

        cumulative_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        cumulative_figure.update_layout(
            height=420,
            showlegend=False,
        )

        st.plotly_chart(
            cumulative_figure,
            use_container_width=True,
        )


# ---------------------------------------------------------
# 탭 3: 수익률 순위
# ---------------------------------------------------------
with tab3:
    st.subheader("조회 기간 수익률 순위")

    if not statistics_df.empty:
        ranking_df = statistics_df.sort_values(
            "기간 수익률(%)",
            ascending=True,
        )

        ranking_figure = px.bar(
            ranking_df,
            x="기간 수익률(%)",
            y="기업",
            orientation="h",
            text="기간 수익률(%)",
            title="기업별 기간 수익률",
            labels={
                "기간 수익률(%)": "기간 수익률(%)",
                "기업": "기업",
            },
            color="기간 수익률(%)",
            color_continuous_scale="RdYlGn",
        )

        ranking_figure.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        ranking_figure.update_layout(
            height=560,
            coloraxis_showscale=False,
            margin=dict(l=20, r=70, t=70, b=20),
        )

        ranking_figure.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
        )

        st.plotly_chart(
            ranking_figure,
            use_container_width=True,
        )

        scatter_figure = px.scatter(
            statistics_df,
            x="연환산 변동성(%)",
            y="기간 수익률(%)",
            text="기업",
            size="거래일 수",
            hover_name="기업",
            title="수익률과 변동성의 관계",
            labels={
                "연환산 변동성(%)": "연환산 변동성(%)",
                "기간 수익률(%)": "기간 수익률(%)",
            },
        )

        scatter_figure.update_traces(
            textposition="top center",
        )

        scatter_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        scatter_figure.update_layout(
            height=540,
            margin=dict(l=20, r=20, t=70, b=20),
        )

        st.plotly_chart(
            scatter_figure,
            use_container_width=True,
        )


# ---------------------------------------------------------
# 탭 4: 데이터
# ---------------------------------------------------------
with tab4:
    st.subheader("기업별 핵심 통계")

    display_statistics = statistics_df.copy()

    numeric_columns = [
        "시작 가격",
        "현재 가격",
        "기간 수익률(%)",
        "연환산 변동성(%)",
        "기간 최고가",
        "기간 최저가",
    ]

    for column in numeric_columns:
        if column in display_statistics.columns:
            display_statistics[column] = (
                display_statistics[column].round(2)
            )

    st.dataframe(
        display_statistics,
        use_container_width=True,
    )

    st.subheader("원본 종가 데이터")

    downloadable_price_data = (
        price_data
        .rename(columns=ticker_to_name)
        .reset_index()
    )

    st.dataframe(
        downloadable_price_data.tail(100),
        use_container_width=True,
    )

    csv_data = downloadable_price_data.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 주가 데이터 CSV 다운로드",
        data=csv_data,
        file_name=(
            f"global_top10_stock_data_"
            f"{start_date}_{end_date}.csv"
        ),
        mime="text/csv",
    )


# ---------------------------------------------------------
# 하단 안내
# ---------------------------------------------------------
st.markdown(
    """
    <div class="notice-box">
        <b>주의</b><br>
        이 대시보드는 데이터 분석 및 교육 목적으로 제작되었습니다.
        표시된 정보는 투자 권유나 매수·매도 추천이 아닙니다.
        Yahoo Finance 데이터에는 지연이나 누락이 발생할 수 있습니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="footer">
        데이터 조회 기간: {start_date} ~ {end_date}<br>
        마지막 화면 생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
    """,
    unsafe_allow_html=True,
)
