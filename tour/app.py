import streamlit as st
import pandas as pd
from api import get_festival

st.set_page_config(
    page_title="대한민국 축제여행",
    page_icon="🎉",
    layout="wide"
)

st.title("🎉 대한민국 축제 여행")

area = st.selectbox(
    "지역 선택",
    [
        "전체","서울","부산","대구","인천",
        "광주","대전","울산","세종",
        "경기","강원","충북","충남",
        "전북","전남","경북","경남","제주"
    ]
)

keyword = st.text_input("검색")

if st.button("축제 검색"):

    df = get_festival(area, keyword)

    if len(df)==0:
        st.warning("검색 결과가 없습니다.")
    else:
        st.dataframe(df)

        for _, row in df.iterrows():

            with st.container():

                col1,col2 = st.columns([1,3])

                with col1:
                    st.image(row["image"], width=220)

                with col2:

                    st.subheader(row["title"])

                    st.write(row["addr"])

                    st.write(row["date"])

                    st.write(row["overview"])

                    st.link_button(
                        "상세보기",
                        row["homepage"]
                    )
