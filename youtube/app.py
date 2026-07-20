import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import urllib.request
import os
from collections import Counter
from datetime import datetime

# --- 한글 폰트 설정 (스트림릿 클라우드용 자동 다운로드) ---
@st.cache_resource
def get_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    return font_path

font_path = get_korean_font()
plt.rc('font', family='NanumGothic') 
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# --- 유튜브 댓글 수집 함수 ---
@st.cache_data(show_spinner=False)
def fetch_youtube_comments(api_key, video_id, max_results):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results),
            textFormat="plainText"
        )
        
        while request and len(comments) < max_results:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'published_at': comment['publishedAt'],
                    'like_count': comment['likeCount']
                })
                
                if len(comments) >= max_results:
                    break
                    
            request = youtube.commentThreads().list_next(request, response)
            
    except Exception as e:
        st.error(f"댓글을 가져오는 중 오류가 발생했습니다: {e}")
        
    return pd.DataFrame(comments)

# --- 간단한 한글 감성 분석 함수 (사전 기반) ---
def analyze_sentiment(text):
    positive_words = ['좋', '최고', '완벽', '감사', '응원', '멋', '재밌', '재미', '사랑', '대박', '유익', '화이팅']
    negative_words = ['노잼', '별로', '최악', '싫', '망', '짜증', '구리', '쓰레기', '지루', '아쉽']
    
    pos_score = sum(1 for word in positive_words if word in text)
    neg_score = sum(1 for word in negative_words if word in text)
    
    if pos_score > neg_score:
        return '긍정'
    elif neg_score > pos_score:
        return '부정'
    else:
        return '중립'

# --- UI 및 앱 로직 ---
st.set_page_config(page_title="유튜브 댓글 분석기", layout="wide")
st.title("📊 유튜브 댓글 분석기")

# 사이드바 설정
st.sidebar.header("설정 (Settings)")
api_key = st.sidebar.text_input("YouTube API Key를 입력하세요", type="password")
video_url = st.sidebar.text_input("유튜브 영상 링크를 입력하세요")
max_comments = st.sidebar.slider("분석할 댓글 수", min_value=50, max_value=500, value=100, step=50)
analyze_btn = st.sidebar.button("분석 시작")

if analyze_btn:
    if not api_key:
        st.sidebar.error("API Key를 입력해주세요.")
    elif not video_url:
        st.sidebar.error("유튜브 영상 링크를 입력해주세요.")
    else:
        # 영상 ID 추출
        video_id = ""
        if "v=" in video_url:
            video_id = video_url.split("v=")[1][:11]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1][:11]
            
        if not video_id:
            st.error("유효한 유튜브 링크가 아닙니다.")
        else:
            # 1. 영상 보여주기
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            st.divider()
            
            with st.spinner('댓글을 수집하고 분석하는 중입니다...'):
                df = fetch_youtube_comments(api_key, video_id, max_comments)
                
                if not df.empty:
                    # 데이터 전처리
                    df['published_at'] = pd.to_datetime(df['published_at'])
                    df['date'] = df['published_at'].dt.date
                    df['sentiment'] = df['text'].apply(analyze_sentiment)
                    
                    col1, col2 = st.columns(2)
                    
                    # 2. 시간대별 댓글 작성 추이
                    with col1:
                        st.subheader("📈 일자별 댓글 작성 추이")
                        trend_data = df.groupby('date').size()
                        st.line_chart(trend_data)
                        
                    # 3. 댓글 반응도 (감성 분석)
                    with col2:
                        st.subheader("😊 댓글 반응도 (긍정/부정)")
                        sentiment_counts = df['sentiment'].value_counts()
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                               colors=['#66b3ff', '#99ff99', '#ff9999'])
                        st.pyplot(fig)
                        
                    st.divider()
                    
                    # 4. 한글 워드클라우드
                    st.subheader("☁️ 댓글 키워드 워드클라우드")
                    # 한글만 추출
                    all_text = ' '.join(df['text'].tolist())
                    korean_text = re.sub(r'[^가-힣\s]', '', all_text)
                    
                    if len(korean_text.strip()) > 0:
                        wordcloud = WordCloud(
                            font_path=font_path,
                            width=800, 
                            height=400, 
                            background_color='white',
                            max_words=100
                        ).generate(korean_text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("워드클라우드를 생성할 한글 데이터가 부족합니다.")
                        
                    # 원본 데이터 확인
                    with st.expander("원본 댓글 데이터 보기"):
                        st.dataframe(df[['author', 'text', 'published_at', 'sentiment']])
                else:
                    st.warning("수집된 댓글이 없습니다. API 할당량이 초과되었거나 댓글이 차단된 영상일 수 있습니다.")
