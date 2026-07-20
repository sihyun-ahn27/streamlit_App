import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import urllib.request
import os
from datetime import datetime

# --- 한글 폰트 설정 (스트림릿 클라우드용) ---
@st.cache_resource
def get_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    return font_path

font_path = get_korean_font()
plt.rc('font', family='NanumGothic') 
plt.rcParams['axes.unicode_minus'] = False 

# --- API 키 설정 (Streamlit Secrets에서 가져오기) ---
try:
    API_KEY = st.secrets["YOUTUBE_API_KEY"]
except KeyError:
    st.error("⚠️ Streamlit Secrets에 'YOUTUBE_API_KEY'가 설정되지 않았습니다.")
    st.stop()

# --- 유튜브 영상 통계 수집 함수 (조회수, 좋아요) ---
@st.cache_data(show_spinner=False)
def fetch_video_stats(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.videos().list(part="statistics,snippet", id=video_id)
        response = request.execute()
        if response['items']:
            return response['items'][0]
    except Exception as e:
        st.error(f"영상 정보를 가져오는 중 오류가 발생했습니다: {e}")
    return None

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
        st.warning("댓글을 가져올 수 없습니다. 댓글이 사용 중지된 영상일 수 있습니다.")
    return pd.DataFrame(comments)

# --- 간단한 한글 감성 분석 함수 ---
def analyze_sentiment(text):
    positive_words = ['좋', '최고', '완벽', '감사', '응원', '멋', '재밌', '재미', '사랑', '대박', '유익', '화이팅', '기대', '축하']
    negative_words = ['노잼', '별로', '최악', '싫', '망', '짜증', '구리', '쓰레기', '지루', '아쉽', '실망']
    
    pos_score = sum(1 for word in positive_words if word in text)
    neg_score = sum(1 for word in negative_words if word in text)
    
    if pos_score > neg_score: return '긍정 😊'
    elif neg_score > pos_score: return '부정 😠'
    else: return '중립 😐'

# --- 불용어 처리 함수 (워드클라우드용) ---
def clean_text_for_wordcloud(text):
    text = re.sub(r'[^가-힣\s]', '', text) # 한글과 공백만 남기기
    stop_words = ['진짜', '너무', '정말', '많이', '있는', '하는', '이런', '그리고', '이', '그', '저', '수', '것', '영상', '댓글']
    words = text.split()
    clean_words = [word for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(clean_words)

# ==========================================
# UI 및 앱 로직
# ==========================================
st.set_page_config(page_title="유튜브 인사이트 분석기", layout="wide", page_icon="📊")
st.title("📊 유튜브 영상 & 댓글 분석기")

# 사이드바 설정
st.sidebar.header("설정 (Settings)")
video_url = st.sidebar.text_input("유튜브 영상 링크를 입력하세요")
max_comments = st.sidebar.slider("분석할 최대 댓글 수", min_value=50, max_value=1000, value=200, step=50)
analyze_btn = st.sidebar.button("분석 시작", type="primary", use_container_width=True)

if analyze_btn:
    if not video_url:
        st.sidebar.error("유튜브 영상 링크를 입력해주세요.")
    else:
        # 영상 ID 추출 로직
        video_id = ""
        if "v=" in video_url:
            video_id = video_url.split("v=")[1][:11]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1][:11]
            
        if not video_id:
            st.error("유효한 유튜브 링크가 아닙니다.")
        else:
            with st.spinner('데이터를 수집하고 분석하는 중입니다...'):
                # 1. 영상 통계 가져오기
                stats = fetch_video_stats(API_KEY, video_id)
                
                col_video, col_stats = st.columns([1, 1])
                
                with col_video:
                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                
                with col_stats:
                    if stats:
                        title = stats['snippet']['title']
                        channel = stats['snippet']['channelTitle']
                        view_count = int(stats['statistics'].get('viewCount', 0))
                        like_count = int(stats['statistics'].get('likeCount', 0))
                        comment_count = int(stats['statistics'].get('commentCount', 0))
                        
                        st.subheader(title)
                        st.write(f"📺 **채널명:** {channel}")
                        
                        # 통계 메트릭 표시
                        m1, m2, m3 = st.columns(3)
                        m1.metric("👀 조회수", f"{view_count:,}회")
                        m2.metric("👍 좋아요", f"{like_count:,}개")
                        m3.metric("💬 총 댓글수", f"{comment_count:,}개")
                
                st.divider()
                
                # 2. 댓글 가져오기 및 분석
                df = fetch_youtube_comments(API_KEY, video_id, max_comments)
                
                if not df.empty:
                    # 데이터 전처리 (날짜 및 시간 변환)
                    df['published_at'] = pd.to_datetime(df['published_at'])
                    df['date'] = df['published_at'].dt.date
                    df['hour'] = df['published_at'].dt.hour
                    df['sentiment'] = df['text'].apply(analyze_sentiment)
                    
                    st.subheader(f"🔍 최근 댓글 {len(df)}개 분석 결과")
                    
                    col1, col2 = st.columns(2)
                    
                    # 3. 시간대별 댓글 작성 추이 (0시~23시)
                    with col1:
                        st.markdown("#### 🕒 시간대별 댓글 작성 추이")
                        trend_data = df.groupby('hour').size().reindex(range(24), fill_value=0)
                        
                        fig_trend, ax_trend = plt.subplots(figsize=(8, 4))
                        ax_trend.plot(trend_data.index, trend_data.values, marker='o', color='#FF4B4B')
                        ax_trend.set_xticks(range(0, 24, 2))
                        ax_trend.set_xlabel("시간 (Hour)")
                        ax_trend.set_ylabel("댓글 수")
                        ax_trend.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig_trend)
                        
                    # 4. 댓글 반응도 (감성 분석)
                    with col2:
                        st.markdown("#### 😊 댓글 선호도 (반응도)")
                        sentiment_counts = df['sentiment'].value_counts()
                        
                        fig_pie, ax_pie = plt.subplots(figsize=(8, 4))
                        # 색상 매칭 (긍정:파랑, 중립:회색, 부정:빨강)
                        colors = ['#66b3ff' if '긍정' in x else '#ff9999' if '부정' in x else '#dddddd' for x in sentiment_counts.index]
                        ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                        st.pyplot(fig_pie)
                        
                    st.divider()
                    
                    # 5. 한글 워드클라우드
                    st.markdown("#### ☁️ 댓글 핵심 키워드 워드클라우드")
                    all_text = ' '.join(df['text'].tolist())
                    clean_text = clean_text_for_wordcloud(all_text)
                    
                    if len(clean_text.strip()) > 0:
                        wordcloud = WordCloud(
                            font_path=font_path,
                            width=1000, 
                            height=400, 
                            background_color='white',
                            colormap='viridis',
                            max_words=150
                        ).generate(clean_text)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.info("워드클라우드를 생성할 의미 있는 한글 데이터가 부족합니다.")
                        
                    # 원본 데이터 확인
                    with st.expander("원본 댓글 데이터 확인하기"):
                        st.dataframe(df[['author', 'text', 'published_at', 'sentiment', 'like_count']].sort_values(by='like_count', ascending=False))
                else:
                    st.warning("분석할 수 있는 댓글이 없습니다.")
