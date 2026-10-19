import re
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# [클라우드 서버용] 가상 브라우저 자동 설치 (최초 1회 실행)
@st.cache_resource
def ensure_playwright_browser():
    subprocess.run(["playwright", "install", "chromium"], check=False)

ensure_playwright_browser()

# ---------------------------------------------------------
# 1. 데이터 수집 모듈 (당근, 중고나라 전용)
# ---------------------------------------------------------
EXCLUDE_KEYWORDS = ["케이스", "박스만", "빈박스", "필름", "스트랩", "부품용", "파손", "매입", "삽니다", "교환", "구합니다"]

def is_valid_item(title: str, price: int, min_price_limit: int = 10000) -> bool:
    """액세서리, 부품, 구매글 등 미끼 매물 제외 필터"""
    if price < min_price_limit:
        return False
    for kw in EXCLUDE_KEYWORDS:
        if kw in title:
            return False
    return True

def fetch_platform_data(keyword: str, min_price_limit: int = 10000):
    """당근과 중고나라에서 실시간 매물 수집"""
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # 1) 당근마켓 수집
        try:
            page.goto(f"https://www.daangn.com/kr/buy-sell/?search={keyword}", timeout=10000)
            page.wait_for_timeout(1500)
            soup = BeautifulSoup(page.content(), "html.parser")
            
            for a_tag in soup.select("a[href*='/kr/buy-sell/']")[:25]:
                text_blocks = [t.strip() for t in a_tag.stripped_strings if t.strip()]
                href = a_tag.get("href", "")
                link = f"https://www.daangn.com{href}" if href.startswith("/") else href
                
                if len(text_blocks) >= 2:
                    title = text_blocks[0]
                    for t in text_blocks[1:]:
                        if "원" in t or "," in t:
                            price_num = re.sub(r"[^\d]", "", t)
                            if price_num:
                                price = int(price_num)
                                if is_valid_item(title, price, min_price_limit):
                                    results.append({"플랫폼": "당근", "제목": title, "가격": price, "링크": link})
                                break
        except Exception:
            pass

        # 2) 중고나라 수집
        try:
            page.goto(f"https://web.joongna.com/search/{keyword}", timeout=10000)
            page.wait_for_timeout(1500)
            soup = BeautifulSoup(page.content(), "html.parser")
            
            for card in soup.select("a[href*='/product/']")[:25]:
                strings = [s.strip() for s in card.stripped_strings if s.strip()]
                href = card.get("href", "")
                link = f"https://web.joongna.com{href}" if href.startswith("/") else href
                
                if len(strings) >= 2:
                    title = strings[0]
                    for s in strings[1:]:
                        if "원" in s:
                            price_num = re.sub(r"[^\d]", "", s)
                            if price_num:
                                price = int(price_num)
                                if is_valid_item(title, price, min_price_limit):
                                    results.append({"플랫폼": "중고나라", "제목": title, "가격": price, "링크": link})
                                break
        except Exception:
            pass

        browser.close()
    return pd.DataFrame(results)

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """사분위수(IQR) 기준 극단적인 이상치 가격 자동 제거"""
    if df.empty or len(df) < 4:
        return df
    cleaned_dfs = []
    for platform, group in df.groupby("플랫폼"):
        if len(group) < 4:
            cleaned_dfs.append(group)
            continue
        q25 = np.percentile(group["가격"], 25)
        q75 = np.percentile(group["가격"], 75)
        iqr = q75 - q25
        lower = max(0, q25 - 1.5 * iqr)
        upper = q75 + 1.5 * iqr
        cleaned_dfs.append(group[(group["가격"] >= lower) & (group["가격"] <= upper)])
    return pd.concat(cleaned_dfs, ignore_index=True)

# ---------------------------------------------------------
# 2. UI 화면 구성 (당근 vs 중고나라 집중 비교)
# ---------------------------------------------------------
st.set_page_config(page_title="당근·중고나라 시세 조회기", layout="wide")

st.title("🥕 당근 · 중고나라 중고 시세 통합 조회기")
st.markdown("모델명을 검색하면 **당근**과 **중고나라**의 **최저가, 평균가, 최고가**를 한눈에 비교합니다.")

with st.sidebar:
    st.header("⚙️ 검색 설정")
    mode = st.radio("데이터 수집 모드", ["샘플 데이터 시뮬레이션 (빠른 확인)", "실시간 웹 크롤링"])
    min_price = st.number_input("최소 가격 필터 (액세서리/미끼 제외용)", value=30000, step=10000)

keyword = st.text_input("🔍 검색할 모델명을 입력하세요", placeholder="예: 아이폰 15 프로 256, 갤럭시탭 S9, 에어팟 프로 2")

if st.button("시세 조회하기", type="primary") and keyword:
    with st.spinner(f"'{keyword}'의 당근·중고나라 시세를 분석 중입니다..."):
        if mode == "실시간 웹 크롤링":
            raw_df = fetch_platform_data(keyword, min_price)
        else:
            np.random.seed(len(keyword))
            base = 750000
            sample_data = (
                [
                    {
                        "플랫폼": "당근",
                        "제목": f"{keyword} 직거래 풀박스 S급 매물 {i+1}",
                        "가격": int(np.random.normal(base * 0.96, 40000, 1)[0] // 1000 * 1000),
                        "링크": f"https://www.daangn.com/kr/buy-sell/?search={keyword}"
                    }
                    for i in range(15)
                ] +
                [
                    {
                        "플랫폼": "중고나라",
                        "제목": f"{keyword} 상태좋음 안전결제 택배가능 {i+1}",
                        "가격": int(np.random.normal(base * 1.0, 45000, 1)[0] // 1000 * 1000),
                        "링크": f"https://web.joongna.com/search/{keyword}"
                    }
                    for i in range(15)
                ]
            )
            raw_df = pd.DataFrame(sample_data)

        df = remove_outliers(raw_df)

    if df.empty:
        st.warning("실시간 매물을 가져오지 못했습니다. 왼쪽 메뉴에서 '샘플 데이터 시뮬레이션'을 선택하거나 최소 가격 필터를 낮춰보세요.")
    else:
        # 1) 전체 통합 시세 (최하, 평균, 최대)
        total_min = int(df["가격"].min())
        total_avg = int(df["가격"].mean())
        total_med = int(df["가격"].median())
        total_max = int(df["가격"].max())

        st.subheader(f"📌 '{keyword}' 전체 통합 시세 (총 {len(df)}건 분석)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 최저가 (최하)", f"{total_min:,}원")
        col2.metric("🔵 평균가 (평균)", f"{total_avg:,}원")
        col3.metric("🟣 중앙값", f"{total_med:,}원")
        col4.metric("🔴 최고가 (최대)", f"{total_max:,}원")

        st.divider()

        # 2) 당근 vs 중고나라 플랫폼별 비교표 및 그래프
        st.subheader("⚖️ 당근 vs 중고나라 시세 비교")
        
        summary_df = (
            df.groupby("플랫폼")["가격"]
            .agg(매물수="count", 최저가="min", 평균가="mean", 중앙값="median", 최고가="max")
            .reset_index()
        )
        for col in ["최저가", "평균가", "중앙값", "최고가"]:
            summary_df[col] = summary_df[col].astype(int)

        left_col, right_col = st.columns([1.2, 1])
        with left_col:
            styled_df = summary_df.style.format({
                "최저가": "{:,}원", "평균가": "{:,}원", "중앙값": "{:,}원", "최고가": "{:,}원"
            })
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        with right_col:
            chart_data = summary_df.set_index("플랫폼")[["최저가", "평균가", "최고가"]]
            st.bar_chart(chart_data)

        st.divider()

        # 3) 수집된 매물 상세 내역 (링크 포함)
        st.subheader("📋 매물 상세 목록 (가격 낮은 순)")
        sorted_df = df.sort_values(by="가격", ascending=True).reset_index(drop=True)
        st.dataframe(
            sorted_df,
            column_config={
                "가격": st.column_config.NumberColumn("가격", format="%d원"),
                "링크": st.column_config.LinkColumn("매물 보러가기", display_text="바로가기 🔗")
            },
            use_container_width=True,
            hide_index=True
        )