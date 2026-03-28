import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import ollama
import requests
from bs4 import BeautifulSoup

# --- [1. KOSPI 100 실시간 크롤링] ---
@st.cache_data(ttl=3600)
def get_kospi_100():
    items = []
    for page in [1, 2]:
        url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}'
        res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
        res.encoding = 'euc-kr'
        soup = BeautifulSoup(res.text, 'html.parser')
        for link in soup.select('a.tltle'):
            items.append({'name': link.text, 'code': f"{link['href'].split('=')[-1]}.KS"})
    return items[:100]

# --- [2. 퀀타멘탈 (가치 + 기술) 평가 엔진] ---
def calculate_quantamental_kpi(ticker_code):
    try:
        stock = yf.Ticker(ticker_code)
        df = stock.history(period="6mo", interval="1d")
        if len(df) < 35: return None
        
        # 재무 데이터(info) 호출 (시간이 조금 걸릴 수 있음)
        info = stock.info
        close = df['Close'].ffill()

        # ---------------------------------------------------------
        # [A] 펀더멘털 평가 (가중치 60%) - 사용자 요구사항 반영
        # ---------------------------------------------------------
        # 1. 가격결정권 (Operating Margin): 영업이익률 15% 이상 만점
        op_margin = info.get('operatingMargins', 0)
        margin_score = np.clip((op_margin / 0.15) * 100, 0, 100) if op_margin else 0
        
        # 2. 기술적 해자 (ROE): 자기자본이익률 15% 이상 만점
        roe = info.get('returnOnEquity', 0)
        roe_score = np.clip((roe / 0.15) * 100, 0, 100) if roe else 0
        
        # 3. 수주잔고/성장성 프록시 (Revenue Growth): 매출성장률 10% 이상 만점
        rev_growth = info.get('revenueGrowth', 0)
        growth_score = np.clip((rev_growth / 0.10) * 100, 0, 100) if rev_growth else 0
        
        fund_score = (margin_score * 0.4) + (roe_score * 0.4) + (growth_score * 0.2)

        # ---------------------------------------------------------
        # [B] 기술적 타이밍 평가 (가중치 40%) - 모멘텀 트랩 방지
        # ---------------------------------------------------------
        # 1. MACD (추세 방향성)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        macd_score = 100 if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1] else 0
        
        # 2. RSI 정상화 (40~60 사이 안정적일 때 고득점, 70 이상 과열 시 감점)
        rsi = ta.rsi(close, length=14).iloc[-1]
        if rsi >= 70:
            rsi_score = 10 # 추격 매수 금지
        elif rsi <= 30:
            rsi_score = 80 # 낙폭 과대 반등
        else:
            rsi_score = 100 - abs(50 - rsi) * 2
            
        tech_score = (macd_score * 0.5) + (rsi_score * 0.5)

        # ---------------------------------------------------------
        # [C] 리스크 관리 및 최종 연산
        # ---------------------------------------------------------
        atr_pct = (ta.atr(df['High'], df['Low'], close, length=14).iloc[-1] / close.iloc[-1]) * 100
        total_kpi = (fund_score * 0.6) + (tech_score * 0.4)
        
        return {
            "price": close.iloc[-1], "kpi": total_kpi,
            "fund_score": fund_score, "tech_score": tech_score,
            "op_margin": op_margin * 100 if op_margin else 0,
            "roe": roe * 100 if roe else 0,
            "rsi": rsi, "v_pct": atr_pct
        }
    except Exception as e:
        return None

# --- [3. 메인 웹 UI] ---
st.set_page_config(page_title="Quantamental AI", layout="wide")
st.title("🦅 KOSPI 100 퀀타멘탈 AI 리포트 (가치+차트 통합)")

with st.expander("📚 평가 로직 설명 (Fund 60% + Tech 40%)", expanded=False):
    st.markdown("""
    * **펀더멘털 (기업가치)**: 가격결정권(영업이익률), 기술적 해자(ROE), 성장성(매출증가율)을 수치화합니다.
    * **테크니컬 (타이밍)**: MACD 추세와 RSI를 봅니다. 단, RSI가 70이 넘는 과열 종목은 점수를 대폭 깎습니다.
    * **추천 수량**: 변동성이 높은 종목은 리스크 관리를 위해 자동으로 매수 수량을 줄여 계산합니다.
    """)

with st.sidebar:
    st.header("⚙️ 운용 파라미터")
    total_budget = st.number_input("총 투자 예산 (원)", value=100000000, step=10000000)
    target_count = st.slider("포트폴리오 편입 종목 수", 3, 20, 10)
    budget_per_stock = total_budget / target_count
    model_name = st.selectbox("AI 분석 엔진", ["gemma2:2b", "llama3-ko-instruct", "gemma"], index=0)
    st.caption("주의: 100개 종목의 재무제표를 불러오므로 약 1~2분이 소요됩니다.")

# 데이터 수집 및 연산
stocks = get_kospi_100()
results = []
progress_bar = st.progress(0, text="[1/3] KOSPI 100 재무 및 차트 데이터 교차 검증 중...")

for i, s in enumerate(stocks):
    res = calculate_quantamental_kpi(s['code'])
    if res and res['kpi'] > 0:
        # 매수 수량 산출 (자금 배분 로직)
        qty = int((budget_per_stock * (res['kpi']/100)) / (res['price'] * (1 + res['v_pct']/100)))
        results.append({
            "종목명": s['name'], "종합점수": res['kpi'], "현재가": res['price'],
            "가치점수": res['fund_score'], "차트점수": res['tech_score'],
            "영업이익률": res['op_margin'], "ROE": res['roe'], 
            "RSI": res['rsi'], "추천수량": qty
        })
    progress_bar.progress((i + 1) / 100)

df_final = pd.DataFrame(results).sort_values(by="종합점수", ascending=False)
top_data = df_final.head(target_count)

# --- [4. AI 자동 리포트 생성] ---
progress_bar.progress(0.9, text="[2/3] AI가 퀀타멘탈 분석 리포트를 작성 중입니다...")

try:
    report_data = top_data[["종목명", "종합점수", "가치점수", "차트점수", "영업이익률", "ROE"]].to_string(index=False)
    prompt = f"""
    당신은 세계 최고의 퀀타멘탈(Quantamental) 펀드 매니저입니다.
    아래는 KOSPI 시총 100위 내에서 '재무적 해자(가치점수)'와 '기술적 타이밍(차트점수)'이 모두 높은 상위 {target_count}개 종목입니다.
    
    데이터:
    {report_data}
    
    요구사항:
    1. 이 종목들이 왜 강력한 '가격결정권(영업이익률)'과 '기술적 해자(ROE)'를 가졌는지 데이터에 기반해 분석하세요.
    2. 단순 차트 분석을 넘어, 기업 가치 관점에서 가장 매력적인 원픽(Top Pick) 종목을 선정하고 그 이유를 한국어로 명확히 설명하세요.
    """
    ai_res = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
    
    st.success("### 📜 AI 퀀타멘탈 포트폴리오 전략")
    st.markdown(ai_res['message']['content'])
except Exception as e:
    st.warning("AI 리포트 생성 중 오류가 발생했습니다. 로컬 모델 구동 상태를 확인하세요.")
ㄴ
progress_bar.progress(1.0, text="[3/3] 분석 완료!")

# --- [5. 전체 데이터 시각화] ---
st.divider()
st.subheader(f"📊 KOSPI 100 퀀타멘탈 랭킹 (Top {target_count} 추천수량 포함)")
st.dataframe(
    df_final.style.background_gradient(subset=['종합점수', '가치점수'], cmap='RdYlGn')
    .format({
        "현재가": "{:,.0f}원", "종합점수": "{:.1f}점", "가치점수": "{:.1f}점", "차트점수": "{:.1f}점",
        "영업이익률": "{:.1f}%", "ROE": "{:.1f}%", "RSI": "{:.1f}", "추천수량": "{:,.0f}주"
    }),
    use_container_width=True
)