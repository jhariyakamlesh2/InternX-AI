import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import time

# ──────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
#  CUSTOM CSS — Dark financial terminal look
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #070d12;
    color: #c8d8e8;
}
.stApp { background: #070d12; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1821; }
::-webkit-scrollbar-thumb { background: #00e5a0; border-radius: 2px; }

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d1f2d 0%, #0a1a1f 50%, #071215 100%);
    border: 1px solid #00e5a033;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, #00e5a015 0%, transparent 70%);
    border-radius: 50%;
}
.header-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 5%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, #00aaff10 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e5a0, #00aaff, #00e5a0);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    margin: 0;
    line-height: 1.1;
}
.header-subtitle {
    font-family: 'Space Mono', monospace;
    color: #4a7a8a;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1520 !important;
    border-right: 1px solid #00e5a020;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00e5a0;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: #0d1f2d !important;
    border: 1px solid #1a3a4a !important;
    border-radius: 8px !important;
    color: #c8d8e8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 2px #00e5a020 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00e5a0, #00aaff) !important;
    color: #070d12 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px #00e5a030 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #00e5a050 !important;
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #0d1f2d, #0a1820);
    border: 1px solid #1a3a4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
.card:hover {
    border-color: #00e5a040;
    transform: translateY(-2px);
}
.card-accent {
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(180deg, #00e5a0, #00aaff);
    border-radius: 12px 0 0 12px;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00e5a0;
    margin-bottom: 0.75rem;
}
.card-content {
    color: #c8d8e8;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* ── Metric Boxes ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-box {
    background: #0d1f2d;
    border: 1px solid #1a3a4a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-box:hover { border-color: #00e5a050; }
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7a8a;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #00e5a0;
}
.metric-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a7a8a;
}

/* ── Chain Steps ── */
.chain-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.75rem;
    padding: 0.75rem 1rem;
    background: #0a1520;
    border-radius: 8px;
    border-left: 3px solid #00e5a040;
    animation: fadeInLeft 0.5s ease forwards;
    opacity: 0;
}
.chain-step.active { border-left-color: #00e5a0; }
@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}
.step-num {
    background: #00e5a020;
    color: #00e5a0;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-text {
    font-size: 0.88rem;
    color: #8aaabb;
    font-family: 'Space Mono', monospace;
}

/* ── Tag Pills ── */
.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.pill-green { background: #00e5a015; color: #00e5a0; border: 1px solid #00e5a040; }
.pill-blue  { background: #00aaff15; color: #00aaff; border: 1px solid #00aaff40; }
.pill-red   { background: #ff445515; color: #ff4455; border: 1px solid #ff445540; }
.pill-gold  { background: #ffcc0015; color: #ffcc00; border: 1px solid #ffcc0040; }

/* ── Divider ── */
.neon-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00e5a040, transparent);
    margin: 1.5rem 0;
}

/* ── Ticker Tape ── */
.ticker-wrap {
    background: #0a1520;
    border-top: 1px solid #00e5a020;
    border-bottom: 1px solid #00e5a020;
    overflow: hidden;
    padding: 0.5rem 0;
    margin-bottom: 2rem;
}
.ticker {
    display: inline-flex;
    animation: tick 20s linear infinite;
    white-space: nowrap;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #4a7a8a;
}
.ticker span { margin: 0 2rem; }
.ticker .up { color: #00e5a0; }
.ticker .dn { color: #ff4455; }
@keyframes tick { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

/* ── Output section ── */
.output-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00aaff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.output-header::before {
    content: '';
    width: 8px; height: 8px;
    background: #00aaff;
    border-radius: 50%;
    animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Selectbox label ── */
label { color: #8aaabb !important; font-size: 0.85rem !important; }
.stSelectbox label { font-family: 'Space Mono', monospace !important; }

/* ── Info/Warning overrides ── */
.stAlert { background: #0d1f2d !important; border: 1px solid #1a3a4a !important; color: #c8d8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1 class="header-title">📈 StockSense AI</h1>
    <p class="header-subtitle">⬡ LangChain · Prompt Engineering · Sequential Chain Analysis · Real-time Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Ticker tape
st.markdown("""
<div class="ticker-wrap">
  <div class="ticker">
    <span>NIFTY50 <span class="up">▲ +0.84%</span></span>
    <span>SENSEX <span class="up">▲ +0.91%</span></span>
    <span>RELIANCE <span class="up">▲ +1.2%</span></span>
    <span>TCS <span class="dn">▼ -0.3%</span></span>
    <span>INFY <span class="up">▲ +0.7%</span></span>
    <span>HDFC BANK <span class="dn">▼ -0.5%</span></span>
    <span>ITC <span class="up">▲ +2.1%</span></span>
    <span>WIPRO <span class="up">▲ +0.4%</span></span>
    <span>BAJFINANCE <span class="dn">▼ -1.1%</span></span>
    <span>ONGC <span class="up">▲ +0.6%</span></span>
    <!-- duplicate for seamless loop -->
    <span>NIFTY50 <span class="up">▲ +0.84%</span></span>
    <span>SENSEX <span class="up">▲ +0.91%</span></span>
    <span>RELIANCE <span class="up">▲ +1.2%</span></span>
    <span>TCS <span class="dn">▼ -0.3%</span></span>
    <span>INFY <span class="up">▲ +0.7%</span></span>
    <span>HDFC BANK <span class="dn">▼ -0.5%</span></span>
    <span>ITC <span class="up">▲ +2.1%</span></span>
    <span>WIPRO <span class="up">▲ +0.4%</span></span>
    <span>BAJFINANCE <span class="dn">▼ -1.1%</span></span>
    <span>ONGC <span class="up">▲ +0.6%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<hr style='border-color:#1a3a4a'>", unsafe_allow_html=True)

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key. Never shared or stored."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Analysis Mode")

    analysis_depth = st.selectbox(
        "Depth",
        ["Quick Scan", "Standard Analysis", "Deep Dive"],
        index=1
    )

    analysis_style = st.selectbox(
        "Investment Style",
        ["Growth Investor", "Value Investor", "Swing Trader", "Long-term Investor"],
        index=0
    )

    risk_profile = st.selectbox(
        "Risk Profile",
        ["Conservative 🛡️", "Moderate ⚖️", "Aggressive 🔥"],
        index=1
    )

    st.markdown("<hr style='border-color:#1a3a4a'>", unsafe_allow_html=True)
    st.markdown("### 📡 LangChain Pipeline")

    st.markdown("""
    <div class="chain-step active" style="animation-delay:0s">
      <div class="step-num">1</div>
      <div class="step-text">PromptTemplate → Company Overview</div>
    </div>
    <div class="chain-step active" style="animation-delay:0.15s">
      <div class="step-num">2</div>
      <div class="step-text">LLMChain → Financial Metrics</div>
    </div>
    <div class="chain-step active" style="animation-delay:0.3s">
      <div class="step-num">3</div>
      <div class="step-text">Chain → Risk Assessment</div>
    </div>
    <div class="chain-step active" style="animation-delay:0.45s">
      <div class="step-num">4</div>
      <div class="step-text">SequentialChain → Final Report</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a3a4a'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#2a4a5a; line-height:1.8;">
    ⚡ Powered by LangChain<br>
    🤖 GPT-4o-mini backend<br>
    📊 Sequential Chain pipeline<br>
    ⚠️ Not financial advice
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────
#  MAIN LAYOUT
# ──────────────────────────────────────────
col_input, col_output = st.columns([1, 1.6], gap="large")

with col_input:
    st.markdown("### 🔍 Stock Input")

    stock_name = st.text_input(
        "Stock / Company Name",
        placeholder="e.g. Reliance Industries, Apple, TCS...",
        help="Enter the full company name for best results"
    )

    sector = st.selectbox(
        "Sector",
        [
            "Technology", "Banking & Finance", "Energy & Oil",
            "Pharma & Healthcare", "FMCG", "Auto & EV",
            "Real Estate", "Metals & Mining", "Telecom", "Other"
        ]
    )

    timeframe = st.selectbox(
        "Investment Horizon",
        ["Short-term (1-3 months)", "Medium-term (6-12 months)", "Long-term (2-5 years)"]
    )

    extra_context = st.text_area(
        "Additional Context (optional)",
        placeholder="e.g. Recent news, earnings data, your concerns...",
        height=100
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ Run AI Analysis Chain")

    # ── Prompt Preview ──
    st.markdown("<hr style='border-color:#1a3a4a;margin-top:1.5rem'>", unsafe_allow_html=True)
    st.markdown("### 🧩 Prompt Templates")

    with st.expander("📌 Chain 1 — Company Overview Prompt"):
        st.code("""
PromptTemplate(
  input_variables=["stock", "sector"],
  template=\"\"\"
You are a senior equity research analyst.
Analyze {stock} in the {sector} sector.

Provide:
1. Business model overview
2. Key revenue drivers
3. Competitive moat
4. Market position
Keep response concise and data-focused.
\"\"\"
)""", language="python")

    with st.expander("📌 Chain 2 — Risk Assessment Prompt"):
        st.code("""
PromptTemplate(
  input_variables=["overview","risk_profile"],
  template=\"\"\"
Based on this analysis:
{overview}

Assess risks for a {risk_profile} investor:
1. Business risks
2. Market risks  
3. Macro risks
4. Risk score (1-10)
\"\"\"
)""", language="python")

    with st.expander("📌 Chain 3 — Final Recommendation Prompt"):
        st.code("""
PromptTemplate(
  input_variables=["overview","risks",
                   "style","timeframe"],
  template=\"\"\"
Company Analysis: {overview}
Risk Assessment: {risks}

As a {style} with {timeframe} horizon:
- BUY / HOLD / SELL verdict
- Target price range
- Key catalysts to watch
- Portfolio allocation %
\"\"\"
)""", language="python")

# ──────────────────────────────────────────
#  ANALYSIS ENGINE + OUTPUT
# ──────────────────────────────────────────
with col_output:
    st.markdown("### 📊 Analysis Output")

    if analyze_btn:
        if not api_key:
            st.error("🔑 Please enter your OpenAI API Key in the sidebar.")
        elif not stock_name:
            st.warning("📌 Please enter a stock name to analyze.")
        else:
            try:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model="gpt-4o-mini",
                    temperature=0.4
                )

                # ── PROMPT 1: Company Overview ──
                overview_prompt = PromptTemplate(
                    input_variables=["stock", "sector", "context"],
                    template="""You are a senior equity research analyst at a top investment bank.

Analyze {stock} in the {sector} sector.
Additional context: {context}

Provide a structured analysis with these sections:
1. **Business Model** - Core revenue streams and value proposition
2. **Competitive Advantages** - Moat, market share, differentiators
3. **Financial Health** - Key metrics, growth trajectory
4. **Recent Developments** - Major news, product launches, partnerships

Be specific, data-driven, and concise. Format with clear headers."""
                )

                # ── PROMPT 2: Risk Assessment ──
                risk_prompt = PromptTemplate(
                    input_variables=["overview", "risk_profile"],
                    template="""Based on this equity analysis:
{overview}

Conduct a comprehensive risk assessment for a {risk_profile} investor.

Include:
1. **Business Risks** - Operational, competitive, management risks
2. **Market Risks** - Volatility, liquidity, sector rotation
3. **Macro Risks** - Interest rates, regulatory, geopolitical
4. **Overall Risk Score** - Rate 1-10 (1=minimal risk, 10=extreme risk) with justification

Conclude with a one-sentence risk summary."""
                )

                # ── PROMPT 3: Final Recommendation ──
                recommendation_prompt = PromptTemplate(
                    input_variables=["overview", "risks", "style", "timeframe"],
                    template="""You have:

COMPANY ANALYSIS:
{overview}

RISK ASSESSMENT:
{risks}

Generate a final investment recommendation for a {style} with a {timeframe} horizon.

Structure:
1. **VERDICT** - Clearly state BUY / HOLD / SELL with conviction level (High/Medium/Low)
2. **Target Price Range** - Conservative, base, and bull case
3. **Entry Strategy** - When/how to enter
4. **Key Catalysts** - Top 3 events that could move the stock
5. **Stop Loss / Exit** - Clear risk management levels
6. **Portfolio Allocation** - Recommended % weight

End with a 2-sentence executive summary."""
                )

                # ── CHAIN EXECUTION WITH PROGRESS ──
                progress_placeholder = st.empty()
                
                with progress_placeholder.container():
                    st.markdown("""
                    <div class="output-header">CHAIN EXECUTING</div>
                    """, unsafe_allow_html=True)

                    prog = st.progress(0)
                    status = st.empty()

                    # Step 1
                    status.markdown("""
                    <div class="chain-step active">
                      <div class="step-num">1</div>
                      <div class="step-text">Running Company Overview Chain...</div>
                    </div>""", unsafe_allow_html=True)
                    prog.progress(15)

                    chain1 = LLMChain(llm=llm, prompt=overview_prompt, output_key="overview")
                    result1 = chain1.invoke({
                        "stock": stock_name,
                        "sector": sector,
                        "context": extra_context or "No additional context provided."
                    })
                    overview_text = result1["overview"]
                    prog.progress(40)
                    time.sleep(0.3)

                    # Step 2
                    status.markdown("""
                    <div class="chain-step active">
                      <div class="step-num">2</div>
                      <div class="step-text">Running Risk Assessment Chain...</div>
                    </div>""", unsafe_allow_html=True)
                    
                    chain2 = LLMChain(llm=llm, prompt=risk_prompt, output_key="risks")
                    result2 = chain2.invoke({
                        "overview": overview_text,
                        "risk_profile": risk_profile
                    })
                    risk_text = result2["risks"]
                    prog.progress(70)
                    time.sleep(0.3)

                    # Step 3
                    status.markdown("""
                    <div class="chain-step active">
                      <div class="step-num">3</div>
                      <div class="step-text">Running Final Recommendation Chain...</div>
                    </div>""", unsafe_allow_html=True)
                    
                    chain3 = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")
                    result3 = chain3.invoke({
                        "overview": overview_text,
                        "risks": risk_text,
                        "style": analysis_style,
                        "timeframe": timeframe
                    })
                    rec_text = result3["recommendation"]
                    prog.progress(100)
                    time.sleep(0.3)

                progress_placeholder.empty()

                # ── DISPLAY RESULTS ──
                st.markdown(f"""
                <div style="margin-bottom:1rem">
                    <span class="pill pill-blue">📊 {sector}</span>
                    <span class="pill pill-green">🎯 {analysis_style}</span>
                    <span class="pill pill-gold">⏱ {timeframe.split('(')[0].strip()}</span>
                    <span class="pill pill-red">⚠️ {risk_profile}</span>
                </div>
                """, unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(["🏢 Overview", "⚠️ Risk Analysis", "💡 Recommendation"])

                with tab1:
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-accent"></div>
                      <div class="card-title">🏢 Company Deep-Dive — {stock_name}</div>
                      <div class="card-content">{overview_text.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with tab2:
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-accent" style="background:linear-gradient(180deg,#ff4455,#ffcc00)"></div>
                      <div class="card-title">⚠️ Risk Assessment — {risk_profile}</div>
                      <div class="card-content">{risk_text.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with tab3:
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-accent" style="background:linear-gradient(180deg,#00aaff,#00e5a0)"></div>
                      <div class="card-title">💡 Investment Recommendation — {analysis_style}</div>
                      <div class="card-content">{rec_text.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── LangChain Code Insight ──
                st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
                st.markdown("### 🔗 LangChain Pipeline Used")
                st.code(f"""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# Chain 1 — Company Overview
chain1 = LLMChain(llm=llm, prompt=overview_prompt, output_key="overview")
result1 = chain1.invoke({{"stock": "{stock_name}", "sector": "{sector}"}})

# Chain 2 — Risk Assessment  
chain2 = LLMChain(llm=llm, prompt=risk_prompt, output_key="risks")
result2 = chain2.invoke({{"overview": result1["overview"], "risk_profile": "{risk_profile}"}})

# Chain 3 — Final Recommendation
chain3 = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")
result3 = chain3.invoke({{
    "overview": result1["overview"],
    "risks": result2["risks"],
    "style": "{analysis_style}",
    "timeframe": "{timeframe}"
}})
""", language="python")

                st.success(f"✅ Sequential Chain complete! 3 prompts → 3 chains → 1 full analysis for **{stock_name}**")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your OpenAI API key is valid and has sufficient credits.")

    else:
        # Placeholder when no analysis run yet
        st.markdown("""
        <div class="card" style="border-style:dashed; text-align:center; padding:3rem 2rem">
            <div class="card-accent"></div>
            <div style="font-size:3rem; margin-bottom:1rem">📡</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.15em; color:#2a4a5a; text-transform:uppercase; margin-bottom:0.5rem">Awaiting Input</div>
            <div style="color:#4a7a8a; font-size:0.9rem; line-height:1.6">
                Enter a stock name, configure your settings,<br>
                and click <strong style="color:#00e5a0">Run AI Analysis Chain</strong><br>
                to start the LangChain pipeline.
            </div>
        </div>

        <div class="card" style="margin-top:1rem">
            <div class="card-accent" style="background:linear-gradient(180deg,#00aaff,#00e5a0)"></div>
            <div class="card-title">🔗 Sequential Chain Architecture</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.78rem; color:#4a7a8a; line-height:2">
                Input<br>
                &nbsp;&nbsp;↓<br>
                <span style="color:#00e5a0">LLMChain #1</span> → Company Overview<br>
                &nbsp;&nbsp;↓<br>
                <span style="color:#00aaff">LLMChain #2</span> → Risk Assessment<br>
                &nbsp;&nbsp;↓<br>
                <span style="color:#ffcc00">LLMChain #3</span> → Final Recommendation<br>
                &nbsp;&nbsp;↓<br>
                Structured Report ✅
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1.5rem; border-top:1px solid #1a3a4a; margin-top:2rem">
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#2a4a5a; letter-spacing:0.15em; text-transform:uppercase">
        ⚠️ StockSense AI is for educational purposes only. Not financial advice. Always consult a SEBI-registered advisor.
    </div>
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#1a3a4a; margin-top:0.5rem">
        Built with LangChain · Streamlit · OpenAI GPT-4o-mini
    </div>
</div>
""", unsafe_allow_html=True)
