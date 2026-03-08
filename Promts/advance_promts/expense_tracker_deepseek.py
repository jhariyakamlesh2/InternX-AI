from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import datetime
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="Expense Tracker · DeepSeek AI", layout="wide")

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #060614; color: #e8e8f0; }

section[data-testid="stSidebar"] { background: #0d0d20; border-right: 1px solid #1e1e40; }
section[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -1px; color: #e8e8f0 !important; }
h2 { font-size: 1.3rem !important; font-weight: 700 !important; color: #4d9fff !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #e8e8f0 !important; }

[data-testid="metric-container"] {
    background: #0d0d20; border: 1px solid #1e1e40;
    border-radius: 12px; padding: 1rem 1.2rem !important; transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #4d9fff; }
[data-testid="metric-container"] label {
    color: #666 !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #4d9fff !important; font-family: 'DM Mono', monospace !important; font-size: 1.8rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a6fff, #0a4fcc) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.9rem !important; padding: 0.5rem 1.4rem !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a7fff, #1a5fdd) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(77,159,255,0.4) !important;
}

[data-testid="stFileUploader"] {
    background: #0d0d20 !important; border: 2px dashed #1e1e40 !important;
    border-radius: 12px !important; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #4d9fff !important; }

.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox select {
    background: #0d0d20 !important; border: 1px solid #1e1e40 !important;
    color: #e8e8f0 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #4d9fff !important; box-shadow: 0 0 0 2px rgba(77,159,255,0.15) !important;
}

[data-testid="stDataFrame"] { border: 1px solid #1e1e40 !important; border-radius: 12px !important; overflow: hidden; }

[data-testid="stForm"] {
    background: #0d0d20; border: 1px solid #1e1e40; border-radius: 16px; padding: 1.5rem;
}

hr { border-color: #1e1e40 !important; }
label { color: #888 !important; font-size: 0.82rem !important; letter-spacing: 0.5px; }

/* DeepSeek AI Summary Box */
.ai-summary-box {
    background: linear-gradient(135deg, #070720 0%, #0d0d28 100%);
    border: 1px solid #4d9fff;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.92rem;
    line-height: 1.9;
    color: #c8dcff;
    box-shadow: 0 0 40px rgba(77,159,255,0.08);
    white-space: pre-wrap;
}
.ai-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(90deg, #1a6fff, #0a4fcc);
    color: #fff;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 0.72rem; letter-spacing: 1.5px;
    padding: 4px 12px; border-radius: 20px;
    margin-bottom: 1rem; text-transform: uppercase;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0d20; }
::-webkit-scrollbar-thumb { background: #1e1e40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4d9fff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  DEEPSEEK — SUMMARY FUNCTION
# ─────────────────────────────────────────

# DeepSeek uses OpenAI-compatible API
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"   # or "deepseek-reasoner" for R1

SUMMARY_PROMPT_TEMPLATE = """You are a smart personal finance assistant. Analyze the expense data below and give a clear, friendly, structured summary.

Expense Data:
{data}

Key Stats:
- Total Spent: {total}
- Number of Transactions: {num_tx}
- Average per Transaction: {avg_tx}
- Top Spending Category / Recipient: {top_cat}

Provide your analysis with these exact sections:

## 📊 Overall Summary
Brief overview of spending pattern.

## 🏷 Top Spending Areas
Which categories or recipients took the most budget.

## 🔁 Spending Pattern
Notable habits — frequent small spends vs rare large ones, any anomalies.

## ⚠️ Key Concern
One important financial concern based on the data.

## 💡 Money-Saving Tip
One practical, specific suggestion to reduce expenses.

Keep the tone friendly, concise, and helpful. Use bullet points where appropriate."""


def generate_deepseek_summary(df: pd.DataFrame, api_key: str, model: str) -> str:
    """Call DeepSeek API using OpenAI-compatible client."""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # Build stats
    data_text = df.head(50).to_string(index=False)
    total     = f"{df['Amount Paid'].sum():,.2f}" if "Amount Paid" in df.columns else "N/A"
    avg_tx    = f"{df['Amount Paid'].mean():,.2f}" if "Amount Paid" in df.columns else "N/A"
    num_tx    = str(len(df))

    # Top recipient or category
    if "Paid To" in df.columns and not df.empty:
        top_cat = df.groupby("Paid To")["Amount Paid"].sum().idxmax()
    else:
        top_cat = "N/A"

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        data=data_text,
        total=total,
        num_tx=num_tx,
        avg_tx=avg_tx,
        top_cat=str(top_cat),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a smart, friendly personal finance assistant. "
                    "Analyze expense data and give clear, actionable insights. "
                    "Use markdown headers and bullet points for structure."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(
        columns=[
            "Date of Payment",
            "Method of Payment",
            "Paid To",
            "Description",
            "Amount Paid",
            "Running Total",
        ]
    )
if "ai_summary" not in st.session_state: st.session_state.ai_summary = ""
if "ai_error"   not in st.session_state: st.session_state.ai_error   = ""


# ─────────────────────────────────────────
#  SIDEBAR — API SETTINGS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 DeepSeek AI Settings")
    st.divider()

    env_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key_input = st.text_input(
        "DeepSeek API Key",
        value=env_key,
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="Get your key from https://platform.deepseek.com",
    )
    st.caption("🔒 Set `DEEPSEEK_API_KEY` in `.env` to auto-load")

    model_choice = st.selectbox(
        "Model",
        ["deepseek-chat", "deepseek-reasoner"],
        help="deepseek-chat = V3 (fast) | deepseek-reasoner = R1 (smarter, slower)",
    )

    st.divider()
    st.markdown(
        "<p style='color:#1e1e40;font-size:0.75rem;text-align:center'>ExpenseIQ · DeepSeek Edition</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("# 📊 Expense Tracker")
st.markdown(
    "<p style='color:#444;margin-top:-12px;font-size:0.9rem'>Powered by <b style='color:#4d9fff'>DeepSeek AI</b> · Track · Analyse · Summarise</p>",
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────
#  INPUT FORM  (same columns as original)
# ─────────────────────────────────────────
st.markdown("### ➕ Add New Expense")
with st.form("expense_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        date   = st.date_input("Date of Payment")
        method = st.text_input("Method of Payment", placeholder="UPI / Cash / Card")

    with col2:
        paid_to     = st.text_input("Paid To",      placeholder="Shop / Person / Service")
        description = st.text_input("Description",  placeholder="Brief note")

    with col3:
        amount = st.number_input("Amount Paid", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("➕ Add Expense")


# ─────────────────────────────────────────
#  ADD EXPENSE LOGIC
# ─────────────────────────────────────────
if submitted:
    if amount <= 0:
        st.warning("Amount must be greater than 0.")
    else:
        prev_total = (
            st.session_state.expenses["Running Total"].iloc[-1]
            if not st.session_state.expenses.empty else 0
        )
        running_total = prev_total + amount

        new_row = {
            "Date of Payment":   date,
            "Method of Payment": method,
            "Paid To":           paid_to,
            "Description":       description,
            "Amount Paid":       amount,
            "Running Total":     running_total,
        }
        st.session_state.expenses = pd.concat(
            [st.session_state.expenses, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        st.success(f"✅ Added ₹{amount:,.2f} — {description}")


# ─────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
total_to_date = (
    st.session_state.expenses["Amount Paid"].sum()
    if not st.session_state.expenses.empty else 0.0
)
num_records = len(st.session_state.expenses)
avg_spend   = (total_to_date / num_records) if num_records > 0 else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("💰 Total to Date",    f"₹{total_to_date:,.2f}")
m2.metric("🔢 Transactions",      str(num_records))
m3.metric("📊 Avg per Txn",      f"₹{avg_spend:,.2f}")


# ─────────────────────────────────────────
#  DATA TABLE
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📋 Expense Table")
if st.session_state.expenses.empty:
    st.info("No expenses yet. Add one above!")
else:
    st.dataframe(st.session_state.expenses, use_container_width=True, height=320)


# ─────────────────────────────────────────
#  🤖 DEEPSEEK AI SUMMARY
# ─────────────────────────────────────────
st.divider()
st.markdown("### 🤖 DeepSeek AI Expense Summary")
st.markdown(
    f"<p style='color:#444;font-size:0.83rem;margin-top:-10px'>"
    f"Powered by <b style='color:#4d9fff'>{model_choice}</b> · DeepSeek</p>",
    unsafe_allow_html=True,
)

btn_col, clear_col, _ = st.columns([2, 1.2, 5])

with btn_col:
    gen_clicked = st.button("✨ Generate AI Summary", use_container_width=True)

with clear_col:
    if st.session_state.ai_summary:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.ai_summary = ""
            st.session_state.ai_error   = ""
            st.rerun()

# ── Trigger DeepSeek ──
if gen_clicked:
    st.session_state.ai_summary = ""
    st.session_state.ai_error   = ""

    active_key = api_key_input.strip()

    if st.session_state.expenses.empty:
        st.warning("⚠️ No expense data to analyse. Add some expenses first!")
    elif not active_key:
        st.session_state.ai_error = (
            "**API Key missing!**\n\n"
            "Enter your DeepSeek API key in the sidebar, or add to `.env`:\n"
            "```\nDEEPSEEK_API_KEY=sk-xxxxxxxxxxxx\n```\n"
            "Get a key at: https://platform.deepseek.com"
        )
    else:
        with st.spinner(f"🧠 DeepSeek ({model_choice}) is analysing your expenses..."):
            try:
                result = generate_deepseek_summary(
                    st.session_state.expenses, active_key, model_choice
                )
                st.session_state.ai_summary = result
            except Exception as e:
                err = str(e)
                if "401" in err or "authentication" in err.lower() or "api key" in err.lower():
                    st.session_state.ai_error = (
                        "**Invalid API Key!**\n\n"
                        "Check your key at https://platform.deepseek.com\n"
                        "Make sure it starts with `sk-`"
                    )
                elif "429" in err or "rate limit" in err.lower():
                    st.session_state.ai_error = (
                        "**Rate limit reached.**\n\n"
                        "Wait a moment and try again."
                    )
                elif "connection" in err.lower() or "timeout" in err.lower():
                    st.session_state.ai_error = (
                        "**Connection failed.**\n\n"
                        "Check your internet and try again."
                    )
                else:
                    st.session_state.ai_error = f"**Error:** {err}"

# ── Show Result ──
if st.session_state.ai_summary:
    st.markdown(
        f'<div class="ai-summary-box">'
        f'<div><span class="ai-badge">🤖 DeepSeek · {model_choice}</span></div>'
        f'{st.session_state.ai_summary}'
        f'</div>',
        unsafe_allow_html=True,
    )

if st.session_state.ai_error:
    st.error(st.session_state.ai_error)
