import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import datetime

# ── LangChain + Ollama ──
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0d0d0d; color: #f0ede6; }

section[data-testid="stSidebar"] { background: #141414; border-right: 1px solid #2a2a2a; }
section[data-testid="stSidebar"] * { color: #f0ede6 !important; }

h1 { font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: -1px; color: #f0ede6 !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #c8ff00 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #f0ede6 !important; }

[data-testid="metric-container"] {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 12px; padding: 1rem 1.2rem !important; transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #c8ff00; }
[data-testid="metric-container"] label { color: #888 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #c8ff00 !important; font-family: 'DM Mono', monospace !important; font-size: 1.8rem !important; }

.stButton > button {
    background: #c8ff00 !important; color: #0d0d0d !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.9rem !important; padding: 0.5rem 1.4rem !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #b0e000 !important; transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(200,255,0,0.3) !important;
}

[data-testid="stFileUploader"] {
    background: #1a1a1a !important; border: 2px dashed #333 !important;
    border-radius: 12px !important; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #c8ff00 !important; }

.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox select {
    background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
    color: #f0ede6 !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #c8ff00 !important; box-shadow: 0 0 0 2px rgba(200,255,0,0.15) !important;
}

[data-testid="stDataFrame"] { border: 1px solid #2a2a2a !important; border-radius: 12px !important; overflow: hidden; }

.stTabs [data-baseweb="tab-list"] { background: #1a1a1a; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #888 !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; border-radius: 8px !important; }
.stTabs [aria-selected="true"] { background: #c8ff00 !important; color: #0d0d0d !important; }

[data-testid="stForm"] { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 16px; padding: 1.5rem; }

hr { border-color: #2a2a2a !important; }
label { color: #aaa !important; font-size: 0.82rem !important; letter-spacing: 0.5px; }

.ai-summary-box {
    background: linear-gradient(135deg, #0f1f00 0%, #1a1a1a 100%);
    border: 1px solid #c8ff00; border-radius: 16px;
    padding: 1.8rem 2rem; margin-top: 1rem;
    font-family: 'DM Mono', monospace; font-size: 0.92rem;
    line-height: 1.8; color: #e8f5c0;
    box-shadow: 0 0 30px rgba(200,255,0,0.08);
    white-space: pre-wrap;
}
.ai-badge {
    display: inline-block; background: #c8ff00; color: #0d0d0d;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 0.7rem; letter-spacing: 2px;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: 1rem; text-transform: uppercase;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c8ff00; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  LANGCHAIN + OLLAMA
# ─────────────────────────────────────────

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["expense_data", "total", "num_tx", "top_cat", "avg_tx"],
    template="""You are a smart personal finance assistant. Analyze the expense data below and give a clear, friendly summary.

Expense Data (sample):
{expense_data}

Key Stats:
- Total Spent: Rs {total}
- Transactions: {num_tx}
- Average per Transaction: Rs {avg_tx}
- Top Spending Category: {top_cat}

Provide your analysis in these sections:
1. Overall Summary - Brief spending overview
2. Top Categories - Which categories took most budget
3. Spending Pattern - Notable habits (small frequent vs large rare)
4. Key Concern - One thing to watch out for
5. Money-Saving Tip - One practical actionable suggestion

Keep it friendly, concise, and helpful. Use bullet points where needed.
"""
)


@st.cache_resource
def get_llm(model_name: str, base_url: str):
    return OllamaLLM(model=model_name, base_url=base_url, temperature=0.7)


def generate_ai_summary(df: pd.DataFrame, model_name: str, base_url: str) -> str:
    llm   = get_llm(model_name, base_url)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()

    sample  = df.head(40).to_string(index=False)
    total   = f"{df['Amount'].sum():,.2f}"  if "Amount"   in df.columns else "N/A"
    avg_tx  = f"{df['Amount'].mean():,.2f}" if "Amount"   in df.columns else "N/A"
    num_tx  = str(len(df))
    top_cat = (df.groupby("Category")["Amount"].sum().idxmax()
               if "Category" in df.columns and not df.empty else "N/A")

    return chain.invoke({
        "expense_data": sample,
        "total":        total,
        "num_tx":       num_tx,
        "top_cat":      str(top_cat),
        "avg_tx":       avg_tx,
    })


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

REQUIRED_COLS = ["Date", "Category", "Description", "Amount", "Payment Method"]


def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type."
        return df, None
    except Exception as e:
        return None, f"Error reading file: {e}"


def normalize_columns(df):
    col_map    = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    mappings   = {
        "date":           ["date", "date of payment", "payment date", "transaction date"],
        "category":       ["category", "type", "expense type", "expense category"],
        "description":    ["description", "desc", "details", "note", "notes", "narration"],
        "amount":         ["amount", "amount paid", "cost", "price", "value", "debit"],
        "payment method": ["payment method", "method", "mode", "method of payment", "paid via"],
    }
    for std, aliases in mappings.items():
        for alias in aliases:
            if alias in lower_cols:
                col_map[lower_cols[alias]] = std.title()
                break
    return df.rename(columns=col_map)


def compute_summary(df):
    total   = df["Amount"].sum()
    avg_tx  = df["Amount"].mean()
    max_tx  = df["Amount"].max()
    num_tx  = len(df)
    top_cat = df.groupby("Category")["Amount"].sum().idxmax() if "Category" in df.columns else "N/A"
    return total, avg_tx, max_tx, num_tx, top_cat


def to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Expenses")
    return buf.getvalue()


def color_amount(val):
    try:
        v = float(val)
        if v > 5000: return "color: #ff4d4d"
        if v > 1000: return "color: #ffb400"
        return "color: #c8ff00"
    except:
        return ""


# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
defaults = {
    "df":         pd.DataFrame(columns=REQUIRED_COLS),
    "upload_key": 0,
    "ai_summary": "",
    "ai_error":   "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💸 ExpenseIQ")
    st.markdown("<p style='color:#f44336;font-size:0.8rem;margin-top:-10px'>Smart Tracker + AI Summary</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📂 Upload File")
    uploaded = st.file_uploader(
        "CSV or Excel", type=["csv", "xlsx", "xls"],
        key=f"uploader_{st.session_state.upload_key}",
        label_visibility="collapsed",
    )
    if uploaded:
        df_raw, err = load_file(uploaded)
        if err:
            st.error(err)
        else:
            df_raw = normalize_columns(df_raw)
            if "Amount" in df_raw.columns:
                df_raw["Amount"] = pd.to_numeric(df_raw["Amount"], errors="coerce").fillna(0)
            if "Date" in df_raw.columns:
                df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
            st.session_state.df         = df_raw
            st.session_state.ai_summary = ""
            st.session_state.ai_error   = ""
            st.success(f"✅ Loaded {len(df_raw)} rows")

    st.divider()

    # ── Ollama Settings ──
    st.markdown("### 🤖 Ollama AI Settings")
    ollama_url = st.text_input(
        "Base URL", value="http://localhost:11434",
        help="Default local Ollama URL",
    )
    ollama_model = st.text_input(
        "Model", value="llama3",
        help="Model pulled in Ollama e.g. llama3, mistral, gemma2, phi3",
    )
    st.caption("💡 `ollama serve` + `ollama pull llama3`")

    st.divider()

    # ── Filters ──
    if not st.session_state.df.empty and "Category" in st.session_state.df.columns:
        st.markdown("### 🔍 Filters")
        cats    = ["All"] + sorted(st.session_state.df["Category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category", cats)
        date_range = None
        if "Date" in st.session_state.df.columns:
            dates = st.session_state.df["Date"].dropna()
            if not dates.empty:
                min_d, max_d = dates.min().date(), dates.max().date()
                date_range   = st.date_input("Date Range", value=(min_d, max_d))
    else:
        sel_cat    = "All"
        date_range = None

    st.divider()
    st.markdown("<p style='color:#333;font-size:0.73rem;text-align:center'>ExpenseIQ v2.0 · Ollama Edition</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  FILTER DATA
# ─────────────────────────────────────────
df = st.session_state.df.copy()

if not df.empty:
    if sel_cat != "All" and "Category" in df.columns:
        df = df[df["Category"] == sel_cat]
    if date_range and "Date" in df.columns and len(date_range) == 2:
        try:
            df = df[
                (df["Date"] >= pd.Timestamp(date_range[0])) &
                (df["Date"] <= pd.Timestamp(date_range[1]))
            ]
        except:
            pass


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 💸 Expense Tracker")
    st.markdown("<p style='color:#555;margin-top:-12px;font-size:0.9rem'>Upload · Analyse · AI Summary · Export</p>", unsafe_allow_html=True)
with col_h2:
    if not st.session_state.df.empty:
        st.download_button(
            "⬇ Export Excel",
            data=to_excel_bytes(st.session_state.df),
            file_name=f"expenses_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.divider()

# ─────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────
if not df.empty and "Amount" in df.columns:
    total, avg_tx, max_tx, num_tx, top_cat = compute_summary(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Total Spent",  f"₹{total:,.0f}")
    m2.metric("🔢 Transactions", f"{num_tx}")
    m3.metric("📊 Avg per Txn",  f"₹{avg_tx:,.0f}")
    m4.metric("🔺 Highest Txn",  f"₹{max_tx:,.0f}")
    m5.metric("🏷 Top Category", str(top_cat))
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")


# ─────────────────────────────────────────
#  🤖 AI SUMMARY SECTION
# ─────────────────────────────────────────
if not df.empty and "Amount" in df.columns:

    st.markdown("### 🤖 AI Expense Summary")
    st.markdown(f"<p style='color:#555;font-size:0.83rem;margin-top:-10px'>Powered by <b style='color:#c8ff00'>{ollama_model}</b> running locally via Ollama</p>", unsafe_allow_html=True)

    btn_col, clear_col, _ = st.columns([2, 1.5, 5])

    with btn_col:
        gen_clicked = st.button("✨ Generate AI Summary", use_container_width=True)

    with clear_col:
        if st.session_state.ai_summary:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.ai_summary = ""
                st.session_state.ai_error   = ""
                st.rerun()

    if gen_clicked:
        st.session_state.ai_summary = ""
        st.session_state.ai_error   = ""
        with st.spinner(f"🧠 Thinking with **{ollama_model}**... (this may take 10–30s)"):
            try:
                result = generate_ai_summary(df, ollama_model, ollama_url)
                st.session_state.ai_summary = result
            except Exception as e:
                err = str(e)
                if "Connection refused" in err or "connect" in err.lower():
                    st.session_state.ai_error = (
                        "**Ollama is not running.**\n\n"
                        "Open a terminal and run:\n"
                        "```\nollama serve\n```\n"
                        f"Then pull your model:\n```\nollama pull {ollama_model}\n```"
                    )
                elif "not found" in err.lower():
                    st.session_state.ai_error = (
                        f"**Model `{ollama_model}` not found.**\n\n"
                        f"Pull it first:\n```\nollama pull {ollama_model}\n```"
                    )
                else:
                    st.session_state.ai_error = f"**Error:** {err}"

    if st.session_state.ai_summary:
        st.markdown(
            f'<div class="ai-summary-box">'
            f'<div><span class="ai-badge">🤖 AI · {ollama_model}</span></div>'
            f'{st.session_state.ai_summary}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.ai_error:
        st.error(st.session_state.ai_error)

    st.divider()


# ─────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📈 Charts", "➕ Add Entry", "📐 Column Map"])

# ── TAB 1 : DATA ──
with tab1:
    if df.empty:
        st.markdown("<p style='color:#555;text-align:center;padding:3rem'>No data yet. Upload a file!</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"### Showing **{len(df)}** records")
        search = st.text_input("🔎 Search", placeholder="e.g. groceries, uber...")
        if search:
            mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
            df   = df[mask]

        if "Amount" in df.columns:
            styled = df.style.applymap(color_amount, subset=["Amount"]).format({"Amount": "₹{:,.2f}"})
            st.dataframe(styled, use_container_width=True, height=420)
        else:
            st.dataframe(df, use_container_width=True, height=420)

        st.markdown(f"<p style='color:#444;font-size:0.8rem'>{len(df)} rows shown</p>", unsafe_allow_html=True)
        st.download_button("⬇ Download filtered CSV", data=df.to_csv(index=False),
                           file_name="filtered_expenses.csv", mime="text/csv")

# ── TAB 2 : CHARTS ──
with tab2:
    if df.empty or "Amount" not in df.columns:
        st.markdown("<p style='color:#555;text-align:center;padding:3rem'>No data to chart.</p>", unsafe_allow_html=True)
    else:
        chart_bg = "#141414"
        text_col = "#f0ede6"

        c1, c2 = st.columns(2)

        with c1:
            if "Category" in df.columns:
                st.markdown("#### Spend by Category")
                cat_sum = df.groupby("Category")["Amount"].sum().reset_index()
                fig_pie = px.pie(cat_sum, names="Category", values="Amount", hole=0.55,
                                 color_discrete_sequence=px.colors.sequential.Greens_r)
                fig_pie.update_traces(textfont_color=text_col)
                fig_pie.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
                                      font_color=text_col, legend=dict(font=dict(color=text_col)),
                                      margin=dict(t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            if "Category" in df.columns:
                st.markdown("#### Category Bar Chart")
                fig_bar = px.bar(cat_sum.sort_values("Amount", ascending=False),
                                 x="Category", y="Amount", color="Amount",
                                 color_continuous_scale=["#1a1a1a", "#c8ff00"])
                fig_bar.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
                                      font_color=text_col, xaxis=dict(gridcolor="#2a2a2a"),
                                      yaxis=dict(gridcolor="#2a2a2a"), coloraxis_showscale=False,
                                      margin=dict(t=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        if "Date" in df.columns:
            st.markdown("#### Spend Over Time")
            df_time = (df.dropna(subset=["Date"]).sort_values("Date")
                         .groupby("Date")["Amount"].sum().reset_index())
            df_time["Cumulative"] = df_time["Amount"].cumsum()
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_time["Date"], y=df_time["Amount"], mode="lines+markers",
                name="Daily Spend", line=dict(color="#c8ff00", width=2),
                marker=dict(size=5, color="#c8ff00"),
                fill="tozeroy", fillcolor="rgba(200,255,0,0.07)"))
            fig_line.add_trace(go.Scatter(
                x=df_time["Date"], y=df_time["Cumulative"], mode="lines",
                name="Cumulative", line=dict(color="#ffb400", width=1.5, dash="dot")))
            fig_line.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
                font_color=text_col, xaxis=dict(gridcolor="#2a2a2a"),
                yaxis=dict(gridcolor="#2a2a2a"), legend=dict(font=dict(color=text_col)),
                margin=dict(t=10, b=10))
            st.plotly_chart(fig_line, use_container_width=True)

        if "Payment Method" in df.columns:
            st.markdown("#### Payment Method Split")
            pm_sum = df.groupby("Payment Method")["Amount"].sum().reset_index()
            fig_pm = px.bar(pm_sum.sort_values("Amount"), x="Amount", y="Payment Method",
                            orientation="h", color="Amount",
                            color_continuous_scale=["#1a1a1a", "#c8ff00"])
            fig_pm.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
                font_color=text_col, xaxis=dict(gridcolor="#2a2a2a"),
                yaxis=dict(gridcolor="#2a2a2a"), coloraxis_showscale=False,
                margin=dict(t=10, b=10))
            st.plotly_chart(fig_pm, use_container_width=True)

# ── TAB 3 : ADD ENTRY ──
with tab3:
    st.markdown("### ➕ Add a New Expense")
    with st.form("add_expense_form", clear_on_submit=True):
        a1, a2 = st.columns(2)
        with a1:
            e_date   = st.date_input("Date", value=datetime.date.today())
            e_cat    = st.text_input("Category", placeholder="e.g. Food, Travel, Utilities")
        with a2:
            e_amt    = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")
            e_method = st.text_input("Payment Method", placeholder="e.g. UPI, Cash, Card")
        e_desc  = st.text_input("Description", placeholder="Brief note about this expense")
        add_btn = st.form_submit_button("Add Expense ➜")

    if add_btn:
        if e_amt <= 0:
            st.warning("Amount must be greater than 0.")
        else:
            new_row = {"Date": pd.Timestamp(e_date), "Category": e_cat,
                       "Description": e_desc, "Amount": e_amt, "Payment Method": e_method}
            for col in st.session_state.df.columns:
                if col not in new_row:
                    new_row[col] = None
            st.session_state.df = pd.concat(
                [st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"✅ Added ₹{e_amt:,.2f} — {e_desc}")
            st.rerun()

# ── TAB 4 : COLUMN MAP ──
with tab4:
    st.markdown("### 📐 Column Mapping")
    st.markdown("<p style='color:#555'>Rename columns if your file has different headers.</p>", unsafe_allow_html=True)
    if st.session_state.df.empty:
        st.info("Upload a file first.")
    else:
        current_cols = list(st.session_state.df.columns)
        st.markdown(f"**Current columns:** `{'` · `'.join(current_cols)}`")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("col_rename_form"):
            rename_map = {}
            for col in current_cols:
                new_name = st.text_input(f"`{col}` →", key=f"rename_{col}", placeholder=col)
                if new_name.strip():
                    rename_map[col] = new_name.strip()
            apply_btn = st.form_submit_button("Apply Rename ➜")
        if apply_btn and rename_map:
            st.session_state.df.rename(columns=rename_map, inplace=True)
            st.success(f"✅ Renamed: {rename_map}")
            st.rerun()
