from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt 
from transformers import pipeline
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Expense Tracker", layout="wide", page_icon="💸")
st.title("📊 EXPENSE TRACKER")

# ─────────────────────────────────────────────
#  LangChain + HuggingFace Local Pipeline Setup
# ─────────────────────────────────────────────
@st.cache_resource  # ✅ Model ek baar load hoga, har baar nahi
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256,   # ✅ GPT2 ke liye zaruri
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ─────────────────────────────────────────────
#  Prompt Template (inline — no .json needed)
# ─────────────────────────────────────────────
template = load_prompt("exptracker.json")

chain = template | llm

# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  Input Form
# ─────────────────────────────────────────────
st.header("➕ Add New Expense")

with st.form("expense_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        date   = st.date_input("Date of Payment")
        method = st.selectbox(
            "Method of Payment",
            ["Cash", "UPI", "Credit Card", "Debit Card", "Net Banking", "Other"],
        )

    with col2:
        paid_to     = st.text_input("Paid To", placeholder="e.g. Amazon, Zomato")
        description = st.text_input("Description", placeholder="e.g. Groceries, Dinner")

    with col3:
        amount = st.number_input("Amount Paid (Rs.)", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("✅ Add Expense", use_container_width=True)

# ─────────────────────────────────────────────
#  Add Row Logic
# ─────────────────────────────────────────────
if submitted:
    if amount <= 0:
        st.error("Amount 0 se zyada hona chahiye!")
    elif not paid_to.strip():
        st.error("'Paid To' field khali nahi honi chahiye!")
    else:
        prev_total = (
            st.session_state.expenses["Running Total"].iloc[-1]
            if not st.session_state.expenses.empty
            else 0
        )
        running_total = prev_total + amount

        new_row = {
            "Date of Payment":  str(date),
            "Method of Payment": method,
            "Paid To":          paid_to.strip(),
            "Description":      description.strip(),
            "Amount Paid":      amount,
            "Running Total":    running_total,
        }

        st.session_state.expenses = pd.concat(
            [st.session_state.expenses, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        st.success(f"✅ Rs. {amount:.2f} ka expense add ho gaya!")
        st.rerun()

# ─────────────────────────────────────────────
#  Totals + Table
# ─────────────────────────────────────────────
st.header("📋 Expense Log")

if not st.session_state.expenses.empty:
    total_to_date = st.session_state.expenses["Amount Paid"].sum()
    st.metric("💰 Total to Date (Rs.)", f"{total_to_date:,.2f}")
    st.dataframe(st.session_state.expenses, use_container_width=True, hide_index=True)

    # Clear button
    if st.button("🗑️ Clear All Expenses"):
        st.session_state.expenses = pd.DataFrame(
            columns=[
                "Date of Payment", "Method of Payment", "Paid To",
                "Description", "Amount Paid", "Running Total",
            ]
        )
        st.rerun()
else:
    st.info("Koi expense nahi hai abhi. Upar se add karo!")

# ─────────────────────────────────────────────
#  AI Summary
# ─────────────────────────────────────────────
st.header("🤖 AI Expense Analysis")

if st.button("🔍 Generate Expense Summary"):
    if st.session_state.expenses.empty:
        st.warning("Pehle kuch expenses add karo!")
    else:
        with st.spinner("AI analyze kar raha hai..."):
            try:
                data_text   = st.session_state.expenses.to_string(index=False)
                ai_response = chain.invoke({"data": data_text})
                st.subheader("📝 AI Analysis")
                # ✅ HuggingFacePipeline string return karta hai, .content nahi
                st.write(ai_response)
            except Exception as e:
                st.error(f"Error: {str(e)}")


