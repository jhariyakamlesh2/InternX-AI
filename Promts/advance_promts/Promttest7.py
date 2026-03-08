from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt 
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found! .env file check karo.")
st.set_page_config(page_title="Expense Tracker", layout="wide")
st.title("📊 EXPENSE TRACKER")

# LangChain Setup
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # ya koi bhi model
    huggingfacehub_api_token=token,  # explicitly pass karo
    task="text-generation",
    max_new_tokens=512,
)

chat_model = ChatHuggingFace(llm=llm)

st.header("🎓 AI Expense Tracker Assistant")

template = load_prompt("exptracker.json")

chain = template | chat_model   # ✅ chain with chat_model, NOT llm

# Session State
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

# Input Form
with st.form("expense_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Date of Payment")
        method = st.text_input("Method of Payment")

    with col2:
        paid_to = st.text_input("Paid To")
        description = st.text_input("Description")

    with col3:
        amount = st.number_input("Amount Paid", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Add Expense")

# Add Expense Logic
if submitted:
    prev_total = (
        st.session_state.expenses["Running Total"].iloc[-1]
        if not st.session_state.expenses.empty
        else 0
    )
    running_total = prev_total + amount

    new_row = {
        "Date of Payment": date,
        "Method of Payment": method,
        "Paid To": paid_to,
        "Description": description,
        "Amount Paid": amount,
        "Running Total": running_total,
    }

    st.session_state.expenses = pd.concat(
        [st.session_state.expenses, pd.DataFrame([new_row])],
        ignore_index=True,
    )

# Totals
if not st.session_state.expenses.empty:
    total_to_date = st.session_state.expenses["Amount Paid"].sum()
else:
    total_to_date = 0.0

st.metric("Total to Date (Rs.)", f"{total_to_date:.2f}")
st.dataframe(st.session_state.expenses, use_container_width=True)

# AI Summary
if st.button("Generate Expense Summary"):
    if not st.session_state.expenses.empty:
        data_text = st.session_state.expenses.to_string(index=False)
        ai_response = chain.invoke({"data": data_text})
        st.subheader("AI Expense Analysis")
        st.write(ai_response.content)   # ✅ use .content for chat model output
    else:
        st.warning("No expense data available for analysis.")