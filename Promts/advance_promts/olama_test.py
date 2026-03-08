from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt 
from langchain_core.prompts import PromptTemplate
import datetime
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
# Streamlit Page Config
st.set_page_config(page_title="Expense Tracker", layout="wide")

st.title(" 📊 EXPENSE TRACKER  ")

llm = OllamaLLM(model="llama3")

# Hugging Face endpoint
st.header("🎓 AI Expense Tracker Assistant")

template = load_prompt("exptracker.json")

chain = template | llm

# Session State Initialization

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

# -------------------------------
# Input Form
# -------------------------------
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

# -------------------------------
# Add Expense Logic
# -------------------------------
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

# -------------------------------
# Display Totals
# -------------------------------
if not st.session_state.expenses.empty:
    total_to_date = st.session_state.expenses["Amount Paid"].sum()
else:
    total_to_date = 0.0

st.metric("Total to Date (Rs.)", f"{total_to_date:.2f}")

# -------------------------------
# Display Table
# -------------------------------
st.dataframe(st.session_state.expenses, use_container_width=True)

# -------------------------------
# LangChain AI Explanation
# -------------------------------
if st.button("Generate Expense Summary"):
    if not st.session_state.expenses.empty:
        data_text = st.session_state.expenses.to_string(index=False)
        ai_response = chain.invoke({"data": data_text})
        st.subheader("AI Expense Analysis")
        st.write(ai_response)
    else:
        st.warning("No expense data available for analysis.")