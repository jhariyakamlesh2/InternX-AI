from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Expense Tracker", layout="wide")
st.title("📊 EXPENSE TRACKER")

# LLM Setup
llm = ChatHuggingFace(
    model=HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/gpt-3.5-turbo",
        task="text-generation",
        model_kwargs={"max_new_tokens": 512, "temperature": 0.7}
    )
)

# Prompt Template (inline - reliable)
prompt = PromptTemplate(
    input_variables=["data"],
    template="""You are a personal finance assistant. Analyze the following expense data and provide a concise summary in plain English.

Do NOT write any code. Do NOT show any Python or Streamlit code. Only give a financial analysis.

Your response must include:
1. Total amount spent
2. Spending breakdown by recipient/category
3. Most expensive transaction
4. Payment method used most often
5. 2-3 practical money-saving tips based on the data

Expense Data:
{data}

Provide a clear, helpful financial summary:"""
)

chain = prompt | llm

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
col1, col2, col3 = st.columns(3)

with col1:
    date = st.date_input("Date of Payment")
    method = st.text_input("Method of Payment")

with col2:
    paid_to = st.text_input("Paid To")
    description = st.text_input("Description")

with col3:
    amount = st.number_input("Amount Paid", min_value=0.0, format="%.2f")

if st.button("Add Expense"):
    if amount > 0 and paid_to.strip():
        prev_total = (
            st.session_state.expenses["Running Total"].iloc[-1]
            if not st.session_state.expenses.empty
            else 0
        )
        running_total = prev_total + amount

        new_row = {
            "Date of Payment": str(date),
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
        st.success(f"Expense added: ₹{amount:.2f} to {paid_to}")
    else:
        st.warning("Please enter a valid amount and 'Paid To' field.")

# -------------------------------
# Display Totals
# -------------------------------
total_to_date = (
    st.session_state.expenses["Amount Paid"].sum()
    if not st.session_state.expenses.empty
    else 0.0
)

st.metric("Total to Date ($)", f"{total_to_date:.2f}")

# -------------------------------
# Display Table
# -------------------------------
st.dataframe(st.session_state.expenses, use_container_width=True)

# -------------------------------
# AI Expense Summary
# -------------------------------
if st.button("Generate Expense Summary"):
    if not st.session_state.expenses.empty:
        data_text = st.session_state.expenses.to_string(index=False)
        with st.spinner("Analyzing your expenses..."):
            ai_response = chain.invoke({"data": data_text})
        st.subheader("🤖 AI Expense Analysis")
        st.write(ai_response)
    else:
        st.warning("No expense data available for analysis.")
