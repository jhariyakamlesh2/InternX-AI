from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint( repo_id="Qwen/Qwen2.5-1.5B-Instruct", task="text-generation" )

cm = ChatHuggingFace(llm=endpoint)

# Hugging Face endpoint
st.header("🎓 AI Lecture Assistant")

topic = st.text_input("Enter topic:")
number_of_lines = st.number_input("Enter number of lines:", min_value=1, value=2)
style = st.text_input("Enter style (e.g. simple, professional, storytelling):")
language = st.text_input("Enter language (e.g. English, Hindi):")

template = load_prompt("promts.json")


if st.button("Generate"):
    chain = template | cm
    result = chain.invoke({
        "style": style,
        "number_of_lines": number_of_lines,
        "topic": topic,
        "language": language
    })
    st.write(result.content)
else:
    st.warning("Please fill all fields.")