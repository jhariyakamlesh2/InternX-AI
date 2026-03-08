from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

cm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation"))

# Hugging Face endpoint
st.header("🎓 AI Lecture Assistant")

topic = st.text_input("Enter topic:")
number_of_lines = st.number_input("Enter number of lines:", min_value=1, value=2)
style = st.text_input("Enter style (e.g. simple, professional, storytelling):")
language = st.text_input("Enter language (e.g. English, Hindi):")

system_prompt = f"You are a helpful history assistant.Write about the topic '{topic}' in {style} style.The response should be {number_of_lines} lines long.Write the content in {language}."


if st.button("Generate"):
    if topic and style and language:
        result = cm.invoke(system_prompt)
        st.write(result.content)
    else:
        st.warning("Please fill all fields.")