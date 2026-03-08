from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

cm = ChatHuggingFace(llm=llm)

# Hugging Face endpoint
st.header("🎓 AI Lecture Assistant")

topic = st.text_input("Enter topic:")
number_of_lines = st.number_input("Enter number of lines:", min_value=1, value=2)
style = st.text_input("Enter style (e.g. simple, professional, storytelling):")
language = st.text_input("Enter language (e.g. English, Hindi):")

template = load_prompt("promts.json")

innput_variables = ["style", "number_of_lines", "topic", "language"],

system_prompt = template.invoke({
    "style": style,   # <-- lowercase
    "number_of_lines": number_of_lines,
    "topic": topic,
    "language": language
})


if st.button("Generate"):
    result = cm.invoke(system_prompt)
    st.write(result.content)
else:
    st.warning("Please fill all fields.")