from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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

template = PromptTemplate(
    template="""

    you are highly skilled literature assistant.

    TASK: 
    - Answer the user qustions in the style of {style}.
    - The responce must be exactly {number_of_lines} lines long.
    - Every line should be meaningful and informative.
    - The contebt must be focused on the topic of {topic}.
    - Do not include anything outside the topic.

    LANGUAGE RULES:
    - Responce only in {language}.language.
    - Do not use any language other than {language}.Unless they are anavoidable proper nouns or technical terms.

    FORMATE RULES:
    - Do not include heanding, explanation, or extra notes.
    - Always start a new sentance from a new line.

    QUALITY RULES:
    - Ensure the responce matches the ton and vocabulary.
    - Avoid repeatation and keep he writing nutral.

    Before producing the final answer, internally verify:
    1. Style is correctly applied.
    2. Topice is respected.
    3. Line count is exact.
    4. Language is currect.

    Now generete the final responce.
"""
)
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