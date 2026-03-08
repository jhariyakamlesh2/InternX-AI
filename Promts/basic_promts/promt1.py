from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Hugging Face endpoint
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

cm = ChatHuggingFace(llm=endpoint)

# User inputs
topic = input("Enter topic: ")
number_of_lines = int(input("Enter number of lines: "))
style = input("Enter style: ")
language = input("Enter language: ")

print()

# Correct prompt
system_prompt = f"""
You are a helpful history assistant.
Write about the topic "{topic}" in {style} style.
The response should be {number_of_lines} lines long.
Write the content in {language}.
"""

result = cm.invoke(system_prompt)
print(result.content)

