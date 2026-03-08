from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint( repo_id="Qwen/Qwen2.5-1.5B-Instruct", task="text-generation" ) 
cm = ChatHuggingFace(llm=endpoint)

result = cm.invoke("Where is Eiffel Tower located?") 
print(result)
