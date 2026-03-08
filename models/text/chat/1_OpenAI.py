from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()    
cm = OpenAI(model="gpt-4 ") 
result = cm.invoke("Where is the Eiffel Tower?")
print()
print(result)
