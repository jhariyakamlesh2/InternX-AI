from langchain_DeepSeek import DeepSeek
from dotenv import load_dotenv

load_dotenv()    
cm = DeepSeek(Model = "R1" )
result = cm.invoke("Where is the Eiffel Tower?")
print()
print(result)