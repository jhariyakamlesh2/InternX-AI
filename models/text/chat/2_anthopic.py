from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv  

load_dotenv()  

cm= ChatAnthropic(model= "Claude-2")
result = cm.invoke("Where is Kanha National Park located?")
print() 
print(result)
