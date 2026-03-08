from langchain_google import ChatGooglePalm
from dotenv import load_dotenv
load_dotenv()

cm = ChatGooglePalm(model="models/chat-bison-001")
result = cm.invoke("Where is the Eiffel Tower?")
print(result)
print(result.content)
