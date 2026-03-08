from langchain_openai import OpenAIEmbeddings
em = OpenAIEmbeddings(Model_Name="sentence-transformers/all-MiniLM-L6-v2")
text = "Where is the Eiffel Tower?"
result = em.embed_query(text)

print(result)
