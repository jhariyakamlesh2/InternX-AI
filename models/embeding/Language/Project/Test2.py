from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
en = HuggingFaceEmbeddings(Model_name="google/translategemma-27b-it")
user_docs = [
    "Where is the Kanha Natioanal Park located?",
    "The Taj Mahal is located in Agra, India."
]
user_query = "Where is Kanha National Park located?"
user_docs_embeddings = en.embed_documents(user_docs)
user_query_embedding = en.embed_query(user_query)
similarities = cosine_similarity([user_query_embedding], user_docs_embeddings)
print("Similarities:", similarities)
print(similarities)
print("Most similar document:", user_docs[similarities.argmax()])
