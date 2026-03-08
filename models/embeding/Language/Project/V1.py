from langchain_huggingface import HuggingFaceEmbeddings
en = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

user_docs = [
    "Taj Mahal is located in Agra, India.",
    "The Eiffel Tower is in Paris, France.",
    "the colosseum is in Rome, Italy."

]
User_query = "Where is the Eiffel Tower located?"
user_docs_embeddings = en.embed_documents(user_docs)
User_query_embedding = en.embed_query(User_query)
#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([User_query_embedding], user_docs_embeddings)
print("Similarities :", similarities)
print(similarities) 
print("Most similar document :", user_docs[similarities.argmax()])
