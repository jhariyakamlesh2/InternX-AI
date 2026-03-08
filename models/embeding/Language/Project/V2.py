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

score = cosine_similarity([User_query_embedding], user_docs_embeddings)[0]
index, score = sorted(list(enumerate(score)), key=lambda x: x[1])[-1]
                      
print("---------START---------")
print("question:", User_query)
print("LLM answer:", user_docs[index])
print("countdence score:", score)
print("---------END---------")