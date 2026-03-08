from langchain_huggingface  import HuggingFaceEmbeddings    
from sklearn.metrics.pairwise import cosine_similarity # use for model measure, sklearn is ML library
em=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   


#Docs
user_docs=  [ "Climate change is mainly caused by greenhouse gases released from burning fossil fuels and industrial activities.", 
             "The effects of climate change include rising global temperatures, melting glaciers, and increasing sea levels.", 
             "Climate change leads to extreme weather events such as heatwaves, floods, droughts, and stronger storms.", 
             "Climate change disrupts ecosystems, damages agriculture, and increases risks to human health worldwide.",
             "Using renewable energy, planting trees, and reducing emissions are key solutions to slow climate change.",
             "Climate change causes rising temperatures, melting ice caps, and sea level rise, impacting life globally.",
             "Global warming results in extreme weather like droughts, floods, and heatwaves, which are major climate impacts.", 
             "Reducing carbon emissions through clean energy and sustainability can help control climate change effects." ]

#user query
user_query= "What are the effects of climate change?"
 #embedding
user_docs_embeddings=em.embed_documents(user_docs)
query_embedding=em.embed_query(user_query)  

#similarity
similarities=cosine_similarity([query_embedding], user_docs_embeddings)[0] #2s- 1d
results = list(zip(user_docs, similarities))

#threshold
Threshold=0.9

#non_increasing order
filtered_results = [(doc, score) for doc, score in results if score >= Threshold]
filtered_results.sort(key=lambda x: x[1], reverse=True)

#top 3

top_3_results = filtered_results[:3]    


#output
if not top_3_results:
    print("No relevant documents found above the threshold.")
else:    
    print("Top 3 similar documents:")
    for doc, score in top_3_results:
         print(f"Document: {doc}\nSimilarity Score: {score:.4f}\n")