from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)

print(llm.invoke("What is the capital of India?"))

