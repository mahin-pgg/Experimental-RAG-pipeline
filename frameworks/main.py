from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b", 
    temperature=0
    )

response = llm.invoke(
    "Explain how retrieval augmented generation works in simple terms."
)

print(response.content)