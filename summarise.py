from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

# ollama
Settings.llm = Ollama(model="mistral:v0.3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)

documents = SimpleDirectoryReader("data").load_data()
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("List what the document says about Discipline.")
print(response)
