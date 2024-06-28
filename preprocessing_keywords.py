from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor, MetadataReplacementPostProcessor, LongContextReorder
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, Settings
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("./data").load_data()
# bge-base embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

# ollama
Settings.llm = Ollama(model="mistral:latest", request_timeout=360.0)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7),
        KeywordNodePostprocessor(
            required_keywords=["word1", "word2"], exclude_keywords=["word3", "word4"]),
        MetadataReplacementPostProcessor(
    target_metadata_key="window"),
        LongContextReorder()
    ]
)

# all node post-processors will be applied during each query
response = query_engine.query("query string")