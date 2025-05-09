from src.helper import *
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
#OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

extracted_data = load_pdf('data')
text_chunk = text_split(extracted_data)
embedding = download_hf_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("qchatbot")

texts = [t.page_content for t in text_chunk]
doc_search = PineconeVectorStore.from_texts(
    texts=texts,
    embedding=embedding,
    index_name="qchatbot",
    namespace="default"  # optional
)

