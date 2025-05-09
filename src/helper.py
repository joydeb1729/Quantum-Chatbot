from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    document = loader.load()
    
    return document


def text_split(extracted_data):
    splited_text = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    text_chunk = splited_text.split_documents(extracted_data)
    
    return text_chunk

def download_hf_embeddings():
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding