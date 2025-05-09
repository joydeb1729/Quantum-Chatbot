from flask import Flask, render_template, jsonify,request
from src.helper import download_hf_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.prompt import template
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

embeddings = download_hf_embeddings()

index_name = "qchatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
    )

chain_type_kwargs = {'prompt': PROMPT}

llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="shisa-ai/shisa-v2-llama3.3-70b:free",
    temperature=0.5,
)

qna = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type='stuff',
                                  retriever=docsearch.as_retriever(search_kwargs={'k':2}),
                                  return_source_documents=True,
                                  chain_type_kwargs=chain_type_kwargs
                                  )


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    result = qna({'query':input})
    print('Response: ', result['result'])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8081,debug=True)