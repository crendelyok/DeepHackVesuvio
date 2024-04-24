import os

from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("example.pdf")
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(text)
print(f"Total documents: {len(text)}")

embeddings = GigaChatEmbeddings(
    credentials=os.environ["GIGACHAD_VIP_CREDS"],
    verify_ssl_certs=False,
    scope="GIGACHAT_API_CORP",
)
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

model = GigaChat(
    credentials=os.environ["GIGACHAD_VIP_CREDS"],
    verify_ssl_certs=False,
    scope="GIGACHAT_API_CORP",
    model="GigaChat-Plus",
)

qa_chain = RetrievalQA.from_chain_type(model, retriever=db.as_retriever()) | print

while True:
    user_input = input("User: ")
    qa_chain.invoke({"query": user_input})
