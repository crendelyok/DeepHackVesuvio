import os

from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.vectorstores import Chroma


def load_and_split_pdf(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf)
    text = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(text)
    return documents


def define_db():
    list_of_path_to_pdf = ["../articles/korakianitis_valves.pdf"]
    documents = []
    [documents.append(load_and_split_pdf(path)) for path in list_of_path_to_pdf]
    documents = [doc for sublist in documents for doc in sublist]

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

    return db
