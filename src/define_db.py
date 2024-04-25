import concurrent.futures
import os

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.gigachat import GigaChat
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

model = GigaChat(
    credentials=os.environ["GIGACHAD_VIP_CREDS"],
    verify_ssl_certs=False,
    scope="GIGACHAT_API_CORP",
    model="GigaChat-Pro",
)

prompt_template = """Write a concise summary of the following in order to generate new research hypothesis.
                    Pay special attention to the abstract, conclusions and further developments:
                    {text}
                    CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "Extract key information from text to help generate new research idea.\n"
    "Pay special attention to the abstract, conclusions and further developments.\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\exapmles/articles/korakianitis_heart.pdf exapmles/korakianitis_valves.pdfn"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)


def get_article_summary(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf)

    docs = loader.load()

    split_docs = CharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500
    ).split_documents(docs)

    chain = load_summarize_chain(
        llm=model,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    result = chain({"input_documents": split_docs}, return_only_outputs=True)
    return result["output_text"]


def get_summary_of_all_articles(paths):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(get_article_summary, paths))
    return results
