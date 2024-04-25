import concurrent.futures

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from prompts.summarizer import prompt_template, refine_template


class Summarizer:
    def __init__(self, pdf_paths, llm):
        self.pdf_paths = pdf_paths
        self.llm = llm

    def get_article_summary(self, path_to_pdf):
        loader = PyPDFLoader(path_to_pdf)

        docs = loader.load()

        split_docs = CharacterTextSplitter(
            chunk_size=5000, chunk_overlap=500
        ).split_documents(docs)

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=self.prompt,
            refine_prompt=self.refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )

        result = chain({"input_documents": split_docs}, return_only_outputs=True)
        return result["output_text"] + "\n"

    def summarize(self, question: str):
        prompt = PromptTemplate.from_template(
            prompt_template, partial_variables={"question": question}
        )
        refine_prompt = PromptTemplate.from_template(
            refine_template, partial_variables={"question": question}
        )

        self.prompt = prompt
        self.refine_prompt = refine_prompt

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(self.get_article_summary, self.pdf_paths))

        general_result = ""
        for individual_result in results:
            general_result += individual_result

        print(f"Summarizer.summarize({question}): {general_result}")
        return {
            "context": general_result,
        }


class SummarizerMock:
    def __init__(self, pdf_paths):
        pass

    def summarize(self, question: str):
        text = (
            "Sub-center ArcFace improves face recognition by relaxing the intra-class constraint of ArcFace, "
            "resulting in increased robustness against label noise. This is achieved by designing K sub-centers for each "
            "class, where the training sample only needs to be close to any of the K positive sub-centers. The proposed method "
            "encourages one dominant sub-class containing the majority of clean faces and non-dominant sub-classes including "
            "hard or noisy faces. Experiments show improved performance under massive real-world noise."
        )
        return {
            "context": text,
        }
