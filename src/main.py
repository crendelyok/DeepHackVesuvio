import logging
import os
import sys

import yaml
from langchain.chat_models.gigachat import GigaChat

from agent_responder import ResponderWithRetries
from graph import build_graph
from prompts.generator import generation_prompt
from prompts.revisor import revisor_prompt
from summarizer import Summarizer, SummarizerMock

std_logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main(
    config: dict, papers: list[str], question: str, credentials: str, debug: bool = True
) -> str:
    """Generates ideas for a given question and papers.

    Args:
        config (dict): configuration for the llm model
        papers (list[str]): list of paths to papers to summarize. e.g. [https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf]
        question (str): user specified question
        credentials (str): gigachat credentials
        debug (bool, optional): debug allow to mock summarization as the longest operation. Defaults to True.

    Returns:
        str: dialogue between the generator and revisor agents.
    """
    llm = GigaChat(
        credentials=credentials,
        profanity_check=False,
        verify_ssl_certs=False,
        timeout=600,
        model=config["model"],
        top_p=config["top_p"],
        scope=config["scope"],
    )

    if not debug:
        summarizer = Summarizer(
            papers,
            llm=llm,
        )
    else:
        summarizer = SummarizerMock("", llm)  # better to add tests

    revisor_chain = revisor_prompt | llm
    revisor = ResponderWithRetries(runnable=revisor_chain, actor="revisor")

    generator_chain = generation_prompt | llm
    generator = ResponderWithRetries(
        runnable=generator_chain,
        actor="generation",
    )

    graph = build_graph(summarizer, generator, revisor)
    events = graph.stream(
        {
            "iteration": 0,
            "question": question,
        }
    )

    output_str = ""
    for i, step in enumerate(events):
        node, output = next(iter(step.items()))
        if node == "summarize":
            output = output["context"]
        elif node == "generator":
            output = output["generation"]
        else:
            output = output["reflection"]
        std_logger.info(output)
        output_str += f"## {i+1}. {node}\n"
        output_str += str(output) + "\n"
        output_str += "---" * 10 + "\n"

    return output_str


if __name__ == "__main__":
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    main(
        config,
        papers=config["papers"],
        question=config["question"],
        credentials=os.environ["GIGACHAD_VIP_CREDS"],
        debug=False,
    )
