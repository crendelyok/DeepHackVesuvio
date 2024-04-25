import os

import yaml
from langchain.chat_models.gigachat import GigaChat

from agent_responder import ResponderWithRetries
from graph import build_graph
from prompts.generator import generation_prompt
from prompts.revisor import revisor_prompt
from summarizer import Summarizer, SummarizerMock


def main(config: dict, papers: list[str], question: str, debug: bool = True):
    llm = GigaChat(
        credentials=os.environ["GIGACHAD_VIP_CREDS"],
        profanity_check=False,
        verify_ssl_certs=False,
        timeout=600,
        model=config["model"],
        top_p=config["top_p"],
        scope=config["scope"],
    )

    summarizer = SummarizerMock("")
    if not debug:
        summarizer = Summarizer(
            papers,
            llm=llm,
        )

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
        print(output)
        output_str += f"## {i+1}. {node}\n"
        output_str += str(output) + "\n"
        output_str += "---" * 10 + "\n"

    print(output_str)
    return output_str


if __name__ == "__main__":
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    main(config, papers=config["papers"], question=config["question"], debug=True)
