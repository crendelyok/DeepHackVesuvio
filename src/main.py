import asyncio
import os
from typing import List, Sequence

from langchain_community.chat_models import GigaChat
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from define_db import define_db
from graph import build_graph
from prompts import DEFAULT_PROMPT, REFLECTION_PROMPT


async def main():
    db = define_db()
    context_retriever = db.as_retriever()

    llm = GigaChat(
        credentials=os.environ["GIGACHAD_VIP_CREDS"],
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-Pro",
    )

    generate = DEFAULT_PROMPT | llm
    reflect = REFLECTION_PROMPT | llm
    """
    qa_chain_gen = RetrievalQA.from_chain_type(
        llm,
        retriever=context_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": DEFAULT_PROMPT}
    )
    qa_chain_refl = RetrievalQA.from_chain_type(
        llm,
        retriever=context_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": REFLECTION_PROMPT}
    )
    """

    async def generation_node(state: Sequence[BaseMessage]):
        # return await qa_chain_gen.ainvoke({"query" : "", "messages": state})
        return await generate.ainvoke(
            {"messages": state, "context": str(context_retriever)}
        )

    async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        # Other messages we need to adjust
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        # First message is the original user request. We hold it the same for all nodes
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        # res = qa_chain_refl.ainvoke({"messages": translated})
        res = await reflect.ainvoke(
            {"messages": translated, "context": context_retriever}
        )
        # We treat the output of this as human feedback for the generator
        return HumanMessage(content=res.content)

    graph = build_graph(generation_node, reflection_node)

    async for event in graph.astream(
        [
            HumanMessage(
                content="Formulate an innovative hypothesis for further scientific research based on the scientific articles by Korakianintis."
            )
        ],
    ):
        print(event)
        print("---")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
