import asyncio
import os

from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage

from define_db import get_article_summary
from graph import build_graph
from prompts import GENERATION_PROMPT, REFLECTION_PROMPT


async def main():
    llm = GigaChat(
        credentials=os.environ["GIGACHAD_VIP_CREDS"],
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-Pro",
    )

    context = get_article_summary("../articles/example.pdf")

    print(f"context : {context}")

    generate = GENERATION_PROMPT | llm
    reflect = REFLECTION_PROMPT | llm

    async def generation_node(message):
        # print(message)
        # print(type(message))
        return await generate.ainvoke(
            {
                "context": context,
                "reflection": "",
                "question": "Generate research idea based on provided context.",
            }
        )

    async def reflection_node(message):
        # Other messages we need to adjust
        # cls_map = {"ai": HumanMessage, "human": AIMessage}
        # print(f'message from generator : {message}')
        # print(type(message))
        # First message is the original user request. We hold it the same for all nodes
        """
        translated = [message[0]] + [
            cls_map[message.type](content=message.content)
        ]
        """
        res = await reflect.ainvoke(
            {
                "context": context,
                "generation": message,
                "question": "Generate research idea based on provided context.",
            }
        )
        # We treat the output of this as human feedback for the generator
        return HumanMessage(content=res.content)

    graph = build_graph(generation_node, reflection_node)

    async for event in graph.astream(
        [HumanMessage(content="Generate research idea based on provided context.")],
    ):
        print(event)
        print("---")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
