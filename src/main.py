import asyncio
import os
from typing import List, Sequence

from langchain_community.chat_models import GigaChat
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from graph import build_graph
from prompts import DEFAULT_PROMPT, REFLECTION_PROMPT


async def main():
    llm = GigaChat(
        credentials=os.environ["GIGACHAD_VIP_CREDS"],
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-Pro",
    )

    generate = DEFAULT_PROMPT | llm
    reflect = REFLECTION_PROMPT | llm

    async def generation_node(state: Sequence[BaseMessage]):
        return await generate.ainvoke({"messages": state})

    async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        # Other messages we need to adjust
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        # First message is the original user request. We hold it the same for all nodes
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        res = await reflect.ainvoke({"messages": translated})
        # We treat the output of this as human feedback for the generator
        return HumanMessage(content=res.content)

    graph = build_graph(generation_node, reflection_node)

    async for event in graph.astream(
        [
            HumanMessage(
                content="Значение 'Маленького принца' в детстве современных детей"
            )
        ],
    ):
        print(event)
        print("---")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
