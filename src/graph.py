from typing import List

from langchain_core.messages import BaseMessage
from langgraph.graph import END, MessageGraph


def build_graph(generation_node, reflection_node):
    builder = MessageGraph()
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")

    def should_continue(state: List[BaseMessage]):
        if len(state) > 6:
            # End after 3 iterations
            return END
        return "reflect"

    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")
    graph = builder.compile()

    return graph
