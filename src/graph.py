from typing import TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    iteration: int
    question: str
    context: str
    reflection: str
    generation: str


def build_graph(summarizer, generator, revisor, max_iterations):
    builder = StateGraph(AgentState)

    builder.add_node("summarize", summarizer.summarize)
    builder.add_node("generator", generator.respond)
    builder.add_node("revisor", revisor.respond)

    builder.add_edge("summarize", "generator")
    builder.add_edge("revisor", "generator")

    def event_loop(state) -> str:
        # in our case, we'll just stop after N plans
        if state["iteration"] > max_iterations:
            return END
        return "revisor"

    builder.add_conditional_edges("generator", event_loop)
    builder.set_entry_point("summarize")
    graph = builder.compile()
    return graph
