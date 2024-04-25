from langchain_core.messages import HumanMessage
from langsmith import traceable


class ResponderWithRetries:
    def __init__(self, runnable, actor: str, validator=None):
        self.runnable = runnable
        self.validator = validator
        self.actor = actor

    @traceable
    def respond(self, state):
        response = []
        for attempt in range(5):
            try:
                if self.actor == "generation":
                    response = self.runnable.invoke(
                        {
                            "context": state["context"],
                            "question": state["question"],
                            "reflection": state["reflection"],
                        }
                    )
                else:
                    response = self.runnable.invoke(
                        {
                            "context": state["context"],
                            "question": state["question"],
                            "generation": state["generation"],
                        }
                    )

                if self.validator:
                    self.validator.invoke(response)
                if self.actor == "generation":
                    return {
                        "context": state["context"],
                        "iteration": state["iteration"] + 1,
                        "question": state["question"],
                        "reflection": state["reflection"],
                        "generation": response.content,
                    }
                return {
                    "context": state["context"],
                    "iteration": state["iteration"] + 1,
                    "question": state["question"],
                    "reflection": response.content,
                    "generation": state["generation"],
                }
            except Exception as e:
                message = HumanMessage(content=repr(e))
                if response.response_metadata["finish_reason"] != "error":
                    state["messages"] += [message]
        print("something bad happened")
        return {
            "context": state["context"],
            "iteration": state["iteration"] + 1,
            "question": state["question"],
            "reflection": state["reflection"],
            "generation": state["generation"],
        }
