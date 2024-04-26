import logging
import sys

from langsmith import traceable

std_logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ResponderWithRetries:
    def __init__(self, runnable, actor: str, validator=None, n_retries: int = 5):
        self.runnable = runnable
        self.validator = validator
        self.actor = actor
        self.n_retries = n_retries

    @traceable
    def respond(self, state):
        response = []
        default_input_state = {
            "context": state["context"],
            "question": state["question"],
        }
        default_return_state = {
            "context": state["context"],
            "iteration": state["iteration"] + 1,
            "question": state["question"],
        }
        for attempt in range(self.n_retries):
            try:
                if self.actor == "generation":
                    response = self.runnable.invoke(
                        {
                            **default_input_state,
                            "reflection": state["reflection"],
                        }
                    )
                else:
                    response = self.runnable.invoke(
                        {
                            **default_input_state,
                            "generation": state["generation"],
                        }
                    )

                if self.validator:
                    self.validator.invoke(response)
                if self.actor == "generation":
                    return {
                        **default_return_state,
                        "reflection": state["reflection"],
                        "generation": response.content,
                    }
                return {
                    **default_return_state,
                    "reflection": response.content,
                    "generation": state["generation"],
                }
            except Exception as e:
                std_logger.error(repr(e))
        std_logger.warning("ResponderWithRetries: something strange happened")
        return {
            **default_return_state,
            "reflection": state["reflection"],
            "generation": state["generation"],
        }
