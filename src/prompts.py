from langchain_core.prompts import PromptTemplate

GENERATION_PROMPT_TEMPLATE = (
    "You are a young and ambitious scientist who is looking for a fresh idea for inovative research. "
    "Your task is to come up with a briliant research idea based on given context. "
    "Create the best possible idea for your scientific supervisor. "
    "If scientific supervisor provides a critique, respond with a reworked version of your previous attempts. "
    "\nCONTEXT: {context}\n QUESTION: {question} \nREFLECTION: {reflection}"
)
generation_input_variables = ["context", "question", "reflection"]
GENERATION_PROMPT = PromptTemplate(
    input_variables=generation_input_variables, template=GENERATION_PROMPT_TEMPLATE
)


REFLECTION_PROMPT_TEMPLATE = (
    "You are a supervisor evaluating a submitted idea for innovative research. "
    "Form a reasoned critique and recommendation for the idea presented based on given context. "
    "Provide detailed recommendations."
    "\nCONTEXT: {context}\n QUESTION: {question} \nGENERATION: {generation}"
)
reflection_input_variables = ["context", "question", "generation"]
REFLECTION_PROMPT = PromptTemplate(
    input_variables=reflection_input_variables, template=REFLECTION_PROMPT_TEMPLATE
)
