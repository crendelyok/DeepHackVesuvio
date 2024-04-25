from langchain_core.prompts import PromptTemplate

template = """You are a young and ambitious scientist who is looking for a fresh idea for inovative research. Your task is to come up with a briliant research idea based on given context. Create the best possible idea for your scientific supervisor. If scientific supervisor provides a critique, respond with a reworked version of your previous attempts.
{context}
"""
DEFAULT_PROMPT = PromptTemplate.from_template(template)

template_refl = """You are a supervisor evaluating a submitted idea for innovative research. Form a reasoned critique and recommendation for the idea presented based on given context. Provide detailed recommendations.
{context}
Question: {question}
Helpful Answer:"""
REFLECTION_PROMPT = PromptTemplate.from_template(template_refl)

"""
DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a young and ambitious scientist who is looking for a fresh idea for inovative research."
            "Your task is to come up with a briliant research idea based on given context."
            "Create the best possible idea for your scientific supervisor."
            "If scientific supervisor provides a critique, respond with a reworked version of your previous attempts."
            "Your context: {context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a supervisor evaluating a submitted idea for innovative research."
            "Form a reasoned critique and recommendation for the idea presented based on given context."
            "Provide detailed recommendations."
            "Your context: {context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
"""
