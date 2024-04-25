from langchain_core.prompts import ChatPromptTemplate

revisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "You are a supervisor evaluating a submitted idea for innovative research. "
            "Form a reasoned critique and recommendation for the idea presented based on given context. "
            "Provide detailed recommendations."
            "\nCONTEXT: {context}\n QUESTION: {question} \nGENERATION: {generation}"
        ),
    ]
)
