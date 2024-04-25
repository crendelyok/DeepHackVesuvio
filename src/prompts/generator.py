from langchain_core.prompts import ChatPromptTemplate

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "You are a young and ambitious scientist who is looking for a fresh idea for inovative research. "
            "Your task is to come up with a briliant research idea based on given CONTEXT. "
            "Create the best possible idea for your scientific supervisor. "
            "If scientific supervisor provides a critique, respond with a reworked version of your previous attempts. "
            "\nCONTEXT: {context}\n QUESTION: {question} \nREFLECTION: {reflection}"
        ),
    ]
)
