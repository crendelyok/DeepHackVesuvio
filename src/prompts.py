from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы помощник по написанию докладов, ваша задача — писать отличные доклад из 5 абзацев."
            " Создайте лучший доклад по запросу пользователя."
            " Если пользователь предоставляет критику, ответьте переработанной версией ваших предыдущих попыток.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы учитель, оценивающий представленый доклад. Сформируйте критику и рекомендации для представленной работы пользователя."
            " Предоставьте подробные рекомендации, включая требования к объему, глубине, стилю и т.д.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
