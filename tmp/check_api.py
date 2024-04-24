"""Пример работы с чатом через gigachain"""

import os

from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage

# Авторизация в сервисе GigaChat
chat = GigaChat(
    credentials=os.environ["GIGACHAD_VIP_CREDS"],
    verify_ssl_certs=False,
    scope="GIGACHAT_API_CORP",
    model="GigaChat-Pro",
)

messages = [
    SystemMessage(
        content="Ты крутой бот Итачи из Наруто, который помогает пользователю решить его проблемы."
    )
]

while True:
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    print("Bot: ", res.content)
