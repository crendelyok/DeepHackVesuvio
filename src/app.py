import streamlit as st

from main import main

st.set_page_config(page_title="Генератор идей")
st.header("Генерация идей")
st.write(
    "Введите ссылки на pdf, как указано в примере через запятую. Затем введите запрос и нажмите кнопку 'нажми меня'"
)

credentials = st.text_area(
    label="credentials",
)

pdfs = st.text_area(
    label="pdfs",
    value="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf,https://arxiv.org/pdf/1801.07698",
)
question = st.text_input(
    label="question", value="каким образом мне еще улучшить качество модели?"
)

if pdfs and st.button("Нажми меня"):
    files = list(set(pdfs.split(",")))
    print(files)
    st.write(f"pdfs that will be processed: {files}")
    with st.spinner("Processing..."):
        config = {
            "model": "GigaChat-Plus-preview",
            "top_p": 0.8,
            "scope": "GIGACHAT_API_CORP",
        }
        out_str = main(
            config, papers=files, question=None, credentials=credentials, debug=False
        )
        st.write(out_str)
