prompt_template = """Write a concise summary of the following in order to generate new research hypothesis.
                    Pay special attention to the abstract, conclusions, further developments and especially:
                    !!!{question}
                    {text}
                    CONCISE SUMMARY:"""


refine_template = (
    "Your job is to produce a final summary\n"
    "Extract key information from text to help generate new research idea.\n"
    "Pay special attention to the abstract, conclusions, further developments and especially !!!{question}.\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "If the context isn't useful, return the original summary."
)
