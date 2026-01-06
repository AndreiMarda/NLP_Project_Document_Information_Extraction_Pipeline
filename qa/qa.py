from transformers import pipeline

# Load a small English QA model
# (first run will download it)
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


def answer_question(question: str, context: str) -> str:
    """
    Use a QA model to answer a natural-language question
    given the document text as context.
    """
    result = qa_pipeline(question=question, context=context)
    return result.get["answer", ""]
