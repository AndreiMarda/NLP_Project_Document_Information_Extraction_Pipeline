from docx import Document

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs)
