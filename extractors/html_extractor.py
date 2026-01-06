import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    html = response.text

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text(separator=" ")
    # strip multiple spaces/newlines
    text = " ".join(text.split())
    return text
