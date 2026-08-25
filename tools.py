import os
import re
import datetime
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS


def get_llm(temperature: float = 0.4):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)


@tool
def search_ai_news(query: str, max_results: int = 10) -> list:
    """Search the web for recent AI agent news articles and return title, snippet and url for each result."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "Untitled"),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
    except Exception as e:
        results.append({
            "title": "Search unavailable",
            "body": f"Live web search could not be reached ({e}). Using fallback placeholder content.",
            "href": "",
        })
    return results


@tool
def summarize_articles(articles_text: str) -> str:
    """Summarize a block of article text into a concise numbered list of 2-3 sentence summaries."""
    llm = get_llm()
    prompt = (
        "Summarize each article below in 2-3 sentences. Keep the title and url for each. "
        "Output a numbered markdown list.\n\n" + articles_text
    )
    return llm.invoke(prompt).content


@tool
def generate_newsletter_html(summary_text: str) -> str:
    """Turn article summaries into a clean HTML newsletter with a subject line."""
    llm = get_llm()
    prompt = (
        "Write a clean, engaging weekly newsletter about the latest AI agent news, "
        "using the article summaries below as source material. "
        "Return valid HTML with simple inline styling, and start the response with a single line "
        "'SUBJECT: <subject line>' before the HTML body.\n\n" + summary_text
    )
    return llm.invoke(prompt).content


def split_subject_body(text: str):
    match = re.search(r"SUBJECT:\s*(.+)", text)
    subject = match.group(1).strip() if match else "Weekly AI Agent News Digest"
    body = re.sub(r"SUBJECT:\s*.+\n?", "", text, count=1).strip()
    body = re.sub(r"^```html", "", body).strip()
    body = re.sub(r"```$", "", body).strip()
    return subject, body


def save_newsletter(subject: str, body: str) -> str:
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("output", f"newsletter_{timestamp}.html")
    if "<title>" not in body.lower():
        body = f"<title>{subject}</title>\n{body}"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(body)
    return filename
