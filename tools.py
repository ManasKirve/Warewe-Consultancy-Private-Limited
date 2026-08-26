import os
import re
import datetime


def get_llm(temperature: float = 0.4):
    from langchain_groq import ChatGroq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)


def search_ai_news(query: str, max_results: int = 10) -> list:
    from duckduckgo_search import DDGS
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


def summarize_articles(articles_text: str) -> str:
    llm = get_llm()
    prompt = (
        "Summarize each article below in 2-3 sentences. Keep the title and url for each. "
        "Output a numbered markdown list.\n\n" + articles_text
    )
    return llm.invoke(prompt).content


def generate_newsletter_html(summary_text: str) -> str:
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
    output_dir = os.path.join("/tmp", "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"newsletter_{timestamp}.html")
    if "<title>" not in body.lower():
        body = f"<title>{subject}</title>\n{body}"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(body)
    return filename
