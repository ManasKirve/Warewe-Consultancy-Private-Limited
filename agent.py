from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import (
    get_llm,
    search_ai_news,
    summarize_articles,
    generate_newsletter_html,
    split_subject_body,
    save_newsletter,
)


class AgentState(TypedDict):
    goal: str
    mode: str
    plan: str
    research_results: List[dict]
    articles_summary: str
    newsletter_content: str
    critique: str
    final_output: str
    subject: str
    sent: bool
    saved_path: str


def plan_node(state: AgentState) -> dict:
    llm = get_llm()
    prompt = (
        "You are an autonomous newsletter agent. Break the following goal into a short "
        "numbered plan of 4-5 concrete steps (research, summarize, write, critique, send).\n\n"
        f"Goal: {state['goal']}"
    )
    plan = llm.invoke(prompt).content
    return {"plan": plan}


def research_node(state: AgentState) -> dict:
    results = search_ai_news.invoke({"query": "latest AI agent news this week", "max_results": 10})
    return {"research_results": results}


def summarize_node(state: AgentState) -> dict:
    articles = state["research_results"][:7]
    formatted = "\n\n".join(
        f"Title: {a['title']}\nSnippet: {a['body']}\nURL: {a['href']}" for a in articles
    )
    summary = summarize_articles.invoke({"articles_text": formatted})
    return {"articles_summary": summary}


def write_node(state: AgentState) -> dict:
    newsletter = generate_newsletter_html.invoke({"summary_text": state["articles_summary"]})
    return {"newsletter_content": newsletter}


def critique_node(state: AgentState) -> dict:
    llm = get_llm()
    prompt = (
        "Critique this newsletter draft for clarity, tone, accuracy and structure. "
        "List concrete, actionable issues.\n\n" + state["newsletter_content"]
    )
    critique = llm.invoke(prompt).content
    return {"critique": critique}


def revise_node(state: AgentState) -> dict:
    llm = get_llm()
    prompt = (
        "Revise the newsletter draft below using the critique. Return only the improved final "
        "HTML newsletter, starting with a single line 'SUBJECT: <subject line>'.\n\n"
        f"Draft:\n{state['newsletter_content']}\n\nCritique:\n{state['critique']}"
    )
    revised = llm.invoke(prompt).content
    subject, body = split_subject_body(revised)
    return {"final_output": body, "subject": subject}


def send_node(state: AgentState) -> dict:
    path = save_newsletter(state["subject"], state["final_output"])
    return {"sent": True, "saved_path": path}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("write", write_node)
    builder.add_node("critique", critique_node)
    builder.add_node("revise", revise_node)
    builder.add_node("send", send_node)

    builder.set_entry_point("plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "summarize")
    builder.add_edge("summarize", "write")
    builder.add_edge("write", "critique")
    builder.add_edge("critique", "revise")
    builder.add_edge("revise", "send")
    builder.add_edge("send", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer, interrupt_before=["send"])


GRAPH = build_graph()


def start_agent(goal: str, mode: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "goal": goal,
        "mode": mode,
        "sent": False,
    }
    GRAPH.invoke(initial_state, config)
    if mode == "autonomous":
        GRAPH.invoke(None, config)
    return get_state(thread_id)


def approve_and_send(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    GRAPH.invoke(None, config)
    return get_state(thread_id)


def get_state(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = GRAPH.get_state(config)
    state = dict(snapshot.values)
    state["is_finished"] = len(snapshot.next) == 0
    return state
