from typing import TypedDict, List
import os
import logging

log = logging.getLogger(__name__)


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


_graphs = {}


def _build_graph(interrupt_before=None):
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from tools import (
        search_ai_news,
        summarize_articles,
        generate_newsletter_html,
        split_subject_body,
        save_newsletter,
    )

    def plan_node(state):
        from tools import get_llm
        llm = get_llm()
        prompt = (
            "You are an autonomous newsletter agent. Break the following goal into a short "
            "numbered plan of 4-5 concrete steps (research, summarize, write, critique, send).\n\n"
            f"Goal: {state['goal']}"
        )
        plan = llm.invoke(prompt).content
        return {"plan": plan}

    def research_node(state):
        from tools import search_ai_news
        results = search_ai_news("latest AI agent news this week", max_results=10)
        return {"research_results": results}

    def summarize_node(state):
        from tools import summarize_articles
        articles = state["research_results"][:7]
        formatted = "\n\n".join(
            f"Title: {a['title']}\nSnippet: {a['body']}\nURL: {a['href']}" for a in articles
        )
        summary = summarize_articles(formatted)
        return {"articles_summary": summary}

    def write_node(state):
        from tools import generate_newsletter_html
        newsletter = generate_newsletter_html(state["articles_summary"])
        return {"newsletter_content": newsletter}

    def critique_node(state):
        from tools import get_llm
        llm = get_llm()
        prompt = (
            "Critique this newsletter draft for clarity, tone, accuracy and structure. "
            "List concrete, actionable issues.\n\n" + state["newsletter_content"]
        )
        critique = llm.invoke(prompt).content
        return {"critique": critique}

    def revise_node(state):
        from tools import get_llm
        llm = get_llm()
        prompt = (
            "Revise the newsletter draft below using the critique. Return only the improved final "
            "HTML newsletter, starting with a single line 'SUBJECT: <subject line>'.\n\n"
            f"Draft:\n{state['newsletter_content']}\n\nCritique:\n{state['critique']}"
        )
        revised = llm.invoke(prompt).content
        subject, body = split_subject_body(revised)
        return {"final_output": body, "subject": subject}

    def send_node(state):
        path = save_newsletter(state["subject"], state["final_output"])
        return {"sent": True, "saved_path": path}

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
    kwargs = {"checkpointer": checkpointer}
    if interrupt_before:
        kwargs["interrupt_before"] = interrupt_before
    return builder.compile(**kwargs)


def _get_graph(mode="autonomous"):
    key = "hitl" if mode == "human_in_loop" else "auto"
    if key not in _graphs:
        if key == "hitl":
            _graphs[key] = _build_graph(interrupt_before=["send"])
        else:
            _graphs[key] = _build_graph()
    return _graphs[key]


def start_agent(goal: str, mode: str, thread_id: str) -> dict:
    graph = _get_graph(mode)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "goal": goal,
        "mode": mode,
        "sent": False,
    }
    log.info("start_agent thread=%s mode=%s", thread_id, mode)
    graph.invoke(initial_state, config)
    return get_state(thread_id)


def approve_and_send(thread_id: str) -> dict:
    graph = _get_graph("human_in_loop")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
        log.info("approve_and_send thread=%s next=%s values_keys=%s", thread_id, snapshot.next, list(snapshot.values.keys()) if snapshot.values else None)
        if not snapshot.next:
            log.warning("approve_and_send thread=%s: graph already finished, nothing to resume", thread_id)
            return get_state(thread_id)
    except Exception:
        log.exception("approve_and_send thread=%s: get_state failed", thread_id)
    graph.invoke(None, config)
    return get_state(thread_id)


def get_state(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    for key in ("auto", "hitl"):
        graph = _get_graph("human_in_loop" if key == "hitl" else "autonomous")
        try:
            snapshot = graph.get_state(config)
            if snapshot.values:
                state = dict(snapshot.values)
                state["is_finished"] = len(snapshot.next) == 0
                return state
        except Exception:
            continue
    return {"is_finished": True}
