import os
import json
import uuid
import threading
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["GET", "POST"])
def run():
    if request.method == "GET":
        return redirect(url_for("index"))
    goal = request.form.get("goal", "").strip()
    mode = request.form.get("mode", "autonomous")
    if not goal:
        return redirect(url_for("index"))
    thread_id = str(uuid.uuid4())
    try:
        from agent import start_agent
        state = start_agent(goal, mode, thread_id)
        if state["is_finished"]:
            return render_template("result.html", state=state, thread_id=thread_id)
        return render_template("review.html", state=state, thread_id=thread_id)
    except Exception as e:
        log.exception("run failed for thread_id=%s", thread_id)
        return render_template("index.html"), 500


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(silent=True) or {}
    goal = (data.get("goal") or "").strip()
    mode = data.get("mode", "autonomous")
    if not goal:
        return jsonify({"error": "Goal is required."}), 400
    thread_id = str(uuid.uuid4())

    def _run():
        try:
            from agent import start_agent
            start_agent(goal, mode, thread_id)
        except Exception:
            log.exception("background agent failed for thread_id=%s", thread_id)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "thread_id": thread_id})


@app.route("/api/status/<thread_id>")
def api_status(thread_id):
    try:
        from agent import get_state
        state = get_state(thread_id)
        if state["is_finished"]:
            return jsonify({"status": "finished", "thread_id": thread_id})
        if state.get("final_output"):
            return jsonify({"status": "review", "thread_id": thread_id})
        if state.get("goal"):
            return jsonify({"status": "running", "thread_id": thread_id})
        return jsonify({"status": "running", "thread_id": thread_id})
    except Exception:
        return jsonify({"status": "running", "thread_id": thread_id})


@app.route("/approve/<thread_id>", methods=["POST"])
def approve(thread_id):
    try:
        from agent import approve_and_send
        state = approve_and_send(thread_id)
        return render_template("result.html", state=state, thread_id=thread_id)
    except Exception as e:
        log.exception("approve failed for thread_id=%s", thread_id)
        try:
            from agent import get_state
            state = get_state(thread_id)
            return render_template("review.html", state=state, thread_id=thread_id)
        except Exception:
            return redirect(url_for("index"))


@app.route("/status/<thread_id>")
def status(thread_id):
    try:
        from agent import get_state
        state = get_state(thread_id)
        if state["is_finished"]:
            return render_template("result.html", state=state, thread_id=thread_id)
        return render_template("review.html", state=state, thread_id=thread_id)
    except Exception:
        return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
