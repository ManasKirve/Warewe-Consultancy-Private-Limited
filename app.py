import os
import json
import uuid
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv
from agent import start_agent, approve_and_send, get_state

load_dotenv()

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_state(thread_id, state, error=None):
    record = dict(state) if state else {}
    if error:
        record["error"] = str(error)
    path = os.path.join(OUTPUT_DIR, f"thread_{thread_id}.json")
    try:
        with open(path, "w") as f:
            json.dump(record, f)
    except Exception:
        pass


def _load_state(thread_id):
    path = os.path.join(OUTPUT_DIR, f"thread_{thread_id}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _run_agent_bg(goal, mode, thread_id):
    try:
        state = start_agent(goal, mode, thread_id)
        _save_state(thread_id, state)
    except Exception as e:
        _save_state(thread_id, {}, error=e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    goal = request.form.get("goal", "").strip()
    mode = request.form.get("mode", "autonomous")
    if not goal:
        return redirect(url_for("index"))
    thread_id = str(uuid.uuid4())
    try:
        state = start_agent(goal, mode, thread_id)
        if state["is_finished"]:
            return render_template("result.html", state=state, thread_id=thread_id)
        return render_template("review.html", state=state, thread_id=thread_id)
    except Exception as e:
        return render_template("index.html"), 500


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(silent=True) or {}
    goal = (data.get("goal") or "").strip()
    mode = data.get("mode", "autonomous")
    if not goal:
        return jsonify({"error": "Goal is required."}), 400
    thread_id = str(uuid.uuid4())
    t = threading.Thread(target=_run_agent_bg, args=(goal, mode, thread_id), daemon=True)
    t.start()
    return jsonify({"thread_id": thread_id})


@app.route("/status/<thread_id>")
def status(thread_id):
    state = _load_state(thread_id)
    if request.args.get("json") == "1":
        if state is None:
            return jsonify({"status": "running"})
        if state.get("error"):
            return jsonify({"status": "error", "error": state["error"]})
        if state.get("is_finished"):
            return jsonify({"status": "finished"})
        return jsonify({"status": "running"})
    if state and state.get("is_finished"):
        return render_template("result.html", state=state, thread_id=thread_id)
    if state and not state.get("error"):
        return render_template("review.html", state=state, thread_id=thread_id)
    return redirect(url_for("index"))


@app.route("/approve/<thread_id>", methods=["POST"])
def approve(thread_id):
    try:
        state = approve_and_send(thread_id)
        _save_state(thread_id, state)
        return render_template("result.html", state=state, thread_id=thread_id)
    except Exception:
        return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html"), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
