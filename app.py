import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


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
        from agent import start_agent
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
    try:
        from agent import start_agent
        state = start_agent(goal, mode, thread_id)
        if state.get("is_finished"):
            return jsonify({"status": "finished", "thread_id": thread_id})
        return jsonify({"status": "review", "thread_id": thread_id})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/approve/<thread_id>", methods=["POST"])
def approve(thread_id):
    try:
        from agent import approve_and_send
        state = approve_and_send(thread_id)
        return render_template("result.html", state=state, thread_id=thread_id)
    except Exception:
        return redirect(url_for("index"))


@app.route("/status/<thread_id>")
def status(thread_id):
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html"), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
