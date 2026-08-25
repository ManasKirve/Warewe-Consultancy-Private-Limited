import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
from agent import start_agent, approve_and_send, get_state

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
    state = start_agent(goal, mode, thread_id)
    if state["is_finished"]:
        return render_template("result.html", state=state, thread_id=thread_id)
    return render_template("review.html", state=state, thread_id=thread_id)


@app.route("/approve/<thread_id>", methods=["POST"])
def approve(thread_id):
    state = approve_and_send(thread_id)
    return render_template("result.html", state=state, thread_id=thread_id)


@app.route("/status/<thread_id>")
def status(thread_id):
    state = get_state(thread_id)
    if state["is_finished"]:
        return render_template("result.html", state=state, thread_id=thread_id)
    return render_template("review.html", state=state, thread_id=thread_id)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
