"""Flask frontend for the Classroom Voice Analytics pipeline.

An alternative to the Streamlit app (../app.py): a plain HTML/CSS/JS page
talking to Flask over Server-Sent Events (SSE) for real-time upload
progress and a live-updating transcript. SSE rather than a raw WebSocket
is enough here since updates only ever flow server -> browser; a WebSocket
would pull in Flask-SocketIO plus an async worker (eventlet/gevent) for no
functional gain on a one-directional stream, whereas SSE needs nothing
beyond Flask's own dev server.

The actual transcription/diarization/metrics pipeline lives in src/ and is
shared with app.py - this module is only the web layer on top of
src.pipeline.process_audio_streaming().
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline import list_cached, load_cached, process_audio_streaming

LIVE_MODE_MAX_SECONDS = 240  # match the Streamlit demo's hosted-compute cap

app = Flask(__name__)

# In-memory job registry: job_id -> Queue of SSE events. Fine for a
# single-process demo; a job is created on upload and torn down once its
# SSE stream finishes (or the client disconnects) - there's no persistence
# or multi-worker sharing, so this won't survive a process restart or scale
# past one worker without moving the queue to something like Redis.
_jobs: dict[str, "queue.Queue"] = {}
_jobs_lock = threading.Lock()


def _run_job(job_id: str, audio_path: str, language: str | None) -> None:
    q = _jobs[job_id]
    try:
        for partial in process_audio_streaming(
            audio_path,
            model_size=config.WHISPER_MODEL_SIZE_FAST,
            language=language or None,
            max_duration_sec=LIVE_MODE_MAX_SECONDS,
        ):
            q.put({"type": "partial", "data": partial})
    except Exception as exc:  # surface pipeline failures to the browser instead of hanging the stream
        q.put({"type": "error", "message": str(exc)})
    finally:
        q.put({"type": "done"})
        try:
            os.unlink(audio_path)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template(
        "index.html",
        samples=list_cached(),
        fast_model=config.WHISPER_MODEL_SIZE_FAST,
        max_minutes=LIVE_MODE_MAX_SECONDS // 60,
    )


@app.route("/api/samples/<name>")
def get_sample(name: str):
    result = load_cached(name)
    if result is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "no file uploaded"}), 400

    language = request.form.get("language") or None
    suffix = Path(file.filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = queue.Queue()

    threading.Thread(target=_run_job, args=(job_id, tmp_path, language), daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    q = _jobs.get(job_id)
    if q is None:
        return jsonify({"error": "unknown or already-finished job"}), 404

    def generate():
        try:
            while True:
                event = q.get()
                if event["type"] == "done":
                    yield "event: done\ndata: {}\n\n"
                    break
                if event["type"] == "error":
                    # Custom event name, deliberately not "error" - EventSource
                    # reserves that name for connection-level failures, and an
                    # SSE event literally named "error" collides with it.
                    yield f"event: pipeline_error\ndata: {json.dumps({'message': event['message']})}\n\n"
                    break
                yield f"data: {json.dumps(event['data'])}\n\n"
        finally:
            # Runs on normal completion AND on client disconnect
            # (Werkzeug raises GeneratorExit into this generator), so the
            # job is always cleaned up either way.
            with _jobs_lock:
                _jobs.pop(job_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # threaded=True lets the SSE connection stay open while other requests
    # (e.g. another upload) are served concurrently. use_reloader is off:
    # Werkzeug's reloader restarts the app in a second process on file
    # changes (a subprocess on Windows, since it can't fork), and that
    # second process would have its own separate, empty _jobs dict -
    # in-memory state like this can't survive a reload anyway.
    app.run(debug=False, threaded=True, port=5000, use_reloader=False)
