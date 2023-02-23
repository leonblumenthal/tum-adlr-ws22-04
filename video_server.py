"""Script to show videos on webpage that can be forwarded through ssh."""

import argparse
import time
from pathlib import Path

import cv2
from flask import Flask, Response, redirect

parser = argparse.ArgumentParser()
parser.add_argument("--runs_directory", type=Path, default="runs", help="Directory of training logs which includes the training videos.")
parser.add_argument("--port", type=int, default=8008)
args = parser.parse_args()
runs_directory = args.runs_directory


app = Flask(__name__)


def create_video_response(video_path: Path, fps: int = 300):
    def gen():
        last_time = time.perf_counter()

        cap = cv2.VideoCapture(str(video_path))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to viewable representation.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Try to hold fps.
            elapsed_time = time.perf_counter() - last_time
            if elapsed_time < 1 / fps:
                time.sleep(1 / fps - elapsed_time)
            last_time = time.perf_counter()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return redirect(f"http://127.0.0.1:{args.port}/{runs_directory}")


def file_ending(path: Path):
    return str(path).split(".")[-1]


def create_items(path: Path) -> list[str]:
    items = []
    for item_path in sorted(path.iterdir()):
        num_mp4s_inside = len(list(item_path.glob("**/*.mp4")))
        if file_ending(item_path) != "mp4" and num_mp4s_inside == 0:
            continue

        text = str(item_path.relative_to(path))
        if item_path.is_dir():
            text = f"{text} ({num_mp4s_inside})"

        items.append(f'<p><a href="http://127.0.0.1:8008/{item_path}">{text}</a></p>')

    return items


@app.route(f"/<path:path>", methods=["GET"])
def directory(path=str(runs_directory)):
    path = Path(path)

    if file_ending(path) == "mp4":
        return create_video_response(path)

    items = "".join(create_items(path))
    html = f"<!doctype html><html><body><h1>{path}</h1>{items}</body></html>" ""
    return html


app.run(debug=False, threaded=True, port=args.port)
