import argparse
import io
import mimetypes

from PIL import Image

import torch
from flask import Flask, request, send_file

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model([img], size=640)
        results.render()  # updates results.imgs with boxes and labels
        img_savename = f"static/123.png"
        #Image.fromarray(results.ims[0]).show()
        Image.fromarray(results.ims[0]).save(img_savename)
        return send_file(img_savename, mimetype=f"{mimetypes.guess_type(img_savename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model!")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', args.model)
    app.run(host="0.0.0.0", port=args.port)