import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import io
from model import ImageProcessor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

processor = ImageProcessor()


def img_to_base64(img):
    """Convert a BGR numpy array to a base64 PNG string."""
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def read_image_from_bytes(file_bytes):
    """Read image from bytes directly into numpy array (no disk save)."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Read directly from memory — no disk save
    file_bytes = file.read()
    img = read_image_from_bytes(file_bytes)

    if img is None:
        return jsonify({"error": "Cannot read image"}), 400

    processor.load_from_array(img)
    return jsonify({"image": img_to_base64(processor.get_current()), "status": "ok"})


@app.route("/reset", methods=["POST"])
def reset():
    img = processor.reset()
    if img is None:
        return jsonify({"error": "No image loaded"}), 400
    return jsonify({"image": img_to_base64(img), "status": "ok", "can_undo": processor.can_undo()})


@app.route("/undo", methods=["POST"])
def undo():
    img, success = processor.undo()
    if img is None:
        return jsonify({"error": "No image loaded"}), 400
    return jsonify({"image": img_to_base64(img), "status": "ok",
                    "success": success, "can_undo": processor.can_undo()})


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    operation = data.get("operation")
    value = int(data.get("value", 127))

    if processor.get_current() is None:
        return jsonify({"error": "No image loaded"}), 400

    ops = {
        # 1. Point Operations
        "addition": lambda: processor.addition(value),
        "subtraction": lambda: processor.subtraction(value),
        "division": lambda: processor.division(value),
        "complement": lambda: processor.complement(),
        # 2. Color Operations
        "change_red": lambda: processor.change_red_lighting(value),
        "swap_rg": lambda: processor.swap_r_to_g(),
        "eliminate_red": lambda: processor.eliminate_red(),
        # 3. Histogram
        "hist_stretch": lambda: processor.histogram_stretching_gray(),
        "hist_equalize": lambda: processor.histogram_equalization_gray(),
        # 4. Neighborhood
        "avg_filter": lambda: processor.average_filter(value),
        "laplacian": lambda: processor.laplacian_filter(value),
        "max_filter": lambda: processor.maximum_filter(value),
        "min_filter": lambda: processor.minimum_filter(value),
        "median_filter": lambda: processor.median_filter(value),
        "mode_filter": lambda: processor.mode_filter(value),
        # 5. Restoration
        "sp_average": lambda: processor.salt_pepper_average_filter(value),
        "sp_median": lambda: processor.salt_pepper_median_filter(value),
        "sp_outlier": lambda: processor.outlier_method(value),
        "gauss_averaging": lambda: processor.gaussian_image_averaging(value),
        "gauss_avg_filter": lambda: processor.gaussian_average_filter(value),
        "add_salt_pepper": lambda: processor.add_salt_pepper_noise(value),
        "add_gaussian": lambda: processor.add_gaussian_noise(value),
        # 6. Segmentation
        "basic_thresh": lambda: processor.basic_global_thresholding(value),
        "auto_thresh": lambda: processor.automatic_thresholding(),
        "adaptive_thresh": lambda: processor.adaptive_thresholding(value),
        # 7. Edge Detection
        "sobel": lambda: processor.sobel_detector(value),
        # 8. Morphology
        "dilation": lambda: processor.dilation(value),
        "erosion": lambda: processor.erosion(value),
        "opening": lambda: processor.opening(value),
        "closing": lambda: processor.closing(value),
        "internal_boundary": lambda: processor.internal_boundary(value),
        "external_boundary": lambda: processor.external_boundary(value),
        "morph_gradient": lambda: processor.morphological_gradient(value),
    }

    if operation not in ops:
        return jsonify({"error": f"Unknown operation: {operation}"}), 400

    try:
        img = ops[operation]()
        return jsonify({"image": img_to_base64(img), "status": "ok", "can_undo": processor.can_undo()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
