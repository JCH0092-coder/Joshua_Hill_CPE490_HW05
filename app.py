import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import onnxruntime as rt
from PIL import Image
import io

app = Flask(__name__)

# Home route: render the UI (index.html)
@app.route("/")
def home():
    sample_digits = [{"image": f"static/samples/{i}.png", "label": i} for i in range(10)]
    return render_template("index.html", sample_digits=sample_digits)

# Predict route: process uploaded image and run model inference
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.size != (28, 28):
            image = image.resize((28, 28))
        if image.mode != "L":
            image = image.convert("L")
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn_model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError("ONNX model not found. Run Part 1.1 first.")

        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_array})
        probs = outputs[0][0].tolist()
        pred_digit = int(np.argmax(probs))
        return jsonify({"predicted_digit": pred_digit, "confidence_scores": probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
