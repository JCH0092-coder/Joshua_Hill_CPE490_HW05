from flask import Flask, render_template, request, jsonify, send_from_directory
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
model_path = "mnist_cnn_model.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError("ONNX model not found.")
ort_session = ort.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/samples/<filename>')
def sample_file(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route('/predict', methods=['POST'])
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
        outputs = ort_session.run(None, {input_name: img_array})
        probabilities = outputs[0][0].tolist()
        predicted_digit = int(np.argmax(probabilities))
        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence_scores": probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask web server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
