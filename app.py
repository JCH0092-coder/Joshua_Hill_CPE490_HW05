from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the ONNX model globally
model_path = "mnist_cnn_model.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError("ONNX model not found. Please run Part 1.1 to generate mnist_cnn_model.onnx.")
ort_session = ort.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name

# The home route renders a template that shows the upload form and sample images.
@app.route('/')
def home():
    # This example assumes you have sample images in static/samples/0.png ... static/samples/9.png.
    sample_digits = [{"image": f"static/samples/{i}.png", "label": i} for i in range(10)]
    return render_template('index.html', sample_digits=sample_digits)

# The predict route accepts a POST request with an uploaded image file.
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the file and open it as an image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        # Resize to 28x28 if necessary, convert to grayscale ("L")
        if image.size != (28, 28):
            image = image.resize((28, 28))
        if image.mode != "L":
            image = image.convert("L")
        
        # Normalize pixel values to [0,1] and reshape to match model input dimensions
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Run inference using the already-loaded ONNX model
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
    # Running on debug mode; accessible at http://127.0.0.1:5000
    app.run(debug=True, host='0.0.0.0', port=5000)
