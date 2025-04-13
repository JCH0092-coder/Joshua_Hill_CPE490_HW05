from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Ensure the ONNX model file "mnist_cnn_model.onnx" is in the same directory.
model_path = "mnist_cnn_model.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError("ONNX model not found. Please ensure 'mnist_cnn_model.onnx' is present in the working directory.")
    
ort_session = ort.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."})
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if not already.
        if image.mode != "L":
            image = image.convert("L")
        
        # Resize the image to 28x28.
        image = image.resize((28, 28))
        
        # Normalize pixel values to [0, 1].
        img_array = np.array(image).astype("float32") / 255.0
        
        # Reshape the image to match the expected input shape: (1, 28, 28, 1)
        input_data = img_array.reshape(1, 28, 28, 1)
        
        # Run inference using the ONNX model.
        outputs = ort_session.run(None, {input_name: input_data})
        
        # Assuming outputs[0] holds the confidence scores for each digit.
        predictions = outputs[0][0]  
        predicted_digit = int(np.argmax(predictions))
        confidence_scores = predictions.tolist()
        
        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence_scores": confidence_scores
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
