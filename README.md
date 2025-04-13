# Joshua_Hill_CPE490_HW05
# MNIST Digit Classifier

This is a Flask web application that uses an ONNX model to classify handwritten digit images (MNIST). Users can upload an image or select one of the provided sample images (digits 0–9) for classification.

To get started, open Git Bash (make sure you start it in the folder that contains your GitHub repository items) and clone the repository by running:  
`git clone https://github.com/JCH0092-coder/Joshua_Hill_CPE490_HW05.git`

Next, navigate to the project folder. For example, if your repository is on your Desktop and your Windows username is "Joshua", run:  
`cd /c/Users/Joshua/Desktop/Joshua_Hill_CPE490_HW05`  
(If your Windows username is different, replace "Joshua" with your actual username.)

Install the required dependencies by running:  
`py -m pip install flask onnxruntime numpy pillow`

Once the dependencies are installed, start the application by executing:  
`py app.py`  
The Flask server will start at [http://127.0.0.1:5000](http://127.0.0.1:5000). Open your web browser and go to this address to access the application.

To test the application, either use the file input to upload a digit image and click the Predict button, or click on one of the sample digit images (digits 0–9) to select it (which will clear any previously selected file) and then click the Predict button to classify the image using the ONNX model.

For a demonstration and further details on the ONNX model, please visit the Google Colab Notebook at:  
[https://colab.research.google.com/drive/1hZdsnSp-f5WnEp2Pekja4_f73E2x7hyR?usp=sharing](https://colab.research.google.com/drive/1hZdsnSp-f5WnEp2Pekja4_f73E2x7hyR?usp=sharing)

To update your local repository with the latest changes from GitHub, open Git Bash in the project folder and run:  
`git pull origin main`

This project uses several Python modules and packages, including Flask (imported as `from flask import Flask, render_template, request, jsonify, send_from_directory`), ONNX Runtime (imported as `import onnxruntime as ort`), NumPy (imported as `import numpy as np`), Pillow (imported as `from PIL import Image`), io (imported as `import io`), and os (imported as `import os`).
