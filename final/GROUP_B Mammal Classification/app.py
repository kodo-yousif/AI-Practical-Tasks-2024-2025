from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
import os
import json
import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configurations
UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'images[]' not in request.files:
        return jsonify({"error": "No files uploaded."}), 400

    uploaded_files = request.files.getlist('images[]')
    predictions = []

    try:
        # Load the model and label map
        model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'neural_model.keras'))
        with open(os.path.join(MODEL_FOLDER, 'neural_labelmap.json'), 'r') as f:
            label_map = json.load(f)

        for image in uploaded_files:
            filename = image.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            # Preprocess image
            img = Image.open(filepath).convert('RGB').resize((128, 128))
            img_array = np.array(img) / 255.0
            
            # Make prediction
            prediction = np.argmax(model.predict(img_array.reshape(1, 128, 128, 3)), axis=1)[0]
            probabilities = model.predict(img_array.reshape(1, 128, 128, 3))[0]
            confidence = float(probabilities[prediction])

            # Get class name
            class_name = [name for name, idx in label_map.items() if idx == prediction][0]

            predictions.append({
                "prediction": class_name,
                "filename": filename,
                "confidence": confidence
            })

            # Clean up uploaded file
            os.remove(filepath)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)