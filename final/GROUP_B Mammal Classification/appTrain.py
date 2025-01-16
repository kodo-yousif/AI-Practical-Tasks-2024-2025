from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
from skimage import transform
import joblib
from PIL import Image, ImageEnhance
import numpy as np
import os
from werkzeug.utils import secure_filename
import time
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configurations
# Defines folders for uploads, saved models, and datasets. 
UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
DATASET_FOLDER = './mammals'
# Creates them if they don't exist.
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Returns the path to the saved model based on the type (e.g., neural network, KNN, etc.).
# Parameters: model_type (string).
# Return: Path as a string.
def get_model_path(model_type):
    if model_type == 'neural':
        return os.path.join(MODEL_FOLDER, f'{model_type}_model.keras')
    return os.path.join(MODEL_FOLDER, f'{model_type}_model.pkl')


# Performs image augmentation by:
# Rotating randomly.
# Adjusting brightness and contrast.
# Flipping horizontally.
# Cropping and resizing (zoom).
def apply_augmentation(img):
    augmented_images = []
    
    # Original image
    augmented_images.append(np.array(img)) 
    # [original_image, rotate1_image, rotate2_image, brightness_image1, brightness_image2, contrast_image1, contrast_image2, flipped_image[50% chance], zoomed]
    
    # Random rotation
    angles = np.random.uniform(-30, 30, 2)  # Generate 2 random angles between -30 and 30 degrees
    for angle in angles:
        rotated = img.rotate(angle, expand=False)
        augmented_images.append(np.array(rotated))
    
    # Random brightness and contrast adjustments
    enhancer = ImageEnhance.Brightness(img)
    brightness_factors = np.random.uniform(0.7, 1.3, 2)
    for factor in brightness_factors:
        brightened = enhancer.enhance(factor)
        augmented_images.append(np.array(brightened))
    
    enhancer = ImageEnhance.Contrast(img)
    contrast_factors = np.random.uniform(0.7, 1.3, 2)
    for factor in contrast_factors:
        contrasted = enhancer.enhance(factor)
        augmented_images.append(np.array(contrasted))
    
    # Random horizontal flip (50% chance)
    if np.random.random() > 0.5:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(np.array(flipped))
    
    # Random zoom (crop and resize)
    width, height = img.size
    crop_percent = np.random.uniform(0.8, 0.9)
    crop_width = int(width * crop_percent)
    crop_height = int(height * crop_percent)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2
    cropped = img.crop((left, top, right, bottom))
    zoomed = cropped.resize((width, height))
    augmented_images.append(np.array(zoomed))
    
    # Limit the number of augmented images based on original dataset size
    def limit_augmentations(dataset_size):
        if dataset_size < 100:
            return len(augmented_images)  # Keep all augmentations for very small datasets
        elif dataset_size < 1000:
            return min(5, len(augmented_images))  # Limit to 5 augmentations
        else:
            return min(3, len(augmented_images))  # Limit to 3 augmentations
    
    # Modify the load_dataset function to pass the dataset size
    dataset_size = sum(1 for _ in os.listdir(os.path.join(DATASET_FOLDER, next(os.walk(DATASET_FOLDER))[1][0])))
    max_augmentations = limit_augmentations(dataset_size)
    
    # Randomly select subset of augmentations if we have too many
    # Generates max_augmentations unique random indices from the range [0, len(augmented_images) - 1].
    # replace=False: Ensures no index is chosen more than once, so the selected indices are unique.
    if len(augmented_images) > max_augmentations:
        indices = np.random.choice(len(augmented_images), max_augmentations, replace=False)  # [2, 4, 6, 8, 3]
        # Creates a new list containing only the images at the randomly selected indices.
        augmented_images = [augmented_images[i] for i in indices]
    
    return augmented_images


# Loads and preprocesses images for training:
# Converts images to arrays (for neural models).
# Extracts HOG features (for non-neural models).
# Augments images.
# Balances classes by limiting samples per class.
# Emits loading progress via socketio.
# Parameters:
# dataset_path (string): Path to dataset directory.
# model_type (string): Type of model to use.
# Return: Features (X), labels (y), and label map (dictionary).
def load_dataset(dataset_path, model_type='neural'):
    X, y = [], []
    # labels: Contains the names of the subdirectories (e.g., ['cat', 'dog', 'bird']).
    labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
#     label_map: A dictionary mapping label names to integer indices.
#     Example: {'cat': 0, 'dog': 1, 'bird': 2}.
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    total_files = sum(len(files) for _, _, files in os.walk(dataset_path))
    processed_files = 0
    target_size = (128, 128)
    
    
    for label in labels:
        class_path = os.path.join(dataset_path, label)
        class_files = os.listdir(class_path)
        
        # Balance classes if needed for avoiding bias
        max_samples_per_class = min(len(os.listdir(os.path.join(dataset_path, l))) for l in labels)
        if len(class_files) > max_samples_per_class:
            # randomly choose subset
            class_files = np.random.choice(class_files, max_samples_per_class, replace=False)
        
        for img_file in class_files:
            img_path = os.path.join(class_path, img_file)
            
            if os.path.isfile(img_path):
                try:
                    original_img = Image.open(img_path).convert('RGB').resize(target_size)
                    augmented_images = apply_augmentation(original_img)
                    
                    for img_array in augmented_images:
                        if model_type == 'neural':
                            # if not normalized to 0 to 1 then it will be very large and takes a lot of time for cnn because cnn works with raw pixels
                            processed_img = img_array / 255.0
                        else:
                            # HOG requires a grayscale image
                            img_gray = Image.fromarray(img_array).convert('L')
                            features = hog(img_gray, 
                                        orientations=9,
                                        pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2),
                                        transform_sqrt=True,
                                        feature_vector=True)
                            processed_img = features
                        
                        X.append(processed_img) #[elephant1,elephant2,elephant3, fox1,fox2,fox3,fox4, lion1,lion2,lion3]  [(1,2),(2,3),(4,5)] samples
                        y.append(label_map[label]) #[elephant,elephant,elephant, fox, fox, fox, fox, lion, lion, lion] [-1, -1, 1] Ground truth
                    
                    processed_files += 1
                    if 'socketio' in globals():
                        socketio.emit('loading_progress', {
                            'percent': (processed_files / total_files) * 100,
                            'message': f'Loading dataset: {processed_files}/{total_files}'
                        })
                        
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    if model_type == 'neural':
        # reshaping the specific format need for neural network inputs because of 3d shape but we will transfer it to 4d shape
        # -1 This is a special value that tells the reshaping function to infer the size of this dimension based on the size of the other dimensions. It ensures that the total number of elements remains constant before and after reshaping. 
        X = X.reshape(-1, 128, 128, 3)
    
    return X, y, label_map


# Creates and returns a model based on model_type:
# KNN, Naive Bayes, SVM: Classic models from sklearn.
# Neural network: A custom convolutional neural network (CNN) built with TensorFlow.
# Parameters:
# model_type (string): Type of model.
# num_classes (int, optional): Number of output classes (for neural networks).
# Return: Model instance.
def get_model(model_type, num_classes=0):
    if model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=3)
    elif model_type == 'bayes':
        return GaussianNB()
    elif model_type == 'svm':
        return SVC(kernel='rbf', probability=True)
    elif model_type == 'neural':
        # Specifies the input shape for a neural network.
        inputs = Input(shape=(128, 128, 3))

        # First Convolutional block
        # 32 (3,3) filter applying to this layer, relu means it's multi classifier
        x = Conv2D(32, (3, 3), activation='relu', padding='same', 
                  kernel_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x) # best feature extraction
        x = Dropout(0.25)(x)

        # Second Convolutional Block
        x = Conv2D(64, (3, 3), activation='relu', padding='same', 
                  kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Third Convolutional Block
        x = Conv2D(128, (3, 3), activation='relu', padding='same', 
                  kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Flattening and Fully Connected Layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # Output Layer
        outputs = Dense(num_classes, activation='softmax')(x)

        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    else:
        raise ValueError("Invalid model type")



# Evaluates model performance using metrics like precision, recall, and F1-score.
# Generates ROC curves for each class.
# Parameters:
# y_test (array): True labels.
# y_pred (array): Predicted labels.
# y_pred_proba (array): Predicted probabilities (if available).
# classes (list): Class labels.
# Return: Dictionary of metrics and ROC curves.
def calculate_metrics(y_test, y_pred, y_pred_proba, classes):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    

    # Calculate accuracy, precision, recall, and F1 score
    precision = precision_score(y_test, y_pred, average='weighted')  # or 'micro', 'macro', 'samples'
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calculate ROC curve and AUC for each class
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    roc_curves = []
    
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_curves.append({
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)  # Convert numpy float to Python float
        })
    
    return {
        'confusion_matrix': cm.tolist(),
        'roc_curves': roc_curves,
        'classes': list(classes),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

@app.route('/')
def index():
    return render_template('indexTrain.html')


# @app.route('/train', methods=['POST'])
# Trains the model:
# Loads the dataset.
# Splits it into training and test sets.
# Trains the model (classic or neural).
# Saves the model and label map.
# Reports training progress using socketio.
# Request Data:
# model_type (JSON key): Type of model to train.
# Response: JSON with metrics or error message.
@app.route('/train', methods=['POST'])
def train_model():
    model_type = request.json.get('model_type')
    
    try:
        # Pass model_type to load_dataset
        X, y, label_map = load_dataset(DATASET_FOLDER, model_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'neural':
            num_classes = len(label_map)
            model = get_model(model_type, num_classes=num_classes)
            y_train_cat = to_categorical(y_train, num_classes)
            
            history = model.fit(
                X_train, y_train_cat,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                callbacks = [
                            EarlyStopping(
                                monitor='val_loss',
                                patience=5,
                                restore_best_weights=True
                            ),
                            ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                min_lr=1e-6
                            ),
                            tf.keras.callbacks.LambdaCallback(
                                on_epoch_end=lambda epoch, logs: socketio.emit('training_progress', {
                                    "percent": ((epoch + 1) / 20) * 100,
                                    "message": f"Epoch {epoch + 1}/20"
                                })
                            )
                        ]
            )
            
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_pred_proba = model.predict(X_test)
            
            model.save(get_model_path(model_type))
            label_map_path = os.path.join(MODEL_FOLDER, 'neural_labelmap.json')
            with open(label_map_path, 'w') as f:
                json.dump(label_map, f)
            
        else:
            model = get_model(model_type)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            joblib.dump((model, label_map), get_model_path(model_type))

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, list(label_map.keys()))

        return jsonify({
            "message": "Model trained successfully!",
            "accuracy": accuracy,
            "report": report,
            "metrics": metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



# Predicts the class of uploaded images using a trained model:
# Loads the model and label map.
# Processes uploaded images.
# Makes predictions.
# Request Data:
# images[] (files): List of image files.
# modelType (form data): Type of model to use.
# Response: JSON with predictions or error message.
@app.route('/predict', methods=['POST'])
def predict():
    if 'images[]' not in request.files:
        return jsonify({"error": "No files uploaded."}), 400

    model_type = request.form.get('modelType')
    model_path = get_model_path(model_type)

    if not os.path.exists(model_path):
        return jsonify({"error": f"Model not found. Please train the {model_type} model first."}), 400

    uploaded_files = request.files.getlist('images[]')
    predictions = []

    try:
        if model_type == 'neural':
            model = tf.keras.models.load_model(model_path)
            label_map_path = os.path.join(MODEL_FOLDER, 'neural_labelmap.json')
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
        else:
            model, label_map = joblib.load(model_path)

        for image in uploaded_files:
            filename = image.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # saving the file to process for prediction 
            image.save(filepath)

            img = Image.open(filepath).convert('RGB').resize((128, 128))
            img_array = np.array(img) / 255.0 # Normalize for neural model cnn 0 to 1
            
            if model_type == 'neural':
                prediction = np.argmax(model.predict(img_array.reshape(1, 128, 128, 3)), axis=1)[0]
                probabilities = model.predict(img_array.reshape(1, 128, 128, 3))[0]
                confidence = float(probabilities[prediction])
            else:
                features = hog(img.convert('L'), 
                             orientations=9,
                             pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2),
                             transform_sqrt=True,
                             feature_vector=True).reshape(1, -1)
                prediction = model.predict(features)[0]
                confidence = float(model.predict_proba(features).max()) if hasattr(model, 'predict_proba') else None

            class_name = [name for name, idx in label_map.items() if idx == prediction][0]

            prediction_result = {
                "prediction": class_name,
                "filename": filename
            }
            
            if confidence is not None:
                prediction_result["confidence"] = confidence

            predictions.append(prediction_result)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5002)