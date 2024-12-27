from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data preprocessing
def preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]   # Target

        # Normalize the dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

# KNN classifier
def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

# SVM classifier
def train_svm(X_train, y_train):
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    return svm

# Neural Network with TensorFlow
def train_nn_tf(X_train, y_train):
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(os.getcwd(), file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(file_path)

        # Train models
        knn = train_knn(X_train, y_train)
        svm = train_svm(X_train, y_train)
        nn_tf = train_nn_tf(X_train, y_train)

        # Generate results
        results = {
            "KNN": classification_report(y_test, knn.predict(X_test), output_dict=True),
            "SVM": classification_report(y_test, svm.predict(X_test), output_dict=True),
        }

        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {e}"})