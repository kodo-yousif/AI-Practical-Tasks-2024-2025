# uvicorn main:app --reload 
# pip install -r requirements.txt

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

app = FastAPI()

# CORS setup
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelData:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.best_model = None
        self.best_model_name = None

model_data = ModelData()

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def create_nn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(32, input_dim=input_dim, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

class UserSample(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/train")
def train_models():
    try:
        # Load data
        df = pd.read_csv("diabetes.csv")
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Handle missing values (0 values in certain columns)
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            X.loc[X[col] == 0, col] = np.nan

        # Create and fit imputer
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Store preprocessors
        model_data.scaler = scaler
        model_data.imputer = imputer
        
        # Initialize and train models
        models = {
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'NaiveBayes': GaussianNB(),
            'NeuralNetwork': create_nn_model(X_train.shape[1])
        }
        
        results = []
        best_accuracy = 0
        
        for name, model in models.items():
            if isinstance(model, tf.keras.Sequential):
                model.fit(X_train, y_train, 
                         epochs=50, 
                         batch_size=32, 
                         validation_split=0.2,
                         verbose=0)
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            if isinstance(model, tf.keras.Sequential):
                y_pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Store model
            model_data.models[name] = model
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_data.best_model = model
                model_data.best_model_name = name
            
            results.append({
                "model": name,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm
            })
        
        return {
            "message": "Training complete",
            "best_model": model_data.best_model_name,
            "best_accuracy": round(best_accuracy, 4),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/confusion-matrix/{model_name}")
def get_model_confusion_matrix(model_name: str):
    if model_name not in model_data.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load test data
        df = pd.read_csv("diabetes.csv")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Preprocess test data
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            X.loc[X[col] == 0, col] = np.nan
        
        X_imputed = model_data.imputer.transform(X)
        X_scaled = model_data.scaler.transform(X_imputed)
        
        # Get predictions
        model = model_data.models[model_name]
        if isinstance(model, tf.keras.Sequential):
            y_pred = (model.predict(X_scaled) > 0.5).astype(int)
        else:
            y_pred = model.predict(X_scaled)
        
        cm = confusion_matrix(y, y_pred).tolist()
        
        return {
            "confusion_matrix": cm,
            "labels": ["Not Diabetic", "Diabetic"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_diabetes(sample: UserSample):
    if not model_data.models:
        raise HTTPException(status_code=400, detail="No trained models available. Please train first.")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([dict(sample)])
        
        # Handle zero values
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            input_data.loc[input_data[col] == 0, col] = np.nan
        
        # Preprocess input
        input_imputed = model_data.imputer.transform(input_data)
        input_scaled = model_data.scaler.transform(input_imputed)
        
        # Get prediction from best model
        model = model_data.best_model
        
        if isinstance(model, tf.keras.Sequential):
            probability = float(model.predict(input_scaled)[0][0])
            prediction = 1 if probability > 0.5 else 0
            confidence = probability if prediction == 1 else (1 - probability)
        else:
            prediction = int(model.predict(input_scaled)[0])
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = float(probabilities[prediction])
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_type": model_data.best_model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)