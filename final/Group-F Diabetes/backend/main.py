# filename: main.py
# Run with: uvicorn main:app --reload

import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import random
import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.config.experimental.enable_op_determinism()
    
set_seed(42)

app = FastAPI()

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelData:
    def __init__(self):
        self.models = {}             
        self.scaler = None           
        self.imputer = None          
        self.best_model = None      
        self.best_model_name = None  

model_data = ModelData()

def create_nn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
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
        df = pd.read_csv("diabetes.csv")  

        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            X.loc[X[col] == 0, col] = np.nan

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, 
            y_resampled, 
            test_size=0.2, 
            random_state=42,
            stratify=y_resampled
        )

        model_data.scaler = scaler
        model_data.imputer = imputer

        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        knn_params = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        nb_params = {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        svm = SVC(probability=True, random_state=42)
        svm_gs = GridSearchCV(
            estimator=svm,
            param_grid=svm_params,
            scoring='accuracy',
            cv=skf,
            n_jobs=1
        )
        svm_gs.fit(X_train, y_train)
        best_svm = svm_gs.best_estimator_

        knn = KNeighborsClassifier()
        knn_gs = GridSearchCV(
            estimator=knn,
            param_grid=knn_params,
            scoring='accuracy',
            cv=skf,
            n_jobs=-1
        )
        knn_gs.fit(X_train, y_train)
        best_knn = knn_gs.best_estimator_

        nb = GaussianNB()
        nb_gs = GridSearchCV(
            estimator=nb,
            param_grid=nb_params,
            scoring='accuracy',
            cv=skf,
            n_jobs=-1
        )
        nb_gs.fit(X_train, y_train)
        best_nb = nb_gs.best_estimator_

        nn_model = create_nn_model(X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

        nn_model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
            shuffle=False  
        )

        models = {
            'SVM': best_svm,
            'KNN': best_knn,
            'NaiveBayes': best_nb,
            'NeuralNetwork': nn_model
        }

        best_acc = 0.0
        best_model_name = None
        best_model_obj = None

        results = []

        for name, model_obj in models.items():
            if isinstance(model_obj, tf.keras.Sequential):
                y_pred_proba = model_obj.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model_obj.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred).tolist()

            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_model_obj = model_obj

            results.append({
                "model": name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm
            })

        model_data.models = models
        model_data.best_model = best_model_obj
        model_data.best_model_name = best_model_name

        return {
            "message": "Training complete with hyperparameter tuning + SMOTE",
            "best_model": best_model_name,
            "best_accuracy": round(best_acc, 4),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/confusion-matrix/{model_name}")
def get_model_confusion_matrix(model_name: str):
    """
    Return the confusion matrix for the specified model using the entire dataset.
    """
    if model_name not in model_data.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        df = pd.read_csv("diabetes.csv")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            X.loc[X[col] == 0, col] = np.nan

        X_imputed = model_data.imputer.transform(X)
        X_scaled = model_data.scaler.transform(X_imputed)

        model = model_data.models[model_name]

        if isinstance(model, tf.keras.Sequential):
            y_pred_proba = model.predict(X_scaled).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
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
        raise HTTPException(
            status_code=400,
            detail="No trained models available. Please train first."
        )

    try:
        input_df = pd.DataFrame([dict(sample)])

        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            input_df.loc[input_df[col] == 0, col] = np.nan

        input_imputed = model_data.imputer.transform(input_df)
        input_scaled = model_data.scaler.transform(input_imputed)

        best_model = model_data.best_model
        best_model_name = model_data.best_model_name

        if isinstance(best_model, tf.keras.Sequential):
            prob = best_model.predict(input_scaled).flatten()[0]
            prediction = 1 if prob > 0.5 else 0
            confidence = prob if prediction == 1 else (1 - prob)
        else:
            prediction = int(best_model.predict(input_scaled)[0])
            if hasattr(best_model, "predict_proba"):
                probabilities = best_model.predict_proba(input_scaled)[0]
                confidence = float(probabilities[prediction])
            else:
                confidence = 1.0

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_type": best_model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)