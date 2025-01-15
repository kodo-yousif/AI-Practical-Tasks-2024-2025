# pip install -r requirements.txt
# uvicorn main:app --reload

import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# sklearn and imblearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE


def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
set_seed(42)

app = FastAPI()

origins = ["http://localhost:5173",
    "http://127.0.0.1:5173",]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelData:
    def __init__(self):
        self.models = {}             # Dictionary to store all models
        self.scaler = None           # Store the scaler
        self.imputer = None          # Store the imputer
        self.best_model = None       # Best model object
        self.best_model_name = None  # Name of the best model

model_data = ModelData()


class UserSample(BaseModel):

    anxiety_level: float
    self_esteem: float
    mental_health_history: float
    depression: float
    headache: float
    blood_pressure: float
    sleep_quality: float
    breathing_problem: float
    noise_level: float
    living_conditions: float
    safety: float
    basic_needs: float
    academic_performance: float
    study_load: float
    teacher_student_relationship: float
    future_career_concerns: float
    social_support: float
    peer_pressure: float
    extracurricular_activities: float
    bullying: float


@app.post("/train")
def train_models():
    try:
        df = pd.read_csv("dataset.csv")

        target_column = 'stress_level'
        if target_column not in df.columns:
            raise ValueError(f"'{target_column}' column not found in dataset columns: {df.columns.tolist()}")

        if df[target_column].dtype == object:
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])

        X = df.drop(target_column, axis=1)
        y = df[target_column]

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

        best_svm = SVC(
            probability=True,
            random_state=42,
            C=1,
            kernel='rbf',
            gamma='auto'
        )
        best_svm.fit(X_train, y_train)

        best_knn = KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            algorithm='auto',
            p=2
        )
        best_knn.fit(X_train, y_train)

        best_nb = GaussianNB(var_smoothing=1e-9)
        best_nb.fit(X_train, y_train)

        mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='sgd',
            max_iter=200,
            random_state=42,
            early_stopping=True
        )
        mlp_model.fit(X_train, y_train)

        models = {
            'SVM': best_svm,
            'KNN': best_knn,
            'NaiveBayes': best_nb,
            'MLPClassifier': mlp_model
        }

        best_acc = 0.0
        best_model_name = None
        best_model_obj = None
        results = []

        for name, model_obj in models.items():
            y_pred = model_obj.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
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
            "message": "Training complete on Student Stress dataset",
            "best_model": best_model_name,
            "best_accuracy": round(best_acc, 4),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confusion-matrix/{model_name}")
def get_model_confusion_matrix(model_name: str):

    if model_name not in model_data.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        df = pd.read_csv("dataset.csv")
        target_column = 'stress_level'
        if target_column not in df.columns:
            raise ValueError(f"'{target_column}' column not found in dataset.")

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_imputed = model_data.imputer.transform(X)
        X_scaled = model_data.scaler.transform(X_imputed)

        model = model_data.models[model_name]
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y, y_pred).tolist()

        unique_labels = sorted(set(y))

        return {
            "confusion_matrix": cm,
            "labels": [f"Class {lbl}" for lbl in unique_labels]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_stress(sample: UserSample):
    """
    Predict the stress_level class (0, 1, or 2),
    using the best model determined by training.
    """
    if not model_data.models:
        raise HTTPException(
            status_code=400,
            detail="No trained models available. Please train first."
        )

    try:
        input_df = pd.DataFrame([dict(sample)])
        
        input_imputed = model_data.imputer.transform(input_df)
        input_scaled = model_data.scaler.transform(input_imputed)

        best_model = model_data.best_model
        best_model_name = model_data.best_model_name

        prediction = best_model.predict(input_scaled)[0]
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(input_scaled)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0

        prediction = int(prediction)

        return {
            "prediction": prediction,  # e.g. 0, 1, or 2
            "confidence": round(confidence, 4),
            "model_type": best_model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)