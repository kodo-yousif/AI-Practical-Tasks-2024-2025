from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import List
from joblib import Parallel, delayed
import math
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
data = None
scaler = None
trained_models = {}
class TrainRequest(BaseModel):
    validation_method: str
class PredictRequest(BaseModel):
    data_row: List[float]
    model_type: str

class DiabetesNN(nn.Module):
    def __init__(self, input_size):
        super(DiabetesNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        
        # Increase numbers if your model is underfitting (not learning well).
        # Decrease numbers if your model is overfitting (learning too well on training data but failing on unseen data) or if training is too slow.
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def preprocess_data():
    global scaler

    processed_data = data.copy()
    X = processed_data.iloc[:, :-1].values
    y = processed_data.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def load_data():
    global data
    try:
        data = pd.read_csv('./diabetes.csv')
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def load_model(model_name):
    try:
        if model_name == "Neural":
            model = DiabetesNN(input_size=8)
            model.load_state_dict(torch.load(f"{model_name}.pth"))
            return model
        else:
            with open(f"{model_name}.pkl", "rb") as f:
                return joblib.load(f)
    except FileNotFoundError:
        return None
    
def save_model(model, model_name):
    if isinstance(model, DiabetesNN):
        torch.save(model.state_dict(), f"{model_name}.pth")
    else:
        with open(f"{model_name}.pkl", "wb") as f:
            joblib.dump(model, f)

def train_neural_network(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = (test_outputs > 0.5).float().numpy()
        probabilities = test_outputs.numpy()

    return predictions, probabilities

def evaluate_model(model, X_test, y_test):
    try:
        is_pytorch_model = isinstance(model, nn.Module)
        
        if is_pytorch_model:
            model.eval()
            with torch.no_grad():
                outputs = model(torch.FloatTensor(X_test))
                predictions = (outputs > 0.5).float().numpy().flatten()
        else:
            predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0)
        }

        return metrics

    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        return {metric: 0.0 for metric in ["accuracy", "precision", "recall", "f1_score"]}

@app.on_event("startup")
async def startup_event():
    if not load_data():
        raise HTTPException(status_code=500, detail="Failed to load initial data")

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    try:
        X, y = preprocess_data()
        validation_method = request.validation_method

        models = {
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "Bayesian": GaussianNB(),
            "SVM": SVC(probability=True),
            "Neural": DiabetesNN(input_size=8)
        }

        results = {}
        best_model = {"name": None, "metric": 0.0, "reason": ""}
        
        metric_weights = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0
        }

        if validation_method == "holdout":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for name, model in models.items():
                if isinstance(model, DiabetesNN):
                    predictions, _ = train_neural_network(model, X_train, y_train, X_test, y_test)
                else:
                    model.fit(X_train, y_train)
                
                metrics = evaluate_model(model, X_test, y_test)
                results[name] = {
                    key: float(value) if not math.isnan(float(value)) else 0.0
                    for key, value in metrics.items()
                }

                combined_score = sum(metrics[key] * metric_weights[key] for key in metrics)
                if combined_score > best_model["metric"]:
                    best_model = {
                        "name": name,
                        "metric": combined_score,
                        "reason": f"Highest combined score based on all metrics"
                    }
                
                save_model(model, name)

        elif validation_method in ["3-fold", "10-fold"]:
            n_splits = 3 if validation_method == "3-fold" else 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for name, model in models.items():
                metrics_sum = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                }
                valid_folds = 0

                for train_index, test_index in kf.split(X):
                    X_train_cv, X_test_cv = X[train_index], X[test_index]
                    y_train_cv, y_test_cv = y[train_index], y[test_index]

                    if isinstance(model, DiabetesNN):
                        model = DiabetesNN(input_size=8) 
                        predictions, _ = train_neural_network(model, X_train_cv, y_train_cv, X_test_cv, y_test_cv)
                    else:
                        model.fit(X_train_cv, y_train_cv)
                    
                    fold_metrics = evaluate_model(model, X_test_cv, y_test_cv)

                    for key in metrics_sum:
                        if fold_metrics[key] is not None and not math.isnan(fold_metrics[key]):
                            metrics_sum[key] += fold_metrics[key]
                            valid_folds += 1

                results[name] = {
                    key: float(metrics_sum[key] / valid_folds if valid_folds > 0 else 0.0)
                    for key in metrics_sum
                }

                combined_score = sum(results[name][key] * metric_weights[key] for key in results[name])
                if combined_score > best_model["metric"]:
                    best_model = {
                        "name": name,
                        "metric": combined_score,
                        "reason": f"Highest combined score based on all metrics"
                    }

                save_model(model, name)

        else:
            raise HTTPException(status_code=400, detail="Invalid validation method")


        return results,best_model

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    
    try:
        model = load_model(request.model_type)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_type} not found. Please train the model first."
            )

        if scaler is None:
            _, _ = preprocess_data()
            
        data_row_scaled = scaler.transform([request.data_row])
        
        if request.model_type == "Neural":

            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data_row_scaled)
                output = model(input_tensor)
                prediction = int(output.item() > 0.5)
                probability = [1 - output.item(), output.item()]
        else:
            prediction = model.predict(data_row_scaled)[0]
            probability = model.predict_proba(data_row_scaled)[0].tolist() if hasattr(model, 'predict_proba') else None
            
        response = {
            "prediction": int(prediction),
            "probability": probability
        }
        
        return response

    except Exception as e:
        if "is not fitted yet" in str(e):
            print(e)
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_type} needs to be trained first. Please call /train endpoint."
            )
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

