from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

probabilities, class_probabilities, features, possible_values = {}, {}, [], {}


def train(data: pd.DataFrame):
    global probabilities, class_probabilities, features, possible_values

    target = data.columns[-1]
    features = data.columns[:-1].tolist()
    total_samples = len(data)

    # Calculate class probabilities
    class_probabilities = (data[target].value_counts() / total_samples).to_dict()

    # Get unique feature values
    possible_values = {feature: data[feature].unique().tolist() for feature in features}

    # Calculate feature probabilities
    probabilities = {
        feature: {
            f"{value},{class_label}": len(data[(data[feature] == value) & (data[target] == class_label)]) / len(
                data[data[target] == class_label])
            for value in data[feature].unique()
            for class_label in class_probabilities
        } for feature in features
    }

    return {
        "probabilities": probabilities,
        "class_probabilities": class_probabilities,
        "features": features,
        "possible_values": possible_values,
        "target": target
    }


def predict(input_data: Dict[str, str]):
    results = {}

    # Calculate the joint probability for each class
    for class_label in class_probabilities:
        joint_probability = class_probabilities[class_label]

        for feature, value in input_data.items():
            joint_probability *= probabilities[feature].get(f"{value},{class_label}", 1)  # Use 1 if missing

        results[class_label] = joint_probability

    # Calculate the total joint probability
    total = sum(results.values())

    # Check if total is 0
    if total == 0:
        # If total is 0, return the original results (they are all zero probabilities)
        return {class_label: 0 for class_label in results}

    # Normalize the probabilities if total > 0
    for class_label in results:
        results[class_label] /= total  # Divide each by the total to normalize

    return results


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_excel(file.file)
    return train(df)


@app.post("/predict")
async def predict_endpoint(input_data: Dict[str, str]):
    return predict(input_data)


@app.get("/model-info")
async def get_model_info():
    return {
        "probabilities": probabilities,
        "class_probabilities": class_probabilities,
        "features": features,
        "possible_values": possible_values
    }
