from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import random

app = FastAPI()

# Allow web browser to access our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve our HTML/CSS/JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Simple KNN class
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.data_points = None  # Store training data points
        self.data_labels = None  # Store training data labels
    
    # Calculate distance between two points
    def calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    # Save training data
    def train(self, points, labels):
        self.data_points = points
        self.data_labels = labels
    
    # Predict class for a new point
    def predict(self, point, use_sampling=False, sample_size=None):
        if use_sampling:
            # Use only a subset of training data
            random_indices = random.sample(range(len(self.data_points)), sample_size)
            points_to_use = self.data_points[random_indices]
            labels_to_use = self.data_labels[random_indices]
        else:
            # Use all training data
            points_to_use = self.data_points
            labels_to_use = self.data_labels
        
        # Calculate distances to all points
        distances = [self.calculate_distance(point, p) for p in points_to_use]
        
        # Find k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = labels_to_use[nearest_indices]
        
        # Return most common label
        return int(max(set(nearest_labels), key=list(nearest_labels).count))

# Store our model and data
knn_model = None
training_points = None
training_labels = None
cluster_centers = None

# Generate random training data
def create_random_data(num_points, num_classes):
    # Create random points
    points = np.random.randn(num_points, 2) * 2
    
    # Create random cluster centers
    centers = np.random.randn(num_classes, 2) * 4
    
    # Assign each point to nearest center
    labels = []
    for point in points:
        distances_to_centers = [np.sqrt(np.sum((point - center) ** 2)) for center in centers]
        nearest_center = np.argmin(distances_to_centers)
        labels.append(nearest_center)
    
    return points, np.array(labels), centers

# Calculate statistics for each class
def get_class_stats(points, labels):
    stats = {}
    unique_labels = set(labels)
    
    for label in unique_labels:
        # Get points belonging to this class
        class_points = points[labels == label]
        
        stats[int(label)] = {
            "count": len(class_points),
            "center": np.mean(class_points, axis=0).tolist(),
            "std": np.std(class_points, axis=0).tolist()
        }
    
    return stats

# API endpoint to generate new data
@app.post("/api/generate-data")
async def generate_data(request: dict):
    global knn_model, training_points, training_labels, cluster_centers
    
    # Create new random data
    training_points, training_labels, cluster_centers = create_random_data(
        request["n_samples"],
        request["n_classes"]
    )
    
    # Create and train KNN model
    knn_model = KNN(k=3)
    knn_model.train(training_points, training_labels)
    
    # Calculate statistics
    stats = get_class_stats(training_points, training_labels)
    
    # Return data for visualization
    return {
        "points": training_points.tolist(),
        "labels": training_labels.tolist(),
        "centers": cluster_centers.tolist(),
        "stats": stats
    }

# API endpoint to make predictions
@app.post("/api/predict")
async def predict(request: dict):
    if knn_model is None:
        return {"error": "Please generate data first"}
    
    # Get point to predict
    point = np.array(request["point"])
    
    # Update k value
    knn_model.k = request["k"]
    
    # Make predictions using both methods
    traditional_prediction = knn_model.predict(point, use_sampling=False)
    sampling_prediction = knn_model.predict(
        point,
        use_sampling=True,
        sample_size=request["sample_size"]
    )
    
    return {
        "traditional_prediction": traditional_prediction,
        "sampled_prediction": sampling_prediction
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 