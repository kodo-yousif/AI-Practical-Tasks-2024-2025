import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)
            output = self.fc(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            return predicted_class, probabilities.max().item()

# Model configuration
model_path = "vehicle_classifier_customcnn.pt"
image_size = (224, 224)
class_names = [
    "SUV", "Bus", "Family sedan", "Fire engine", "Heavy truck",
    "Jeep", "Minibus", "Racing car", "Taxi", "Truck"
]

# Check if the model file exists
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure the path is correct.")

# Initialize the model and load weights
num_classes = len(class_names)
model = VehicleClassifier(num_classes=num_classes)
state_dict = torch.load(model_path)

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("model.", "")  # Remove 'model.' prefix
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()  # Ensure the model is in evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")

    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Predict class
        with torch.no_grad():
            output = model(img_tensor)  # Directly call the model which invokes forward()
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        return {
            "predicted_class": class_names[predicted_class.item()],
            "probability": probabilities[0, predicted_class.item()].item()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
