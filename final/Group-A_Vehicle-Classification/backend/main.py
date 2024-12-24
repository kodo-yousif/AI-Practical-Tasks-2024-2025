import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model
class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

# Load model
model_path = "vehicle_classifier.pt"
model = VehicleClassifier(num_classes=10)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    logger.info("Model loaded successfully.")
else:
    logger.error("Model file not found.")
    raise RuntimeError("Model file not found. Please ensure 'vehicle_classifier.pt' is present.")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class labels
class_names = [
    "SUV", "Family Sedan", "Fire Engine", "Heavy Truck", "Jeep",
    "Mini Bus", "Racing Car", "Bus", "Taxi", "Truck"
]

@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPEG or PNG.")

    try:
        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Return prediction result
        return {"class": class_names[predicted_class]}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image.")
