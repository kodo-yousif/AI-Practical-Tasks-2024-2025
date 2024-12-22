import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORSMiddleware to allow requests from all origins
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
        super(VehicleClassifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)

# Load the model
model = VehicleClassifier(num_classes=10)
model_path = "vehicle_classifier.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    logger.info("Model loaded successfully.")
else:
    logger.error("Model file not found.")
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class labels
class_names = ["Bus", "Family Sedan", "Fire Engine", "Heavy Truck", "Jeep",
               "Mini Bus", "Racing Car", "SUV", "Taxi", "Truck"]

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type")

        image = Image.open(file.file).convert("RGB")
        logger.info(f"Original image size: {image.size}")

        image = transform(image).unsqueeze(0)
        logger.info(f"Transformed image shape: {image.shape}")

        with torch.no_grad():
            output = model(image)
            logger.info(f"Model output: {output}")
            predicted_class = torch.argmax(output, dim=1)
            logger.info(f"Predicted class index: {predicted_class.item()}")

        class_name = class_names[predicted_class.item()]
        logger.info(f"Predicted class name: {class_name}")
        return JSONResponse(content={"class": class_name})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))