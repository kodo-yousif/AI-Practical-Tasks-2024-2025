from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

config = {
    "train_data_path": "../datasets/train",
    "val_data_path": "../datasets/val",
    "batch_size": 32,
    "image_size": (224, 224),
}

# Verify paths exist
for path in [config["train_data_path"], config["val_data_path"]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Path not found: {path}")


# Define the transformations for training and testing
train_transform = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(root=config["train_data_path"], transform=train_transform)
val_dataset = datasets.ImageFolder(root=config["val_data_path"], transform=val_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

# Temporary for visualization
temp_transform = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.ToTensor(),
])
temp_val_dataset = datasets.ImageFolder(root=config["val_data_path"], transform=temp_transform)
temp_val_loader = DataLoader(temp_val_dataset, batch_size=config["batch_size"], shuffle=False)

__all__ = ["train_loader", "val_loader", "temp_val_loader", "temp_val_dataset", "train_dataset"]