import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import glob
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import time


def save_features_to_csv(features, labels, filename):
    """Save features and labels to a CSV file."""
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(filename, index=False)
    print(f"Features saved to {filename}")


def load_features_from_csv(filename):
    """Load features and labels from a CSV file."""
    df = pd.read_csv(filename)
    labels = df['label'].values
    features = df.drop('label', axis=1).values
    return features, labels


def visualize_sample_images(images, labels, class_labels, num_samples=5):
    """Visualize a few sample images from the dataset."""
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(class_labels[labels[i]])
        plt.axis('off')
    plt.show()


def visualize_hog_features(img):
    """Visualize HOG features on an image."""
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    # If the image is already grayscale, skip the conversion
    if img.ndim == 3:  # 3 channels (BGR), need conversion
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Single channel (grayscale)
        img_gray = img

    h = hog.compute(img_gray)

    # Reshape and plot HOG features
    plt.imshow(img_gray, cmap='gray')
    plt.title("HOG Features Visualization")
    plt.show()



def visualize_lbp_features(img):
    """Visualize LBP features on an image."""
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')

    plt.imshow(lbp, cmap='gray')
    plt.title("LBP Features Visualization")
    plt.show()


def process_image(args):
    """Process a single image (helper for multiprocessing)."""
    img_path, class_idx = args
    try:
        # Normalize path separators for the operating system
        img_path = os.path.normpath(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        return img, class_idx
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None, None


def load_images_from_folder(folder, class_labels):
    """Load and process images from a folder using multiprocessing."""
    # Normalize the base folder path
    folder = os.path.normpath(folder)

    # Create list of image paths and their corresponding labels
    image_paths = []
    for idx, label in enumerate(class_labels):
        class_folder = os.path.join(folder, label)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder not found - {class_folder}")
            continue

        # Look for both .jpg and .jpeg files
        for ext in ['*.jpg', '*.jpeg']:
            pattern = os.path.join(class_folder, ext)
            image_paths.extend([(os.path.normpath(f), idx)
                                for f in glob.glob(pattern)])

    if not image_paths:
        raise ValueError(f"No images found in {folder}")

    print(f"Found {len(image_paths)} images...")

    # Process images in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths)))

    # Filter out None results and separate images and labels
    valid_results = [(img, label) for img, label in results if img is not None]

    if not valid_results:
        raise ValueError("No valid images were processed successfully")

    images, labels = zip(*valid_results)
    return np.array(images), np.array(labels)


def extract_features_parallel(images, feature_func):
    """Extract features from images using multiprocessing."""
    with mp.Pool(processes=mp.cpu_count()) as pool:
        features = list(tqdm(pool.imap(feature_func, images), total=len(images)))
    return np.array(features)


def extract_hog_features(img):
    """Extract Histogram of Oriented Gradients (HOG) features from an image."""
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return hog.compute(img).flatten()


def extract_lbp_features(image, radius=3, n_points=24):
    """Extract Local Binary Patterns (LBP) features from an image."""
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float') / (hist.sum() + 1e-7)
    return hist


def train_and_evaluate_model(model, X_train, X_val, y_train, y_val, model_name, class_labels):
    """Train a model and evaluate its performance."""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Evaluate and display results
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_val, y_pred, target_names=class_labels))

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.close()

    return model, accuracy


def main():
    # Use absolute paths or relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.normpath(os.path.join(script_dir, '..', 'datasets', 'train'))
    val_data_path = os.path.normpath(os.path.join(script_dir, '..', 'datasets', 'val'))

    # Verify paths exist
    if not os.path.exists(train_data_path):
        raise ValueError(f"Training data path not found: {train_data_path}")
    if not os.path.exists(val_data_path):
        raise ValueError(f"Validation data path not found: {val_data_path}")

    # Load class labels from existing directories only
    class_labels = [d for d in os.listdir(train_data_path)
                    if os.path.isdir(os.path.join(train_data_path, d))]
    print(f"Found classes: {class_labels}")

    # Check for precomputed features
    if os.path.exists('train_features.csv') and os.path.exists('val_features.csv'):
        print("Loading precomputed features...")
        X_train, y_train = load_features_from_csv('train_features.csv')
        X_val, y_val = load_features_from_csv('val_features.csv')
    else:
        # Process and load images
        print("Loading and processing images...")
        X_train, y_train = load_images_from_folder(train_data_path, class_labels)
        X_val, y_val = load_images_from_folder(val_data_path, class_labels)

        # Visualize a few sample images from the training dataset
        visualize_sample_images(X_train, y_train, class_labels)

        # Extract features (HOG + LBP)
        X_train_hog = extract_features_parallel(X_train, extract_hog_features)
        X_val_hog = extract_features_parallel(X_val, extract_hog_features)
        X_train_lbp = extract_features_parallel(X_train, extract_lbp_features)
        X_val_lbp = extract_features_parallel(X_val, extract_lbp_features)

        # Visualize HOG features for the first image in the training set
        visualize_hog_features(X_train[0])

        # Visualize LBP features for the first image in the training set
        visualize_lbp_features(X_train[0])

        # Combine and save features
        X_train = np.hstack((X_train_hog, X_train_lbp))
        X_val = np.hstack((X_val_hog, X_val_lbp))
        save_features_to_csv(X_train, y_train, 'train_features.csv')
        save_features_to_csv(X_val, y_val, 'val_features.csv')

    # Scale and reduce dimensionality of features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)

    # Define models
    models = {
        'SVM': GridSearchCV(svm.SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale'], 'kernel': ['rbf']}, cv=3),
        'KNN': GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
                            cv=3),
        'Naive Bayes': GaussianNB()
    }

    # Train and evaluate models
    for model_name, model in models.items():
        train_and_evaluate_model(model, X_train, X_val, y_train, y_val, model_name, class_labels)


if __name__ == "__main__":
    main()