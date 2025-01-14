import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


DATA_DIR = "/Users/oneaboveall/Downloads/MangoLeafBD"  # Replace with your dataset path
CATEGORIES = [
    "Bacterial Canker",
    "Anthracnose",
    "Healthy",
    "Powdery Mildew",
    "Cutting Weevil",
    "Die Back",
    "Sooty Mould",
    "Gall Midge"
]
used_features = {'hsv': True, 'lbp': True, 'glcm': True}

models = {
    'mlp': MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(32,),
                          learning_rate='constant', max_iter=1000, solver='adam'),
    'knn': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance'),
    'gnb': GaussianNB(var_smoothing=1e-9),
    'svm': SVC(C=10, gamma='scale', kernel='linear'),
}

def extract_features(img_path, img_size=(240, 240)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    features = []

    if used_features['hsv']:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
        hist_h /= (hist_h.sum() + 1e-7)
        hist_s /= (hist_s.sum() + 1e-7)
        hist_v /= (hist_v.sum() + 1e-7)
        features.extend([hist_h, hist_s, hist_v])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if used_features['lbp']:
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
        features.append(lbp_hist)

    if used_features['glcm']:
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, prop='contrast').flatten()
        correlation = graycoprops(glcm, prop='correlation').flatten()
        energy = graycoprops(glcm, prop='energy').flatten()
        homogeneity = graycoprops(glcm, prop='homogeneity').flatten()
        features.extend([contrast, correlation, energy, homogeneity])

    return np.concatenate(features)

x, y = [], []
for idx, category in enumerate(CATEGORIES):
    category_path = os.path.join(DATA_DIR, category)
    for file in os.listdir(category_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(category_path, file)
            x.append(filepath)
            y.append(idx)

extracted_features = [extract_features(img_path) for img_path in x]
extracted_features = np.array(extracted_features)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(extracted_features, y, test_size=0.2, random_state=42, stratify=y)

metrics = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

labels = list(models.keys())
x = np.arange(len(labels))
width = 0.2

accuracy_scores = [metrics[name]["accuracy"] for name in labels]
precision_scores = [metrics[name]["precision"] for name in labels]
recall_scores = [metrics[name]["recall"] for name in labels]
f1_scores = [metrics[name]["f1"] for name in labels]

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - 1.5 * width, accuracy_scores, width, label='Accuracy', color='blue')
bar2 = ax.bar(x - 0.5 * width, precision_scores, width, label='Precision', color='orange')
bar3 = ax.bar(x + 0.5 * width, recall_scores, width, label='Recall', color='green')
bar4 = ax.bar(x + 1.5 * width, f1_scores, width, label='F1-Score', color='red')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics by Model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_values(bar1)
add_values(bar2)
add_values(bar3)
add_values(bar4)

plt.tight_layout()
plt.show()
