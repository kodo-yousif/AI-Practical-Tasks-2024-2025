import os
import cv2
import numpy as np
import joblib
import zipfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from fastapi.responses import JSONResponse, FileResponse
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Body
from fastapi.requests import Request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = ""
CATEGORIES = []
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

used_features = {
    'hsv': False,
    'lbp': False,
    'glcm': False
}

trained_models = {
    'mlp': None,
    'knn': None,
    'gnb': None,
    'svm': None
}

x, y = [], []

svm = SVC(C = 10, gamma =  'scale', kernel =  'linear')
knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 3, weights = 'distance')
gnb = GaussianNB(var_smoothing = 1e-09)
mlp = MLPClassifier(activation = 'tanh', alpha = 0.0001, hidden_layer_sizes = (32,), learning_rate = 'constant', max_iter = 1000, solver = 'adam')

def extract_features(img_path, img_size=(256, 256)):
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

@app.post("/load_data")
async def load_data(folder_path: str= Body(..., embed=True)):
    print(folder_path)
    try:
        global DATA_DIR, CATEGORIES
        DATA_DIR = folder_path
        CATEGORIES = os.listdir(DATA_DIR)
        loaded_x, loaded_y = [], []
        CATEGORIES.remove('.DS_Store')
        for idx, category in enumerate(CATEGORIES):
            folder = os.path.join(DATA_DIR, category)
            print(folder)
            for file in os.listdir(folder):
                filepath = os.path.join(folder, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    loaded_x.append(filepath)
                    loaded_y.append(idx)

        loaded_x = np.array(loaded_x)
        loaded_y = np.array(loaded_y)

        global x, y
        x, y = loaded_x, loaded_y

        return JSONResponse({
            "message": "Dataset loaded successfully.",
            "num_classes": len(CATEGORIES),
            "num_samples": len(loaded_y)
        })

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train(request: Request):
    data = await request.json()
    use_hsv = data.get('use_hsv')
    use_lbp = data.get('use_lbp')
    use_glcm = data.get('use_glcm')
    print(use_glcm, use_hsv, use_lbp)

    if not use_hsv and not use_lbp and not use_glcm:
        print("At least one feature must be selected")
        raise HTTPException(status_code=400, detail="At least one feature must be selected.")

    global used_features
    used_features['hsv'] = use_hsv
    used_features['lbp'] = use_lbp
    used_features['glcm'] = use_glcm

    try:
        global x, y

        extracted_features = []
        for img_path in x:
            features = extract_features(img_path)
            extracted_features.append(features)

        extracted_features = np.array(extracted_features)

        x_train, x_test, y_train, y_test = train_test_split(extracted_features, y, test_size=0.2, random_state=42, stratify=y)

        mlp.fit(x_train, y_train)
        knn.fit(x_train, y_train)
        gnb.fit(x_train, y_train)
        svm.fit(x_train, y_train)

        trained_models['mlp'] = mlp
        trained_models['knn'] = knn
        trained_models['gnb'] = gnb
        trained_models['svm'] = svm

        y_pred_mlp = mlp.predict(x_test)
        mlp_acc = accuracy_score(y_test, y_pred_mlp)

        y_pred_knn = knn.predict(x_test)
        knn_acc = accuracy_score(y_test, y_pred_knn)

        y_pred_gnb = gnb.predict(x_test)
        gnb_acc = accuracy_score(y_test, y_pred_gnb)

        y_pred_svm = svm.predict(x_test)
        svm_acc = accuracy_score(y_test, y_pred_svm)

        mlp_class_report = classification_report(y_test, y_pred_mlp, target_names=CATEGORIES, zero_division=0)
        knn_class_report = classification_report(y_test, y_pred_knn, target_names=CATEGORIES, zero_division=0)
        gnb_class_report = classification_report(y_test, y_pred_gnb, target_names=CATEGORIES, zero_division=0)
        svm_class_report = classification_report(y_test, y_pred_svm, target_names=CATEGORIES, zero_division=0)

        mlp_conf_matrix = confusion_matrix(y_test, y_pred_mlp).tolist()
        knn_conf_matrix = confusion_matrix(y_test, y_pred_knn).tolist()
        gnb_conf_matrix = confusion_matrix(y_test, y_pred_gnb).tolist()
        svm_conf_matrix = confusion_matrix(y_test, y_pred_svm).tolist()

        print(mlp_acc, knn_acc, gnb_acc, svm_acc)

        return JSONResponse({
            "message": "Training complete and evaluation finished.",
            "accuracy_scores": {
                "mlp": mlp_acc,
                "k_nn": knn_acc,
                "gnb": gnb_acc,
                "svm": svm_acc
            },
            "classification_reports": {
                "mlp": mlp_class_report,
                "k_nn": knn_class_report,
                "gnb": gnb_class_report,
                "svm": svm_class_report
            },
            "confusion_matrices": {
                "mlp": mlp_conf_matrix,
                "k_nn": knn_conf_matrix,
                "gnb": gnb_conf_matrix,
                "svm": svm_conf_matrix
            }
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        os.makedirs("temp", exist_ok=True)
        img_path = "temp/temp.jpg"
        cv2.imwrite(img_path, img)
        features = extract_features(img_path)
        os.remove(img_path)

        mlp_model = trained_models['mlp']
        mlp_prediction = mlp_model.predict([features])[0]

        knn_model = trained_models['knn']
        knn_prediction = knn_model.predict([features])[0]

        gnb_model = trained_models['gnb']
        gnb_prediction = gnb_model.predict([features])[0]

        svm_model = trained_models['svm']
        svm_prediction = svm_model.predict([features])[0]

        return JSONResponse({
            "predictions": {
                "mlp": CATEGORIES[mlp_prediction],
                "k_nn": CATEGORIES[knn_prediction],
                "gnb": CATEGORIES[gnb_prediction],
                "svm": CATEGORIES[svm_prediction]
            }
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save_models")
async def save_models():
    try:
        if not any(trained_models.values()):
            raise HTTPException(status_code=400, detail="No trained models to save.")

        for model_name, model in trained_models.items():
            if model:
                model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.joblib")
                joblib.dump(model, model_path)

        zip_path = os.path.join(MODEL_SAVE_DIR, "trained_models.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for model_name in trained_models.keys():
                model_file = os.path.join(MODEL_SAVE_DIR, f"{model_name}.joblib")
                if os.path.exists(model_file):
                    zipf.write(model_file, arcname=f"{model_name}.joblib")

        return FileResponse(zip_path, filename="trained_models.zip", media_type="application/zip")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/load_models")
async def load_models(file: UploadFile = File(...)):
    try:
        temp_zip_path = "temp_models.zip"
        with open(temp_zip_path, "wb") as f:
            f.write(await file.read())

        with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
            zipf.extractall(MODEL_SAVE_DIR)

        for model_name in trained_models.keys():
            model_file = os.path.join(MODEL_SAVE_DIR, f"{model_name}.joblib")
            if os.path.exists(model_file):
                trained_models[model_name] = joblib.load(model_file)

        os.remove(temp_zip_path)

        return JSONResponse({"message": "Models loaded successfully."})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
