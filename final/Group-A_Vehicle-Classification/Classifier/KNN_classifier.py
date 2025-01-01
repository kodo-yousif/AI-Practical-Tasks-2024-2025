import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2

class KNNClassifierUI:
    def __init__(self, master):
        self.master = master
        master.title("KNN Vehicle Classifier")
        master.geometry("900x800")  # Increase initial window size
        master.resizable(True, True)  # Allow window resizing
        
        # Configure grid
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        
        # Create main frame
        self.frame = tk.Frame(master, padx=20, pady=20)
        self.frame.grid(sticky="nsew")
        
        # Configure frame grid
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure((1,2,3), weight=1)  # Allow rows to expand
        
        # Title label
        self.title_label = ttk.Label(self.frame, text="Vehicle Classifier", font=("TkDefaultFont", 18))
        self.title_label.grid(row=0, column=0, pady=(0,20))
        
        # Dataset selection
        self.dataset_frame = ttk.LabelFrame(self.frame, text="Dataset", padding=20)
        self.dataset_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.dataset_frame.columnconfigure(0, weight=1)
        
        self.dataset_button = ttk.Button(self.dataset_frame, text="Select Dataset", command=self.load_dataset)
        self.dataset_button.grid(row=0, column=0)
        
        self.load_progress = ttk.Progressbar(self.dataset_frame, length=200, mode='determinate')
        self.load_progress.grid(row=1, column=0, pady=(10,0))
        
        # Model training  
        self.model_frame = ttk.LabelFrame(self.frame, text="Model", padding=20)
        self.model_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.model_frame.columnconfigure(0, weight=1)
        
        self.train_button = ttk.Button(self.model_frame, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.grid(row=0, column=0)
        
        self.train_progress = ttk.Progressbar(self.model_frame, length=200, mode='determinate')
        self.train_progress.grid(row=1, column=0, pady=(10,0))
        
        self.accuracy_label = ttk.Label(self.model_frame, text="", font=("TkDefaultFont", 12))
        self.accuracy_label.grid(row=2, column=0, pady=(10,0))
        
        # Prediction
        self.predict_frame = ttk.LabelFrame(self.frame, text="Prediction", padding=20)
        self.predict_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        self.predict_frame.columnconfigure(0, weight=1)
        self.predict_frame.rowconfigure(1, weight=1)  # Allow result text to expand
        
        self.predict_button = ttk.Button(self.predict_frame, text="Select Prediction Folder", command=self.predict_folder, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=0)
        
        self.result_text = tk.Text(self.predict_frame, height=20, width=80)
        self.result_text.grid(row=1, column=0, pady=(10,0), sticky="nsew")
        
        # Add scrollbar to result text widget
        scrollbar = ttk.Scrollbar(self.predict_frame, command=self.result_text.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.result_text['yscrollcommand'] = scrollbar.set
        
        self.clf = KNNClassifier()
        
    def extract_hog_features(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Compute HOG features
        win_size = (64, 64)
        cell_size = (8, 8)
        block_size = (16, 16)
        hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, 9)
        resized = cv2.resize(gray, win_size)
        features = hog.compute(resized)
        return features.flatten()
        
    def load_dataset(self):
        data_dir = filedialog.askdirectory(title="Select dataset directory")
        
        images = []
        labels = []
        
        self.load_progress['value'] = 0
        self.master.update_idletasks()
        
        try:
            for i, (root, dirs, files) in enumerate(os.walk(data_dir)):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)
                        image = cv2.imread(image_path)
                        if image is not None:
                            features = self.extract_hog_features(image)
                            images.append(features)
                            label = os.path.basename(root)
                            labels.append(label)
                
                self.load_progress['value'] = (i+1) / len(list(os.walk(data_dir))) * 100
                self.master.update_idletasks()
        except KeyboardInterrupt:
            tk.messagebox.showwarning("Warning", "Loading interrupted by user.")
            self.load_progress['value'] = 0
            return
        
        if len(images) == 0:
            tk.messagebox.showwarning("Warning", "No images found in selected directory.")
            return
        
        self.X = np.array(images)
        self.y = np.array(labels)
        
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"Unique labels: {np.unique(self.y)}")
        
        self.train_button["state"] = tk.NORMAL
        
    def train_model(self):
        self.train_button["state"] = tk.DISABLED
        self.predict_button["state"] = tk.DISABLED
        
        self.train_progress['value'] = 0
        self.master.update_idletasks()
        
        # Normalize the feature data
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
        self.clf = KNNClassifier(n_neighbors=5)
        self.clf.train(X_train, y_train)
        
        y_pred = self.clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.accuracy_label["text"] = f"Accuracy: {accuracy:.3f}"
        self.predict_button["state"] = tk.NORMAL
        
    def predict_folder(self):
        if self.clf.model is None:
            tk.messagebox.showerror("Error", "Model not trained yet. Please train the model first.")
            return
        
        dir_path = filedialog.askdirectory(title="Select prediction directory")
        if not dir_path:
            return
        
        images = []
        file_paths = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    image = cv2.imread(file_path)
                    if image is not None:
                        features = self.extract_hog_features(image)
                        images.append(features)
                        file_paths.append(file_path)
        
        if len(images) == 0:
            tk.messagebox.showwarning("Warning", "No images found in selected directory.")
            return
        
        predictions = self.clf.predict(np.array(images))
        
        self.result_text.delete('1.0', tk.END)
        for file_path, prediction in zip(file_paths, predictions):
            filename = os.path.basename(file_path)
            result = f"{filename} => {prediction}\n"
            self.result_text.insert(tk.END, result)
        
        # Add scrollbar to result text widget
        scrollbar = ttk.Scrollbar(self.predict_frame, command=self.result_text.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.result_text['yscrollcommand'] = scrollbar.set
        
class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = None
        self.accuracy = 0
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
    def predict(self, X):
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        return self.model.predict(X)

root = tk.Tk()
gui = KNNClassifierUI(root)
root.mainloop() 