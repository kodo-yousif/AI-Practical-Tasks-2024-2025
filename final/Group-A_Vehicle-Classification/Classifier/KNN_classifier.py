import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KNNClassifierUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced KNN Vehicle Classifier")
        master.geometry("1400x800")  # Wider window for horizontal layout
        master.resizable(True, True)
        
        # Configure main grid
        master.columnconfigure((0, 1), weight=1)
        master.rowconfigure(0, weight=1)
        
        # Create left and right frames for horizontal layout
        self.left_frame = ttk.Frame(master, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        self.right_frame = ttk.Frame(master, padding="10")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure frame weights
        self.left_frame.columnconfigure(0, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        
        self._create_left_panel()
        self._create_right_panel()
        
        # Initialize variables
        self.clf = None
        self.scaler = StandardScaler()
        self.pca = None
        self.current_image = None
        self.feature_importance = None
        
    def _create_left_panel(self):
        # Dataset and Parameters Section
        params_frame = ttk.LabelFrame(self.left_frame, text="Dataset & Parameters", padding="10")
        params_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Dataset buttons
        dataset_frame = ttk.Frame(params_frame)
        dataset_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.dataset_button = ttk.Button(dataset_frame, text="Select Dataset", command=self.load_dataset)
        self.dataset_button.pack(side=tk.LEFT, padx=5)
        
        self.load_progress = ttk.Progressbar(dataset_frame, length=200, mode='determinate')
        self.load_progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Parameters
        params_grid = ttk.Frame(params_frame)
        params_grid.grid(row=1, column=0, sticky="ew", pady=5)
        
        # K neighbors selector
        ttk.Label(params_grid, text="Max K:").grid(row=0, column=0, padx=5)
        self.k_var = tk.StringVar(value="15")
        self.k_entry = ttk.Entry(params_grid, textvariable=self.k_var, width=8)
        self.k_entry.grid(row=0, column=1, padx=5)
        
        # PCA components
        ttk.Label(params_grid, text="PCA Components:").grid(row=0, column=2, padx=5)
        self.pca_var = tk.StringVar(value="50")
        self.pca_entry = ttk.Entry(params_grid, textvariable=self.pca_var, width=8)
        self.pca_entry.grid(row=0, column=3, padx=5)
        
        # Features selection
        features_frame = ttk.LabelFrame(params_frame, text="Features", padding=5)
        features_frame.grid(row=2, column=0, sticky="ew", pady=5)
        
        self.use_hog_var = tk.BooleanVar(value=True)
        self.use_color_var = tk.BooleanVar(value=True)
        self.use_sift_var = tk.BooleanVar(value=True)
        self.use_lbp_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(features_frame, text="HOG", variable=self.use_hog_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(features_frame, text="Color", variable=self.use_color_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(features_frame, text="SIFT", variable=self.use_sift_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(features_frame, text="LBP", variable=self.use_lbp_var).pack(side=tk.LEFT, padx=5)
        
        # Training section
        train_frame = ttk.LabelFrame(self.left_frame, text="Training", padding="10")
        train_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.train_button = ttk.Button(train_frame, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.grid(row=0, column=0, pady=5)
        
        self.train_progress = ttk.Progressbar(train_frame, length=200, mode='determinate')
        self.train_progress.grid(row=1, column=0, pady=5, sticky="ew")
        
        self.accuracy_label = ttk.Label(train_frame, text="", font=("TkDefaultFont", 12))
        self.accuracy_label.grid(row=2, column=0, pady=5)
        
        # Preview section
        preview_frame = ttk.LabelFrame(self.left_frame, text="Image Preview", padding="10")
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, width=300, height=300)
        self.preview_canvas.grid(row=0, column=0, pady=5)
        
    def _create_right_panel(self):
        # Results section
        results_frame = ttk.LabelFrame(self.right_frame, text="Results & Analysis", padding="10")
        results_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Prediction controls
        predict_frame = ttk.Frame(results_frame)
        predict_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.predict_button = ttk.Button(predict_frame, text="Select Prediction Folder", 
                                       command=self.predict_folder, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Results text
        self.result_text = tk.Text(results_frame, height=20, width=60)
        self.result_text.grid(row=1, column=0, sticky="nsew", pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, command=self.result_text.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.result_text['yscrollcommand'] = scrollbar.set
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self.right_frame, text="Visualizations", padding="10")
        viz_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create matplotlib figure for visualizations
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def extract_hog_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_size = (128, 128)
        resized = cv2.resize(gray, target_size)
        
        win_size = target_size
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        features = hog.compute(resized)
        return features.flatten()
    
    def extract_color_features(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        hist_params = {'bins': [32, 32, 32], 'range': [[0, 180], [0, 256], [0, 256]]}
        hsv_hist = [cv2.calcHist([hsv], [i], None, [hist_params['bins'][i]], 
                                [hist_params['range'][i][0], hist_params['range'][i][1]]) 
                   for i in range(3)]
        
        rgb_hist = [cv2.calcHist([rgb], [i], None, [32], [0, 256]) for i in range(3)]
        
        # Normalize and flatten
        all_hist = hsv_hist + rgb_hist
        normalized = [cv2.normalize(hist, hist).flatten() for hist in all_hist]
        return np.concatenate(normalized)
    
    def extract_sift_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return np.zeros(128)  # Return zero vector if no features found
            
        # Use average of all descriptors
        return np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(128)
    
    def extract_lbp_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        
        radius = 3
        n_points = 8 * radius
        
        def local_binary_pattern(image, n_points, radius):
            """Simple LBP implementation"""
            shape = image.shape
            output = np.zeros(shape)
            for i in range(radius, shape[0] - radius):
                for j in range(radius, shape[1] - radius):
                    value = 0
                    for p in range(n_points):
                        r = radius
                        x = i + r * np.cos(2 * np.pi * p / n_points)
                        y = j - r * np.sin(2 * np.pi * p / n_points)
                        if image[i, j] <= image[int(x), int(y)]:
                            value += 2**p
                    output[i, j] = value
            return output
        
        lbp = local_binary_pattern(resized, n_points, radius)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        return hist
    
    def extract_features(self, image):
        features = []
        
        if self.use_hog_var.get():
            hog_features = self.extract_hog_features(image)
            features.append(hog_features)
            
        if self.use_color_var.get():
            color_features = self.extract_color_features(image)
            features.append(color_features)
            
        if self.use_sift_var.get():
            sift_features = self.extract_sift_features(image)
            features.append(sift_features)
            
        if self.use_lbp_var.get():
            lbp_features = self.extract_lbp_features(image)
            features.append(lbp_features)
            
        if not features:
            raise ValueError("At least one feature type must be selected")
            
        return np.concatenate(features)
    
    def load_dataset(self):
        data_dir = filedialog.askdirectory(title="Select dataset directory")
        if not data_dir:
            return
            
        images = []
        labels = []
        image_paths = []  # Store paths for visualization
        
        self.load_progress['value'] = 0
        self.master.update_idletasks()
        
        try:
            class_dirs = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))]
            
            for i, class_dir in enumerate(class_dirs):
                class_path = os.path.join(data_dir, class_dir)
                files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for file in files:
                    image_path = os.path.join(class_path, file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        try:
                            features = self.extract_features(image)
                            images.append(features)
                            labels.append(class_dir)
                            image_paths.append(image_path)
                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")
                
                self.load_progress['value'] = (i + 1) / len(class_dirs) * 100
                self.master.update_idletasks()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
            self.load_progress['value'] = 0
            return
            
        if len(images) == 0:
            messagebox.showwarning("Warning", "No valid images found in selected directory.")
            return
            
        self.X = np.array(images)
        self.y = np.array(labels)
        self.image_paths = image_paths
        
        # Display dataset information
        self._update_dataset_info()
        
        # Enable training button
        self.train_button["state"] = tk.NORMAL
        
        # Show random image preview
        self._update_image_preview()
    
    def _update_dataset_info(self):
        unique_labels = np.unique(self.y)
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, f"Dataset Summary:\n{'-'*50}\n")
        self.result_text.insert(tk.END, f"Total samples: {len(self.y)}\n")
        self.result_text.insert(tk.END, f"Number of classes: {len(unique_labels)}\n")
        self.result_text.insert(tk.END, f"Classes: {', '.join(unique_labels)}\n")
        self.result_text.insert(tk.END, f"Feature vector size: {self.X.shape[1]}\n\n")
        
        # Show class distribution
        for label in unique_labels:
            count = np.sum(self.y == label)
            percentage = (count / len(self.y)) * 100
            self.result_text.insert(tk.END, f"{label}: {count} images ({percentage:.1f}%)\n")
    
    def _update_image_preview(self):
        if hasattr(self, 'image_paths') and self.image_paths:
            # Select random image
            idx = np.random.randint(0, len(self.image_paths))
            image_path = self.image_paths[idx]
            label = self.y[idx]
            
            # Load and resize image for preview
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (300, 300))
            
            # Convert to PhotoImage
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(150, 150, image=photo)
            self.preview_canvas.image = photo  # Keep reference
            self.preview_canvas.create_text(150, 280, text=f"Class: {label}", 
                                         font=("TkDefaultFont", 10, "bold"))
    
    def train_model(self):
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            messagebox.showerror("Error", "Please load a dataset first.")
            return
            
        self.train_button["state"] = tk.DISABLED
        self.predict_button["state"] = tk.DISABLED
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(self.X)
            
            # Apply PCA if specified
            n_components = int(self.pca_var.get())
            if n_components > 0:
                self.pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
                X_scaled = self.pca.fit_transform(X_scaled)
                explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
                self.result_text.insert(tk.END, f"\nPCA: {explained_var:.1f}% variance explained\n")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Create and train classifier
            max_k = int(self.k_var.get())
            self.clf = KNNClassifier(max_k=max_k)
            
            def progress_callback(progress):
                self.train_progress['value'] = progress
                self.master.update_idletasks()
            
            self.clf.train(X_train, y_train, progress_callback)
            
            # Evaluate on test set
            y_pred = self.clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Display results
            self.accuracy_label["text"] = f"Accuracy: {accuracy:.3f}"
            self.result_text.insert(tk.END, "\nTraining Results:\n" + "="*50 + "\n")
            self.result_text.insert(tk.END, f"Best parameters: {self.clf.best_params}\n\n")
            self.result_text.insert(tk.END, "Classification Report:\n")
            self.result_text.insert(tk.END, report)
            
            # Update visualizations
            self._plot_confusion_matrix(y_test, y_pred)
            
            self.predict_button["state"] = tk.NORMAL
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during training: {str(e)}")
        finally:
            self.train_button["state"] = tk.NORMAL
            self.train_progress['value'] = 100
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Clear previous plot
        self.fig.clear()
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        ax = self.fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def predict_folder(self):
        if self.clf is None:
            messagebox.showerror("Error", "Model not trained yet. Please train the model first.")
            return
            
        dir_path = filedialog.askdirectory(title="Select prediction directory")
        if not dir_path:
            return
            
        try:
            results = []
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        image = cv2.imread(file_path)
                        if image is not None:
                            features = self.extract_features(image)
                            features_scaled = self.scaler.transform(features.reshape(1, -1))
                            
                            if self.pca is not None:
                                features_scaled = self.pca.transform(features_scaled)
                            
                            prediction = self.clf.predict(features_scaled)
                            confidence = self.clf.predict_proba(features_scaled)
                            max_confidence = np.max(confidence)
                            results.append((file, prediction[0], max_confidence))
            
            # Display results
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "Prediction Results:\n" + "="*50 + "\n\n")
            for file, pred, conf in results:
                self.result_text.insert(tk.END, 
                    f"File: {file}\nClass: {pred}\nConfidence: {conf:.2f}\n{'-'*30}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")

class KNNClassifier:
    def __init__(self, max_k=15):
        self.model = None
        self.max_k = max_k
        self.best_params = None
        self.classes_ = None
        self.scaler = StandardScaler()
        
    def train(self, X, y, progress_callback=None):
        self.classes_ = np.unique(y)
        
        # Define parameter grid for optimization
        param_grid = {
            'n_neighbors': list(range(3, self.max_k + 1, 2)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2],
            'leaf_size': [10, 20, 30, 40]
        }
        
        base_clf = KNeighborsClassifier()
        
        grid_search = GridSearchCV(
            base_clf,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Track progress
        total_combinations = (len(param_grid['n_neighbors']) * 
                            len(param_grid['weights']) * 
                            len(param_grid['metric']) * 
                            len(param_grid['p']) *
                            len(param_grid['leaf_size']))
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
    def predict(self, X):
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        distances, indices = self.model.kneighbors(X)
        weights = 1 / (distances + 1e-6)
        normalized_weights = weights / weights.sum(axis=1, keepdims=True)
        return normalized_weights

if __name__ == "__main__":
    root = tk.Tk()
    app = KNNClassifierUI(root)
    root.mainloop()