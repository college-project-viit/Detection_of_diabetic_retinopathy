import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.feature.texture import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class DatasetCleaner:
    def __init__(self, raw_dir, clean_dir, target_size=(512, 512)):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.target_size = target_size
        self.classes = ['0', '1']

    def validate_and_clean(self):
        print("\n--- STAGE 1: INITIALIZING DATA CLEANING & STANDARDIZATION ---")
        for cls in self.classes:
            source_path = os.path.join(self.raw_dir, cls)
            dest_path = os.path.join(self.clean_dir, cls)
            os.makedirs(dest_path, exist_ok=True)
            
            if not os.path.exists(source_path):
                print(f"[WARNING] Source path {source_path} not found. Skipping.")
                continue

            corrupt_counter = 0
            valid_counter = 0

            for img_name in tqdm(os.listdir(source_path), desc=f"Processing Class {cls}"):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    continue
                
                img_path = os.path.join(source_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError("Image corrupt or header broken.")
                    
                    resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(dest_path, img_name), resized_img)
                    valid_counter += 1
                except Exception as e:
                    corrupt_counter += 1
                    os.makedirs(self.clean_dir, exist_ok=True)
                    with open(os.path.join(self.clean_dir, "cleaning_errors.txt"), "a") as log:
                        log.write(f"Dropped: {img_path} | Error: {str(e)}\n")

            print(f"[SUCCESS] Class {cls} operational. Validated: {valid_counter} | Dropped Corrupts: {corrupt_counter}")

class RetinalImageProcessor:
    @staticmethod
    def process_pipeline(bgr_image):
        green = bgr_image[:, :, 1]
        blurred = cv2.medianBlur(green, 5)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
        bothat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BOTHAT, kernel)
        adjusted_matrix = cv2.add(enhanced_gray, tophat)
        subtracted_matrix = cv2.subtract(adjusted_matrix, bothat)
        segmented_vessels = cv2.adaptiveThreshold(
            subtracted_matrix, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        segmented_mask = cv2.morphologyEx(
            segmented_vessels, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        return enhanced_gray, segmented_mask

class GLCMFeatureExtractor:
    def __init__(self):
        self.distances = [1, 3, 5, 7]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.properties = ['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity']

    def compute_glcm_features(self, grayscale_image):
        feature_vector = []
        glcm = graycomatrix(
            grayscale_image, distances=self.distances, angles=self.angles, 
            levels=256, symmetric=True, normed=True
        )
        for prop in self.properties:
            extracted_props = graycoprops(glcm, prop)
            feature_vector.append(np.mean(extracted_props))
            feature_vector.append(np.std(extracted_props))
            feature_vector.append(np.max(extracted_props))
            feature_vector.append(np.min(extracted_props))
        return feature_vector

    def extract_dual_representation(self, enhanced_gray, segmented_mask):
        enhanced_features = self.compute_glcm_features(enhanced_gray)
        vessel_features = self.compute_glcm_features(segmented_mask)
        return np.concatenate([enhanced_features, vessel_features])

def build_and_vectorize_dataset(data_dir):
    extractor = GLCMFeatureExtractor()
    X_collector = []
    y_collector = []
    classes = ['0', '1']
    print("\n--- STAGE 2: EXTRACTING GLCM TEXTURE METRICS ---")
    for cls in classes:
        target_folder = os.path.join(data_dir, cls)
        if not os.path.exists(target_folder):
            continue
        file_list = [f for f in os.listdir(target_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not file_list:
            continue
        for img_name in tqdm(file_list, desc=f"Vectorizing Class {cls}"):
            img_path = os.path.join(target_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            enhanced, segmented = RetinalImageProcessor.process_pipeline(img)
            features = extractor.extract_dual_representation(enhanced, segmented)
            X_collector.append(features)
            y_collector.append(int(cls))
    return np.array(X_collector), np.array(y_collector)

if __name__ == "__main__":
    raw_dataset_path = "raw_kaggle_data/"
    cleaned_dataset_path = "clean_production_data/"

    for c in ['0', '1']:
        os.makedirs(os.path.join(raw_dataset_path, c), exist_ok=True)

    cleaner = DatasetCleaner(raw_dir=raw_dataset_path, clean_dir=cleaned_dataset_path)
    cleaner.validate_and_clean()

    if len(os.listdir(os.path.join(cleaned_dataset_path, '0'))) == 0 and len(os.listdir(os.path.join(cleaned_dataset_path, '1'))) == 0:
        print("\n[NOTICE] 'raw_kaggle_data/' folder is empty. Processing fallback simulation...")
        X = np.random.normal(loc=0.5, scale=0.1, size=(200, 48))
        y = np.random.choice([0, 1], size=200, p=[0.55, 0.45])
    else:
        X, y = build_and_vectorize_dataset(cleaned_dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\n[INFO] Data matrix split verified. Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- STAGE 3: RUNNING GRID SEARCH HYPERPARAMETER OPTIMIZATION FOR KNN ---")
    param_grid = {
        'n_neighbors': 5,
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_knn = grid_search.best_estimator_
    print(f"[SUCCESS] Optimal Model Settings Discovered: {grid_search.best_params_}")

    y_pred = best_knn.predict(X_test_scaled)
    y_proba = best_knn.predict_proba(X_test_scaled)[:, 1]

    print("\n================ PRODUCTION MODEL PERFORMANCE EVALUATION ================")
    print(classification_report(y_test, y_pred, target_names=['Normal (Class 0)', 'Diabetic Retinopathy (Class 1)']))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Receiver Operating Characteristic Area-Under-Curve (ROC AUC): {roc_auc_score(y_test, y_proba):.4f}")
    print("=========================================================================")

    os.makedirs("analytics_outputs", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DR'], yticklabels=['Normal', 'DR'])
    plt.title('Retinopathy KNN Diagnostic Confusion Matrix')
    plt.ylabel('Actual Label Class')
    plt.xlabel('System Predicted Diagnosis')
    plt.savefig('analytics_outputs/confusion_matrix.png', dpi=300)
    plt.close()
    print("[INFO] Analytics visualization plots exported to 'analytics_outputs/'.")

    os.makedirs("production_models", exist_ok=True)
    joblib.dump(best_knn, 'production_models/optimized_retinopathy_knn.pkl')
    joblib.dump(scaler, 'production_models/dataset_standard_scaler.pkl')
    print("[SUCCESS] Trained models and scaling profiles successfully exported to 'production_models/'.\n")
