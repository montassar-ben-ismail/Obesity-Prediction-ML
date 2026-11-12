# ==============================================
# 5. ÉVALUATION DES MODÈLES
# ==============================================

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import os

os.makedirs("figures", exist_ok=True)

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
le = joblib.load("models/label_encoder.pkl")

model_files = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Random Forest": "model_random_forest.pkl",
    "Gradient Boosting": "model_gradient_boosting.pkl",
    "XGBoost": "model_xgboost.pkl",
    "SVM": "model_svm.pkl"
}

results = {}

for name, filename in model_files.items():
    model = joblib.load(f"models/{filename}")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'accuracy': acc, 'f1': f1}
    
    print(f"\n{'='*55}")
    print(f"MODÈLE : {name}")
    print(f"{'='*55}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Matrice de confusion - {name}")
    plt.ylabel("Vraie classe")
    plt.xlabel("Classe prédite")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"figures/cm_{name.replace(' ', '_').lower()}.png")
    plt.show()

print("\n=== COMPARAISON FINALE ===")
for name, metrics in results.items():
    print(f"{name:25s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")