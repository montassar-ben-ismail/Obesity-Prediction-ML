# ==============================================
# 4. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ==============================================

from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Chemins du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# Chargement des données
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


print("Dimensions des données :")
print(f"Train      : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# Définition des modèles
models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient_Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
}


results = {}
trained_models = {}


# Entraînement et validation
for name, model in models.items():
    print(f"\n=== Entraînement de {name} ===")

    # Validation croisée uniquement sur les données d'entraînement
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(
        f"CV Accuracy : {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )

    # Entraînement sur les données train
    model.fit(X_train, y_train)

    # Évaluation sur validation
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Validation Accuracy : {val_accuracy:.4f}")

    # Sauvegarde du modèle entraîné sur train
    model_path = MODELS_DIR / f"model_{name.lower()}.pkl"
    joblib.dump(model, model_path)

    results[name] = val_accuracy
    trained_models[name] = model


# Sélection du meilleur modèle avec l'ensemble validation
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n=== Résultats sur validation ===")

for name, score in results.items():
    print(f"{name:25s} : {score:.4f}")

print(f"\nMeilleur modèle : {best_model_name}")
print(f"Validation Accuracy : {results[best_model_name]:.4f}")


# Réentraînement du meilleur modèle sur train + validation
X_train_final = np.concatenate((X_train, X_val), axis=0)
y_train_final = np.concatenate((y_train, y_val), axis=0)

final_model = clone(models[best_model_name])
final_model.fit(X_train_final, y_train_final)


# Évaluation finale sur le test
y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n=== Évaluation finale sur le test ===")
print(f"Test Accuracy : {test_accuracy:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_test_pred))


# Sauvegarde du meilleur modèle final
best_model_path = MODELS_DIR / "best_model.pkl"
joblib.dump(final_model, best_model_path)

print(f"Meilleur modèle sauvegardé dans : {best_model_path}")