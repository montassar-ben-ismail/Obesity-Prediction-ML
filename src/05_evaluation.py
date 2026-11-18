# ==============================================
# 41. ENTRAINEMENT ET EVALUATION DES MODELES
# ==============================================

from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Chemins du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# Chargement des donnees
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


print("Dimensions des donnees :")
print(f"Train      : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# Definition des modeles
models = {
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
    "Decision_Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Extra_Tree": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
}


results = {}
trained_models = {}


# Entrainement et validation
for name, model in models.items():
    print(f"\n=== Entrainement de {name} ===")

    # Validation croisee uniquement sur les donnees d'entrainement
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

    # Entrainement sur les donnees train
    model.fit(X_train, y_train)

    # Evaluation sur validation
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Validation Accuracy : {val_accuracy:.4f}")

    # Sauvegarde du modele entraine sur train
    model_path = MODELS_DIR / f"model41_{name.lower()}.pkl"
    joblib.dump(model, model_path)

    results[name] = val_accuracy
    trained_models[name] = model


# Selection du meilleur modele avec l'ensemble validation
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n=== Resultats sur validation ===")

for name, score in results.items():
    print(f"{name:25s} : {score:.4f}")

print(f"\nMeilleur modele : {best_model_name}")
print(f"Validation Accuracy : {results[best_model_name]:.4f}")


# Reentrainement du meilleur modele sur train + validation
X_train_final = np.concatenate((X_train, X_val), axis=0)
y_train_final = np.concatenate((y_train, y_val), axis=0)

final_model = clone(models[best_model_name])
final_model.fit(X_train_final, y_train_final)


# Evaluation finale sur le test
y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n=== Evaluation finale sur le test ===")
print(f"Test Accuracy : {test_accuracy:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_test_pred))


# Sauvegarde du meilleur modele final
best_model_path = MODELS_DIR / "best_model_41.pkl"
joblib.dump(final_model, best_model_path)

print(f"Meilleur modele sauvegarde dans : {best_model_path}")