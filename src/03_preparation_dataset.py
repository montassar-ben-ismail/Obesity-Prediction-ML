# ==============================================
# 3. PRÉPARATION DU DATASET POUR LE ML
# Répartition : 70% train, 15% validation, 15% test
# ==============================================

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Création des dossiers
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# Chargement du dataset nettoyé
df = pd.read_csv("data/processed/obesity_cleaned.csv")


# Suppression de Height et Weight pour éviter le data leakage
df = df.drop(["Height", "Weight"], axis=1)

print("Colonnes utilisées :")
print(df.columns.tolist())


# Séparation des variables explicatives et de la cible
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]


# Encodage de la cible
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nClasses :")
print(dict(zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
)))


# Définition des colonnes
numeric_features = [
    "Age",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]

categorical_features = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]


# Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features,
        ),
    ]
)


# ==============================================
# Séparation 70% train, 15% validation, 15% test
# ==============================================

# Première séparation :
# 70% pour l'entraînement et 30% temporaire
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded,
)


# Deuxième séparation :
# La partie temporaire est divisée en deux parties égales :
# 15% validation et 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


# Transformation des données
# Le préprocesseur est ajusté uniquement sur les données d'entraînement
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)


# Sauvegarde du préprocesseur
joblib.dump(preprocessor, "models/preprocessor.pkl")


# Sauvegarde des datasets transformés
np.save("data/processed/X_train.npy", X_train_processed)
np.save("data/processed/X_val.npy", X_val_processed)
np.save("data/processed/X_test.npy", X_test_processed)

np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/y_test.npy", y_test)


# Affichage des dimensions
print("\nDimensions des datasets :")
print(f"Train      : {X_train_processed.shape}")
print(f"Validation : {X_val_processed.shape}")
print(f"Test       : {X_test_processed.shape}")

print("\nRépartition : 70% train, 15% validation, 15% test.")
print("Données préparées et sauvegardées avec succès.")