# =================================================================================
#           1. PRÉPARATION DES LABELS & SUPPRESSION DES LIGNES NULL
# ================================================================================

import pandas as pd
import os

# Chemins
RAW_PATH = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
PROCESSED_PATH = "data/processed/obesity_cleaned.csv"

# Création du dossier processed s'il n'existe pas
os.makedirs("../data/processed", exist_ok=True)

# Chargement
df = pd.read_csv(RAW_PATH)

print("=== Informations initiales ===")
print(f"Shape initial : {df.shape}")
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Suppression des lignes nulles
df_clean = df.dropna()
print(f"\nShape après suppression des nulls : {df_clean.shape}")
print(f"Lignes supprimées : {df.shape[0] - df_clean.shape[0]}")

# Nettoyage des labels
df_clean['NObeyesdad'] = df_clean['NObeyesdad'].str.strip()

print("\n=== Distribution des classes ===")
print(df_clean['NObeyesdad'].value_counts())

# Sauvegarde
df_clean.to_csv(PROCESSED_PATH, index=False)
print(f"\nDataset nettoyé sauvegardé : {PROCESSED_PATH}")