# ==============================================
# 2. EXPLORATION DES DONNÉES (EDA)
# ==============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Création dossier figures
os.makedirs("../figures", exist_ok=True)

df = pd.read_csv("data/processed/obesity_cleaned.csv")

print("=== Aperçu ===")
print(df.head())
print("\n=== Statistiques descriptives ===")
print(df.describe())

print("\n=== Distribution de la cible ===")
print(df['NObeyesdad'].value_counts(normalize=True).round(3) * 100)

# Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Distribution des classes
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='NObeyesdad', order=df['NObeyesdad'].value_counts().index)
plt.title("Distribution des niveaux d'obésité")
plt.tight_layout()
plt.savefig("figures/01_distribution_classes.png")
plt.show()

# 2. Distributions numériques
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df['Age'], kde=True, ax=axes[0]).set_title("Âge")
sns.histplot(df['Height'], kde=True, ax=axes[1]).set_title("Taille")
sns.histplot(df['Weight'], kde=True, ax=axes[2]).set_title("Poids")
plt.tight_layout()
plt.savefig("figures/02_distributions_numeriques.png")
plt.show()

# 3. Boxplot Weight vs Obésité
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Weight', y='NObeyesdad')
plt.title("Poids selon le niveau d'obésité")
plt.tight_layout()
plt.savefig("figures/03_boxplot_weight.png")
plt.show()

# 4. Heatmap de corrélation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("figures/04_correlation.png")
plt.show()

print("\nGraphiques sauvegardés dans le dossier figures/")