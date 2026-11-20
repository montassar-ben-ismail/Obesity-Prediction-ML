#permet de construire des chemins compatibles windows/linux
from pathlib import Path

#limite un champ a une liste de valeurs autorisées
from typing import Literal

#recharge les objets python sauvgardés dans les fichiers .pkl
import joblib

import pandas as pd

#framework qui crée l'API HTTP
from fastapi import FastAPI

# BaseModel: décrit et valide le json recu
# ConfigDict: permet de configurer le model
# Field: ajoute des contraintes numeriques (min,max...)
from pydantic import BaseModel, ConfigDict, Field



BASE_DIR = Path(__file__).resolve().parent.parent
# __file__           => .../projet_obesity/api/main.py
# .resolve()         => transforme en chemin absolu
# .parent            => .../projet_obesity/api
# .parent            => .../projet_obesity

MODELS_DIR = BASE_DIR/"models"
#dossier qui contient preprocessor.pkl , label_encoder.pkl et best_model.pkl

#fournir exactement les mm noms de features que le dataset

FEATURES = [
    "Gender",
    "Age",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

LABEL_FR={
    "Insufficient_Weight": "Insuffisance pondérale",
    "Normal_Weight": "Poids normal",
    "Overweight_Level_I": "Surpoids niveau I",
    "Overweight_Level_II": "Surpoids niveau II",
    "Obesity_Type_I": "Obésité type I",
    "Obesity_Type_II": "Obésité type II",
    "Obesity_Type_III": "Obésité type III",
}

#chargement du modele

preprocessor = joblib.load(MODELS_DIR/"preprocessor.pkl")
label_encoder = joblib.load(MODELS_DIR/"label_encoder.pkl")
model = joblib.load(MODELS_DIR/"best_model_41.pkl")

#liste des classes dans l'ordre utilisé par le labelEncoder
#exemple : CLASSES[1]="Normal_Weight"
CLASSES= [str(c) for c in label_encoder.classes_]

#schema d'entré
class ObesityInput(BaseModel):

    #extra="forbid" : si le client envoie un champ inconnu
    #fastapi renvoie une erreur 422 au lieu de l'ignorer silencieusement
    model_config = ConfigDict(extra="forbid")

    #les champs catégoriels
    #Literal impose une valeur parmi celles du dataset
    Gender:Literal["Female","Male"]
    family_history_with_overweight:Literal["yes","no"]
    FAVC:Literal["yes","no"]
    CAEC:Literal["no","Sometimes","Frequently","Always"]
    SMOKE:Literal["yes","no"]
    SCC:Literal["yes","no"]
    CALC:Literal["no","Sometimes","Frequently","Always"]
    MTRANS:Literal["Automobile","Bike","Motorbike","Public_Transportation","Walking"]

    #champs numerique

    Age:float=Field(ge=14,le=100)
    FCVC:float=Field(ge=1,le=3)
    NCP:float=Field(ge=1,le=4)
    CH2O:float=Field(ge=1,le=3)
    FAF:float=Field(ge=0,le=3)
    TUE:float=Field(ge=0,le=2)

#application fastapi
app=FastAPI(title="Obesity Prediction API",version="1.0.0")

#route de santé
@app.get("/health")
def health():
    #permet de verifier que le serveur est lancé et que le modele est bien été chargé
    return {"status":"ok","modelLoaded":True}
    
#route de prédiction
@app.post("/predict")
def predict(data:ObesityInput):
    #recoit les 14 features et applique le préprocesseur , lancer le modèle et renvoie la classe prédicte
    
    row=pd.DataFrame([data.model_dump()],columns=FEATURES)
    # model_dump() : transorme l'objet pydantic en dictionaire python
    #on impose FEATURES comme ordre des colonnes

    transformed=preprocessor.transform(row)
    #le preprocesseur applique : StandardScaler su age ,FCVC ,NCP ,CH2O ,FAF ,TUE
    #                            OneHotEncoder sur les variables cathegorielles

    probabilities=model.predict_proba(transformed)[0]
    # predict_proba() renvoie un tableau de forme (1,7)
    # [0] récupere la ligne correspondant a notre unique patient

    best_index=int(probabilities.argmax())
    #argmax renvoie l'indice de la probabilité la plus elevée

    prediction=CLASSES[best_index]
    #CLASSES transforme cet indice en nom de classe

    ordered=sorted(enumerate(probabilities),key=lambda item:item[1],reverse=True)
    #trie les 7 classes de la probabilité la plus forte a la plus faible


    #repose JSON renvoyer au frontend
    return{
        "prediction":prediction,
        "predictionFr":LABEL_FR.get(prediction,prediction),
        #confiance=probabilité de la classe choisie
        "confidence":round(float(probabilities[best_index]),4),
        #detail des 7 probabilités , deja triées
        "probabilities":[
            {
                "label":CLASSES[index],
                "labelFr":LABEL_FR.get(CLASSES[index],CLASSES[index]),
                "probability":round(float(probability),4),
            } for index,probability in ordered
        ],
        #info utiles pour savoir quel modèle a répondu
        "model":{
            "name":type(model).__name__,
            "version":"best_model_41.pkl",
        },
    }



