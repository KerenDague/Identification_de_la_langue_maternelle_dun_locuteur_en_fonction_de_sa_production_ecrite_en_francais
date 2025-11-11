"""
Classification de texte avec BERT et SVM.

Ce script permet de prédire automatiquement la langue d’un texte à partir d’un fichier CSV,en utilisant des embeddings issus d’un modèle Transformer (BERT) combiné à un classifieur SVM linéaire.
Il inclut les étapes suivantes :
1. Chargement et préparation des données textuelles depuis un fichier CSV
2. Séparation en ensembles d’entrainement et de test (train/test split)
3. Construction d’un pipeline : TransformerVectorizer : convertit les textes en vecteurs BERT ; SVM linéaire : classifieur prenant en compte les déséquilibres de classes
4. Entraînement du pipeline sur l'ensemble d'entraînement
5. Évaluation des performances sur l'ensemble de test 
6. Génération et sauvegarde d’une matrice de confusion sous forme d’image

"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

# Configuration
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_IMAGE_NAME = 'matrice_confusion_bert_svm.png'
TRANSFORMER_MODEL = 'distilbert-base-multilingual-cased'

# Transformer vectorizer pour sklearn
class TransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name=TRANSFORMER_MODEL, device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None

    def fit(self, X, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

    def transform(self, X):
        vectors = []
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                outputs = self.model(**inputs)
                cls_vec = outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy()
                vectors.append(cls_vec)
        return np.array(vectors)

def load_data(file_path, text_col, label_col):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' n'a pas été trouvé.")
        return None, None
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col]
    y = df[label_col]
    print(f"Données chargées : {len(df)} échantillons.")
    print(f"Nombre de classes : {y.nunique()}")
    return X, y

def build_pipeline():
    vectorizer = TransformerVectorizer()
    classifier = SVC(
        kernel='linear',
        class_weight='balanced',#MODIFICATON : SVM prend en compte que les classes ne sont pas équilibrées
        random_state=42
        )
    pipeline = Pipeline([
        ('transformer', vectorizer),
        ('svm', classifier)
    ])
    return pipeline

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    print(f"Génération de la matrice de confusion visuelle...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=.5
        )

    plt.title('Matrice de Confusion - BERT + SVM', fontsize=16)
    plt.ylabel('Vraie Langue', fontsize=12)
    plt.xlabel('Langue Prédite', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Matrice de confusion enregistrée sous : '{filename}'")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de l'image : {e}")

def main():
    parser = argparse.ArgumentParser(description="Classification BERT + SVM")
    parser.add_argument("-f", "--fichierCSV", help="Nom du fichier CSV")
    args = parser.parse_args()
    FILE_PATH = args.fichierCSV

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,# meilleurs résultats avec 0.20
        random_state=42,
        stratify=y
    )
    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)
    
    pipeline = build_pipeline()
    print("Début de l'entraînement...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion_matrix(y_test, y_pred, labels, OUTPUT_IMAGE_NAME)

if __name__ == "__main__":
    main()
