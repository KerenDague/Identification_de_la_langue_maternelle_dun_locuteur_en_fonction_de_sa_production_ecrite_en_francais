"""
Classification de texte avec Word2Vec et Random Forest.

Ce script permet de prédire automatiquement la langue d’un texte à partir d’un fichier CSV en utilisant des embeddings Word2Vec moyens des mots (moyenne des vecteurs de mots d’un texte) combinés à un classifieur Random Forest
Il inclut les étapes suivantes :
1. Chargement et préparation des données textuelles depuis un fichier CSV
2. Séparation en ensembles d’entraînement et de test (train/test split)
3. Construction d’un pipeline : W2VVectorizer : convertit les textes en vecteurs Word2Vec moyens; RandomForestClassifier : classifieur prenant en compte les déséquilibres de classes.
4. Entraînement du modèle et évaluation de ses performances
5. Génération et sauvegarde d’une matrice de confusion pour visualiser les prédictions

"""
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import gensim
from gensim.models import Word2Vec
import numpy as np
import time
import re

# Configuration
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_IMAGE_NAME = 'matrice_confusion_rf_w2v.png'

# Nettoyage et tokenisation simple
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', text)
    tokens = text.split()
    return tokens

# Transformer Word2Vec compatible avec sklearn
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, min_count=1, window=5):
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.model = None

    def fit(self, X, y=None):
        tokenized = [tokenize(doc) for doc in X]
        self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size, 
                              window=self.window, min_count=self.min_count, workers=4)
        return self

    def transform(self, X):
        tokenized = [tokenize(doc) for doc in X]
        vectors = []
        for tokens in tokenized:
            vecs = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
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

def build_rf_pipeline():
    vectorizer = Word2VecVectorizer(
        vector_size=100,
        min_count=1,
        window=5
        )
    classifier = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
        )
    pipeline = Pipeline([
        ('w2v', vectorizer),
        ('rf', classifier)
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
        cmap='Reds',
        linewidths=.5
    )


    plt.title('Matrice de Confusion - Random Forest Word2Vec', fontsize=16)
    plt.ylabel('Vraie Langue (True Label)', fontsize=12)
    plt.xlabel('Langue Prédite (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Matrice de confusion enregistrée sous : '{filename}'")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de l'image : {e}")

def main():
    parser = argparse.ArgumentParser(description="Choix d'une table CSV")
    parser.add_argument("-f", "--fichierCSV", help="Entrez le nom du fichier CSV")
    args = parser.parse_args()

    FILE_PATH = args.fichierCSV

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    # Séparer les données en train/test
    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15, # meilleurs résultats avec une taille 0.15
        random_state=42,
        stratify=y
    )
    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    rf_model = build_rf_pipeline()
    print("Début de l'entraînement...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    print("Évaluation du modèle...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion_matrix(y_test, y_pred, labels, OUTPUT_IMAGE_NAME)

if __name__ == "__main__":
    main()
