"""
Script de classification de texte avec un modèle Naive Bayes
Ce script permet de prédire automatiquement la langue d’un texte à partir d’un fichier CSV.
1. Chargement et préparation des données textuelles
2. Division en ensembles d’entraînement et de test
3. Construction d’un pipeline combinant un vectoriseur TF-IDF (n-grammes de caractères)et un classifieur Naive Bayes multinomial
4. Entraînement du modèle et évaluation de ses performances
5. Génération et sauvegarde d’une matrice de confusion pour visualiser les prédictions

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# Configuration
FILE_PATH = 'cleanB1.csv'
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_IMAGE_NAME = 'matrice_confusion_nb.png'


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


def build_nb_pipeline():
    # construire le pipeline TF-IDF + Naive Bayes
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),   # N-grammes de 3 à 5 caracteres
        max_features=None,
        sublinear_tf=True
    )
    
    classifier = MultinomialNB(alpha=0.1) 
    
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('nb', classifier)
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
        cmap='Greens',
        linewidths=.5
    )
    
    plt.title('Matrice de Confusion - Naive Bayes', fontsize=16)
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
    # 1. Charger les données
    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    # 2. Separer les données
    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    # 3. Construire et entraîner le pipeline
    nb_model = build_nb_pipeline()
    print("Début de l'entraînement...")
    start_time = time.time()
    nb_model.fit(X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    # 4. Évaluer le modèle
    print("Évaluation du modèle...")
    y_pred = nb_model.predict(X_test)

    # 5. Résultats
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion_matrix(y_test, y_pred, labels, OUTPUT_IMAGE_NAME)


if __name__ == "__main__":
    main()
