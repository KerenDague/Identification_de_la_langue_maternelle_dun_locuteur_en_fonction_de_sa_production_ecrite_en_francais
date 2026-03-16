import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import time


nlp = spacy.load("fr_core_news_md")

TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_IMAGE_NAME = 'matrice_confusion_svm.png'

#  Extracteurs
class POSExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X):
        pos_strings = []
        for doc in nlp.pipe(X, disable=["ner"]):
            seq = " ".join([token.pos_ for token in doc])
            pos_strings.append(seq)
        return np.array(pos_strings)

    def fit(self, X, y=None):
        return self

class StylometricExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X):
        features = []
        for text in X:
            length = len(text)
            punct_ratio = sum(1 for c in text if c in ".,;:!?") / max(1, length)
            tokens = text.split()
            lexdiv = len(set(tokens)) / max(1, len(tokens))
            features.append([length, punct_ratio, lexdiv])
        return np.array(features)

    def fit(self, X, y=None):
        return self

class L1TransferFeatureExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X):
        feats = []
        for doc in nlp.pipe(X, disable=["ner"]):
            det_count = 0
            noun_count = 0
            verb_positions = []
            subj_positions = []
            obj_positions = []
            dep_seq = []
            plural_count = 0

            for i, token in enumerate(doc):
                dep_seq.append(token.dep_)
                if token.pos_ == "DET": det_count += 1
                if token.pos_ == "NOUN": noun_count += 1
                if token.pos_ == "VERB": verb_positions.append(i)
                if token.dep_ == "nsubj": subj_positions.append(i)
                if token.dep_ == "obj": obj_positions.append(i)
                if token.tag_ == "NOUN__Number=Plur": plural_count += 1

            det_ratio = det_count / max(1, noun_count)
            subj_mean = np.mean(subj_positions) if subj_positions else 0
            verb_mean = np.mean(verb_positions) if verb_positions else 0
            obj_mean = np.mean(obj_positions) if obj_positions else 0
            dep_string = " ".join(dep_seq)
            feats.append([det_ratio, subj_mean, verb_mean, obj_mean, plural_count, dep_string])

        numeric = np.array([[f[0], f[1], f[2], f[3], f[4]] for f in feats])
        deps = [f[5] for f in feats]
        return numeric, deps

    def fit(self, X, y=None):
        return self

class DependencySplitter(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return X[1]

    def fit(self, X, y=None):
        return self

class NumericSplitter(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return X[0]

    def fit(self, X, y=None):
        return self

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

def build_svm_pipeline():
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), max_features=50000, sublinear_tf=True)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=30000, sublinear_tf=True)

    pos_pipeline = Pipeline([
        ("pos_extract", POSExtractor()),
        ("pos_tfidf", TfidfVectorizer())
    ])

    l1_pipeline_numeric = Pipeline([
        ("l1_extract", L1TransferFeatureExtractor()),
        ("num", NumericSplitter())
    ])

    l1_pipeline_dep = Pipeline([
        ("l1_extract", L1TransferFeatureExtractor()),
        ("dep", DependencySplitter()),
        ("dep_tfidf", TfidfVectorizer())
    ])

    vectorizer = ColumnTransformer([
        ('char_tfidf', char_vectorizer, TEXT_COLUMN),
        ('word_tfidf', word_vectorizer, TEXT_COLUMN),
        ('pos', pos_pipeline, TEXT_COLUMN),
        ('stylometric', StylometricExtractor(), TEXT_COLUMN),
        ('l1_numeric', l1_pipeline_numeric, TEXT_COLUMN),
        ('l1_dep', l1_pipeline_dep, TEXT_COLUMN),
    ], transformer_weights={
        'char_tfidf': 1.0,
        'word_tfidf': 1.0,
        'pos': 0.6,
        'stylometric': 0.3,
        'l1_numeric': 0.8,
        'l1_dep': 1.0,
    })

    classifier = SVC(kernel='linear', C=1.0, class_weight='balanced')

    pipeline = Pipeline([
        ('features', vectorizer),
        ('svm', classifier)
    ])

    return pipeline

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('Matrice de Confusion SVM')
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

    df_train = pd.DataFrame({TEXT_COLUMN: X, LABEL_COLUMN: y})
    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(
        df_train[[TEXT_COLUMN]], df_train[LABEL_COLUMN],
        test_size=0.2, random_state=42, stratify=y
    )

    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")

    svm_model = build_svm_pipeline()
    print("Début de l'entraînement...")
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    print("Évaluation du modèle...")
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion_matrix(y_test, y_pred, labels, OUTPUT_IMAGE_NAME)

if __name__ == "__main__":
    main()
