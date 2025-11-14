"""
Classification de texte avec modèle Transformer T5.

Ce script permet de prédire automatiquement la langue d’un texte à partir d’un fichier CSV en utilisant un modèle Transformer T5.

Il inclut les étapes suivantes :
1. Chargement et préparation des données textuelles depuis un fichier CSV
2. Construction et préparation du modèle T5
3. Encodage des textes
4. Entraînement du modèle sur l’ensemble d’entraînement
5. Prédiction sur l’ensemble de test
6. Préparation de la matrice de confusion et évaluation des performances (accuracy + classification report)
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import torch
import numpy as np
import time

# Configuration
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_IMAGE_NAME = 'matrice_confusion_t5.png'
MODEL_NAME = "t5-small"

# 1. Chargement des données
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

# 2. Construction du modèle T5
def build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4)

    return tokenizer, model, optimizer, device

# 3. Encodage des textes
def encode_batch(tokenizer, texts, labels=None):
    prompts = ["classify: " + t for t in texts]

    enc_inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=128
    )

    if labels is not None:
        enc_labels = tokenizer(
            list(labels), return_tensors="pt",
            padding=True, truncation=True, max_length=5
        )
        enc_labels["input_ids"][enc_labels["input_ids"] == tokenizer.pad_token_id] = -100
        return enc_inputs, enc_labels

    return enc_inputs

# 4. Entraînement
def train_t5(tokenizer, model, optimizer, device, X_train, y_train, epochs=3, batch_size=8):
    model.train()
    n = len(X_train)

    for epoch in range(epochs):
        print(f"\n--- Époque {epoch+1}/{epochs} ---")
        indices = np.random.permutation(n)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]

            texts = X_train.iloc[batch_idx].tolist()
            labels = y_train.iloc[batch_idx].tolist()

            enc_in, enc_lbl = encode_batch(tokenizer, texts, labels)

            enc_in = {k: v.to(device) for k, v in enc_in.items()}
            enc_lbl = {k: v.to(device) for k, v in enc_lbl.items()}

            outputs = model(
                input_ids=enc_in["input_ids"],
                attention_mask=enc_in["attention_mask"],
                labels=enc_lbl["input_ids"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Batch {i//batch_size} — Loss : {loss.item():.4f}")

    return model

# 5. Prédiction
def predict_t5(tokenizer, model, device, texts):
    model.eval()
    preds = []

    with torch.no_grad():
        for text in texts:
            enc = encode_batch(tokenizer, [text])
            enc = {k: v.to(device) for k, v in enc.items()}

            out_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=5
            )

            pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            preds.append(pred)

    return preds

# 6. Matrice de confusion et évaluation
def plot_confusion_matrix(y_true, y_pred, labels, filename, model_name):
    print("Génération de la matrice de confusion visuelle...")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greys", linewidths=.5)

    plt.title(f"Matrice de Confusion - Transformer T5", fontsize=16)
    plt.ylabel("Vraie Langue")
    plt.xlabel("Langue Prédite")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Matrice enregistrée sous : {filename}")

# Main
def main():
    parser = argparse.ArgumentParser(description="Classification T5")
    parser.add_argument("-f", "--fichierCSV", help="Nom du fichier CSV")
    args = parser.parse_args()

    FILE_PATH = args.fichierCSV

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    tokenizer, model, optimizer, device = build_model()

    print("Début de l'entraînement...")
    start_time = time.time()
    model = train_t5(tokenizer, model, optimizer, device, X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    print("\nPrédiction...")
    y_pred = predict_t5(tokenizer, model, device, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=3))

    plot_confusion_matrix(
        y_test, y_pred, labels,
        OUTPUT_IMAGE_NAME,
        MODEL_NAME
    )


if __name__ == "__main__":
    main()
