"""
Classification de texte avec modèles BERT (FlauBERT, CamemBERT et RoBERTa).

Ce script permet de prédire automatiquement la langue d’un texte à partir d’un fichier CSV en utilisant un modèle Transformer Bert pour la classification de séquence textuelle.

Modèles disponibles : flaubert (Français), camembert (Français), roberta (Multilingue XLM-R)

Il inclut les étapes suivantes :
1. Chargement et préparation des données textuelles depuis un fichier CSV
2. Construction et préparation du modèle BERT choisi
3. Encodage des textes et des labels
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
import numpy as np
import time

# Configuration
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
MODELS = {
    "flaubert": "flaubert/flaubert_base_cased",
    "camembert": "camembert-base",
    "roberta": "xlm-roberta-base"  # Version multilingue de Roberta
}

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

# 2. Construction du modèle
def build_model(model_name, num_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    print(f"Modèle sélectionné : {model_name}")
    return tokenizer, model, optimizer, device

# 3. Encodage des textes
def encode_batch(tokenizer, texts, labels=None, max_length=128):
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    if labels is not None:
        enc['labels'] = torch.tensor(labels)
    return enc

# 4. Entraînement
def train_bert(tokenizer, model, optimizer, device, X_train, y_train, epochs=3, batch_size=8):
    model.train()
    n = len(X_train)

    for epoch in range(epochs):
        print(f"\n--- Époque {epoch+1}/{epochs} ---")
        indices = np.random.permutation(n)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]

            texts = X_train.iloc[batch_idx].tolist()
            labels = y_train.iloc[batch_idx].tolist()

            enc = encode_batch(tokenizer, texts, labels)
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Batch {i//batch_size} — Loss : {loss.item():.4f}")

    return model

# 5. Prédiction
def predict_bert(tokenizer, model, device, texts, batch_size=16):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = encode_batch(tokenizer, batch_texts)
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)

    return preds

# 6. Matrice de confusion et évaluation
def plot_confusion_matrix(y_true, y_pred, labels, filename, model_name):
    print("Génération de la matrice de confusion visuelle...")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Oranges", linewidths=.5)

    plt.title(f"Matrice de Confusion - BERT {model_name}", fontsize=16)
    plt.ylabel("Vraie Langue")
    plt.xlabel("Langue Prédite")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Matrice enregistrée sous : {filename}")

# Main
def main():
    parser = argparse.ArgumentParser(description="Classification BERT")
    parser.add_argument("-f", "--fichierCSV", required=True, help="Nom du fichier CSV")
    parser.add_argument("-m", "--modele", choices=["flaubert", "camembert", "roberta"], default="camembert",
                        help="Modèle à utiliser : flaubert, camembert, roberta")
    args = parser.parse_args()

    FILE_PATH = args.fichierCSV
    MODEL_CHOICE = args.modele

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    labels = sorted(y.unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    y_num = y.map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=0.15, random_state=42, stratify=y_num
    )

    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    tokenizer, model, optimizer, device = build_model(MODELS[MODEL_CHOICE], num_labels=len(labels))

    print("Début de l'entraînement...")
    start_time = time.time()
    model = train_bert(tokenizer, model, optimizer, device, X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    print("\nPrédiction...")
    y_pred_num = predict_bert(tokenizer, model, device, X_test.tolist())
    y_pred = [id2label[idx] for idx in y_pred_num]
    y_test_labels = [id2label[idx] for idx in y_test]

    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"\nAccuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test_labels, y_pred, digits=3))

    output_image_name = f"matrice_confusion_{MODEL_CHOICE}.png"

    plot_confusion_matrix(
        y_test_labels, y_pred, labels,
        output_image_name,
        MODEL_CHOICE
    )


if __name__ == "__main__":
    main()
