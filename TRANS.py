"""
Script de classification de texte avec Transformers
Prédit automatiquement la langue d’un texte à partir d’un fichier CSV
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import time
import os
import argparse

# CONFIGURATION
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
MODEL_NAME = 'distilbert-base-multilingual-cased'
OUTPUT_IMAGE_NAME = 'matrice_confusion_transformers.png'
EPOCHS = 2
BATCH_SIZE = 8
MAX_LENGTH = 128

def load_data(file_path, text_col, label_col):
    if not os.path.exists(file_path):
        print(f"ERREUR : Le fichier '{file_path}' n'a pas été trouvé.")
        return None, None, None

    df = pd.read_csv(file_path)
    df = df.dropna(subset=[text_col, label_col])

    labels = sorted(df[label_col].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df['label'] = df[label_col].map(label2id)

    print(f"Données chargées : {len(df)} échantillons.")
    print(f"Nombre de classes : {len(labels)}")
    return df, label2id, id2label

#Tokenization
def tokenize_data(df, tokenizer, text_col=TEXT_COLUMN):
    return tokenizer(
        df[text_col].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )


# Matrice de confusion
def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Purples', linewidths=.5)
    plt.title('Matrice de Confusion - Transformers', fontsize=16)
    plt.ylabel('Vraie Langue', fontsize=12)
    plt.xlabel('Langue Prédite', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Matrice de confusion enregistrée sous : '{filename}'")


# Fonction de métriques
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}


def main():

    # Parser pour choisir le fichier CSV
    parser = argparse.ArgumentParser(description="Choix d'une table CSV")
    parser.add_argument("-f", "--fichierCSV", help="Entrez le nom du fichier CSV")
    args = parser.parse_args()
    FILE_PATH = args.fichierCSV

    # 1. Chargement des données
    df, label2id, id2label = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if df is None:
        return

    # 2. Split train/test
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df['label']
    )
    print(f"Taille de l'entraînement : {len(train_df)}, Taille du test : {len(test_df)}")

    # 3. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenize_data(train_df, tokenizer)
    test_encodings = tokenize_data(test_df, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_df['label'].tolist()
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_df['label'].tolist()
    })

    # 4. Charger le modèle
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 5. Entraînement
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir='./logs',
        logging_steps=50,
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Début de l'entraînement...")
    start_time = time.time()
    trainer.train()
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    # 6. Évaluation
    print("Évaluation du modèle...")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = y_pred.argmax(axis=1)
    y_true = test_df['label'].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy : {accuracy*100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=list(label2id.keys()), digits=3))

    # 7. Matrice de confusion
    plot_confusion_matrix(y_true, y_pred, list(label2id.keys()), OUTPUT_IMAGE_NAME)

if __name__ == "__main__":
    main()
