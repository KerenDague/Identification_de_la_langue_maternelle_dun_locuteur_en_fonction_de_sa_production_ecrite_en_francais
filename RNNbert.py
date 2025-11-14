"""
Classification de texte avec BERT et RNN (RNN/GRU/LSTM).

Ce script permet de prédire automatiquement la langue d’un texte à partir d’un fichier CSV en utilisant des embeddings BERT comme entrée d’un réseau récurrent (RNN, GRU ou LSTM au choix en ligne de commande).

Il inclut les étapes suivantes :
1. Chargement et préparation des données textuelles depuis un fichier CSV
2. Séparation en ensembles d’entraînement et de test (train/test split)
3. Construction d'un DataLoader PyTorch pour le réseau RNN
4. Entraînement du réseau RNN sur l'ensemble d'entraînement 
5. Évaluation des performances sur l'ensemble de test 
6. Génération et sauvegarde d’une matrice de confusion sous forme d’image
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

# Configuration
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 128
MAX_SEQ_LEN = 128
TRANSFORMER_MODEL = 'distilbert-base-multilingual-cased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset PyTorch
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# RNN unifié
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, rnn_type='rnn'):
        super(RNNClassifier, self).__init__()
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type doit être 'rnn', 'gru' ou 'lstm'")
        self.fc = nn.Linear(hidden_size, num_classes)
        self.rnn_type = rnn_type

    def forward(self, x):
        x = x.unsqueeze(1)  # dimension temporelle = 1
        if self.rnn_type == 'lstm':
            out, (h_n, _) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Encodage BERT
class TransformerVectorizer:
    def __init__(self, model_name=TRANSFORMER_MODEL, device=device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(self, texts):
        vectors = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=MAX_SEQ_LEN
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
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

def plot_confusion_matrix(y_true, y_pred, labels, filename, rnn_type):
    print(f"Génération de la matrice de confusion visuelle...")
    labels_int = np.arange(len(labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels_int)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Purples', linewidths=.5)
    plt.title(f'Matrice de Confusion - TF-IDF + {rnn_type.upper()}', fontsize=16)
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
    parser = argparse.ArgumentParser(description="Classification TF-IDF + RNN")
    parser.add_argument("-f", "--fichierCSV", help="Nom du fichier CSV")
    parser.add_argument("-r", "--rnn_type", choices=['simple','gru','lstm'], default='gru',
                        help="Type de RNN à utiliser : 'rnn', 'gru' ou 'lstm'")
    args = parser.parse_args()

    FILE_PATH = args.fichierCSV
    RNN_TYPE = args.rnn_type
    OUTPUT_IMAGE_NAME = f"matrice_confusion_bert_{RNN_TYPE}.png"  # nom du fichier dynamique                                                   

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    print("Encodage des textes avec BERT...")
    start_time = time.time()
    vectorizer = TransformerVectorizer()
    X_train_vec = vectorizer.encode(X_train)
    X_test_vec = vectorizer.encode(X_test)
    print(f"Encodage terminé en {time.time() - start_time:.2f} secondes.")

    train_dataset = TextDataset(X_train_vec, y_train)
    test_dataset = TextDataset(X_test_vec, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = RNNClassifier(input_size=X_train_vec.shape[1], hidden_size=HIDDEN_SIZE,
                          num_classes=len(labels), rnn_type=RNN_TYPE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Début de l'entraînement...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
    print("Entraînement terminé.")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    accuracy = accuracy_score(y_test, all_preds)
    print(f"Accuracy : {accuracy*100:.2f}%\n")
    print(classification_report(y_test, all_preds, target_names=labels, digits=3))
    plot_confusion_matrix(y_test, all_preds, labels, OUTPUT_IMAGE_NAME, RNN_TYPE)

if __name__ == "__main__":
    main()

