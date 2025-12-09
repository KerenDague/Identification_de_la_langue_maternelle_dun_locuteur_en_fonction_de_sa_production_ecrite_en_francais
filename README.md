# Identification de la Langue maternelle d'un locuteur en fonction de sa production écrite en français

## 🎯 Objectif

Ce projet vise à prédire la langue maternelle (L1) d'un locuteur à partir d'un essai qu'il a rédigé en français (L2).
Il s'agit d'une tâche de classification de texte (NLI - Native Language Identification).
L'objectif est d'explorer et de comparer les performances de différents modèles d'apprentissage automatique classiques (SVM, Naive Bayes, Random Forest) et profond (Transformers,BERT, RNN).

L’étude repose sur le corpus **TCFLE-8** (Test de Connaissance du Français Langue Étrangère – 2023).

## 🗃️ Données

**⚠️ Les données ne sont pas publiques.**

Le corpus contient :
- Des textes d'apprenants de différentes langues (japonais, chinois, arabe, portugais, kabyle, russe, espgnol, anglais)
- Des niveaux de langue CECRL (A1 à C2)
- Des métadonnées sur chaque participant (langue maternelle, sexe...)

## 🧠 Classifieur Automatique

1. **Nettoyage du corpus** : Supprime les bruits dans les textes types 'Chine',  'anglais', supprime les langues que l'on ne va pas analyser ( japonais, espagnol, kabyle) et supprime des balises inutiles.
2. **Modèle d'apprentissage de surface** :
   - Naives Bayes (avec comme vectorisation : TF-IDF )
   - Random Forest(avec comme vectorisation : TF-IDF ou Word2Vec ou BERT)
   - SVM (avec comme vectorisation : TF-IDF ou Word2Vec ou BERT) (+ ajout de features pour TF-IDF)
3. **Modèle d'apprentissage profond**:
   - Transformer T5
   - Trois types de RNN : SimpleRNN, LSTM & GRU
   - Trois types de BERT : CamemBERT, FlauBERT & XLM-RoBERTA

### 📊 Nos meilleurs résultats pour chaque modèle :
1. - Naives Bayes & TF-IDF = 56,94% d’ accuracy
   - Random Forest & TF-IDF = 48,23 % d’ accuracy
   - SVM & TF-IDF + features = 65,99% d’ accuracy
2. - Transformer T5 = 44,07 % d'accuracy
   - SimpleRNN = 55,51 % d'accuracy
   - CamemBERT = 64,03% d'accuracy

## 👥 Contributeurs

- [Keren DAGUE](https://github.com/KerenDague)
- [Juliette HENRY](https://github.com/juliettehnr)
- [Inès MARTINS](https://github.com/Inesmartins1912)
