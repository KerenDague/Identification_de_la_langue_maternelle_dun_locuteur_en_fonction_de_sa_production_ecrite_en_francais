# Rapport de projet : Identification de la langue maternelle d'un locuteur en fonction de sa production écrite en français

## I. Présentation du Projet et du Corpus

### A. Projet et Problématique

Ce rapport fait la synthèse du travail réalisé sous la supervision de Mme Sülün
Aykurt-Buchwalter et Iris Eshkol-Taravella, toutes deux enseignantes-chercheuses à
l’université Paris-Nanterre et affiliées au laboratoire MoDyCo. Ce projet s’inscrit dans le
cadre du cours nommé “Apprentissage supervisé” dispensé par Mme Eshkol-Taravella à des
étudiants de M2 en Traitement Automatique des Langues. Il vise à familiariser les élèves à
travailler sur différents modèles d’apprentissage de surface et profond ce qui leur permet
d’appliquer concrètement leurs compétences en TAL.
Lors d’un précédent projet, nous avons travaillé sur l'entraînement d’un classifieur qui
catégorise les différents sens du pronom personnel indéfini de la 3e personne du singulier,
dans les récits d’apprenants du français. Nous avions obtenu des résultats significatifs aussi
bien lors de l’annotation manuelle que lors de la classification automatique de la machine.
Dans ce cadre, nous avions évalué les performances de deux modèles: la régression
logistique et CamemBERT.

Ces premiers résultats ayant été encourageants, nous avons souhaité étendre cette
démarche. Nous nous intéressons désormais à une problématique plus large : l’identification
de la langue maternelle d’un locuteur à partir de sa production écrite en français. Notre
hypothèse est qu’on ne fait pas les mêmes erreurs de français en fonction de notre langue
maternelle. Ainsi, on peut supposer que le locuteur du chinois mandarin (une langue
isolante) fera plus de fautes sur les accords en genre/nombre qu’un locuteur espagnol (une
langue flexionnelle comme le français). L’objectif est donc de déterminer si certaines
caractéristiques linguistiques propres aux différentes L1 peuvent être détectées automatiquement 
par le classifieur dans les textes d’apprenant.

### B. Le Corpus TCFLE-8

Le Test de Connaissance du Français Langue Etrangère 8 (3) est un corpus qui
regroupe plus de 6500 rédactions d’apprenants du français collectées par l’agence France
Éducation International (FEI) sous la tutelle du ministère de l’Éducation nationale. Huit
langues étaient représentées : le japonais, le chinois, le russe, l’anglais, l’espagnol, le
portugais, l’arabe et le kabyle. Chaque participant était soumis à trois tâches : une première
tâche consistait à écrire un court message à but narratif ou informatif, une seconde à écrire
un texte plus long avec un certain nombre de mots attendus et une dernière à rédiger un
texte argumentatif qui défendait un point de vue.

A l’issue de ce test, en fonction de leurs réponses, les participants se voyaient
attribuer un niveau d’acquisition du français allant de A1 à C2 en accord avec le Cadre
Européen Commun de Référence pour les Langues (CECRL).

### C. Prétraitement et Nettoyage du Corpus

Le corpus fourni en l’état par Mme Aykurt-Buchwalter n’était pas directement
exploitable. En effet, le contrat de mise à disposition ne nous permettait pas de le modifier
mais surtout, il est présenté en deux parties : un premier tableur csv comprenant les
réponses des participants au test accompagnées de leurs numéros d’identification et de
leurs niveau de français (A1, A2...), et un second tableur csv comprenant les profils des
candidats : id, langue maternelle, genre, niveau de français, etc.

Ainsi, nous avons commencé par relier ces deux tables avec une requête SQL (voir
ci-dessous) pour ne conserver que les données qui nous intéressaient, c’est-à-dire la langue
maternelle du participant, son niveau et ses réponses aux différentes questions.

Une fois la table des réponses des candidats et la table des métadonnées réunies,
nous avons exporter le tout vers un fichier excel pour visualiser les données. Nous pouvions
remarquer que les textes des apprenants niveau C étaient extrêmement bien rédigés. Or,
notre hypothèse se base sur les erreurs orthographiques/syntaxiques des apprenants. Nous
avons donc décidé d’écarter les réponses de niveau C de notre projet. À l’aide de l’outil de
filtrage d’Excel, nous avons sélectionné le niveau A, le niveau B et les niveaux A et B
réunies pour les exporter vers trois fichiers CSV différents. Ainsi, chacun de nos classifieur a
été entraîné sur ces trois tables : tableA.csv (plus de fautes mais moins de documents sur
lesquels s'entraîner), tableB.csv (plus de documents mais moins de fautes), tableAB.csv
(beaucoup plus de documents et un taux équilibré d’erreurs). Le but ici est de comparer les
évaluations des modèles pour voir si le niveau impacte les résultats.

Avant de fournir les fichiers CSV aux différents classifieurs, il était nécessaire de
nettoyer le corpus afin d’éviter toute perturbation pendant l’apprentissage. Pour cela, nous
avons développé un script nommé clean-data.py qui effectue plusieurs opérations de
prétraitement. D’abord, il supprime les balises HTML présentes dans les textes. Ensuite, il
élimine les termes pouvant révéler directement la langue maternelle des apprenants
(comme “Chine”, “japonais”, “anglais”, etc.) Ce nettoyage s’est fait avec l’utilisation de la
librairie python re.

Le script clean-data.py sert également à exclure du dataframe les langues qui ne
sont pas prises en compte dans notre analyse. En effet, nous avons considéré que les huit
langues du corpus TCFLE-8 rendraient plus difficile la tâche de classification. Par
conséquent, nous avons choisi de ne conserver que cinq langues : anglais (une langue
germanique), arabe (une langue sémitique), chinois (une langue sino-tibétaine), portugais
(une langue romane) et russe (une langue slave). Cet échantillon nous permet d’avoir une
bonne représentation des différentes typologies de langue ainsi que des classes
relativement équilibrées (niveaux A et B confondus).
Une fois ces traitements appliqués, le script génère un fichier CSV propre et prêt à
être utilisé par les modèles.

## IV. Analyse globale des résultats

Au terme de nos expérimentations sur le corpus TCFLE-8, une tendance claire se
dessine : la tâche d'Identification de la langue maternelle obéit à des logiques différentes de
la classification de texte thématique classique. L'analyse des performances de nos différents
modèles nous permet de tirer des conclusions linguistiques et techniques précises.

L'un des constats les plus frappants de cette étude est l'échec relatif des modèles
basés purement sur des embeddings sémantiques, comme Word2Vec ou l'extraction brute
de BERT, comparé aux approches de surface. En effet, les modèles sémantiques cherchent
à capturer le sens du texte ; or, un apprenant chinois et un apprenant espagnol répondant à
la même consigne racontent souvent la même histoire avec les mêmes mots-clés, ce qui
crée de la confusion pour le classifieur. À l'inverse, le TF-IDF appliqué aux n-grammes de
caractères permet un bond significatif de performance car il capture la "texture" de l'écriture.
Il repère les suffixes erronés, les préfixes calqués sur la langue maternelle et les fautes
d'orthographe récurrentes, prouvant ainsi que le modèle n'apprend pas ce que l'apprenant
raconte, mais bien la manière dont il l'écrit.

Nos deux meilleurs modèles, le SVM avec Features et CamemBERT fine-tuné,
obtiennent des scores très proches mais procèdent de manières radicalement opposées. Le
SVM s'appuie sur des règles linguistiques explicites fournies par nos extracteurs, comme le
ratio de déterminants ou la position du verbe, validant ainsi notre hypothèse de départ selon
laquelle les traces de la langue maternelle sont quantifiables syntaxiquement. De son côté,
CamemBERT parvient à un résultat quasi équivalent grâce au mécanisme d'attention, en
apprenant la probabilité d'enchaînement des mots et en repérant les tournures de phrases
"non-françaises".

Lorsque l’on observe les résultats pour chaque langue, un schéma se repère. On
constate que l'arabe et le chinois sont correctement détectés, tandis que l'anglais et le russe
génèrent plus de confusion. Cette différence s'explique par la proximité linguistique avec le
français : plus la langue maternelle est éloignée structurellement, plus les traces laissées
par l'apprenant sont atypiques et faciles à repérer pour le modèle. À l'inverse, une langue
proche comme l'anglais engendre des erreurs plus subtiles qui compliquent la distinction. 

## V. Conclusion
Pour conclure ce rapport, nous allons revenir brièvement sur chaque étape du projet
puis évoquer des pistes d’amélioration. Tout d’abord, ce projet d’apprentissage supervisé
était l’occasion pour nous d’explorer davantage le corpus TCFLE-8, qui regroupe des textes
écrits en français par des apprenants allophones. Notre hypothèse de départ est que les
spécificités linguistiques des langues maternelles des apprenants auraient une influence sur
leurs productions écrites en français. Plus précisément, nous supposions qu’il est possible
pour la machine de distinguer deux écrits d’allophones en se basant sur des indices
orthographiques et syntaxiques. Pour démontrer cette hypothèse, nous avons entraîné une
série de classifieurs dans le but de trouver le meilleur modèle de classification par rapport à
notre tâche.

Dans un premier temps, nous avons commencé par entraîner des modèles
d’apprentissage de surface. Nous avons choisi d’entraîner les modèles NB (modèle
probabiliste), Random Forest (modèle ensembliste) et SVM (modèle linéaire). Les modèles
ont tous été fine-tunés et combinés avec différentes représentations vectorielles (TF-IDF,
Word2Vec et BERT) dans le but de trouver les meilleurs résultats possibles.
Malheureusement, les résultats étaient peu concluants et les scores d’accuracy ne
permettaient pas de discriminer l’hypothèse du hasard. Le meilleur script était celui du
modèle SVM combiné à la vectorisation TF-IDF sur le corpus AB, avec une accuracy de
62,87 %. Nous avons par conséquent essayé de maximiser ce score en ajoutant des
features au script. L’ajout des features a permis d’augmenter le score à 65,99 %, ce qui
reste en dessous de nos attentes.

Par la suite, nous avons entraîné des modèles d’apprentissage profond en
commençant par différents modèles de RNN, puis des Transformers comme T5 et différents
modèles de BERT. Comme évoqué précédemment les résultats n’ont une fois encore pas
été spécialement probants avec des résultats très faibles pour certains modèles et meilleurs
pour d’autres. L’apprentissage profond offre des outils très puissants pour analyser et
classifier du texte, mais nous avons pu constater que leur efficacité dépend fortement de la
manière dont les données sont représentées et du type de modèle utilisé. Nous avons
observé que des approches simples comme TF-IDF combinées à un RNN basique peuvent
surpasser des architectures plus complexes lorsque la représentation du texte est déjà
stable et informative. À l’inverse, avec des embeddings contextuels comme ceux produits
par BERT, des modèles plus avancés tels que les LSTM tirent mieux parti des dépendances
internes et fournissent de meilleures performances. Nous avons également pu constater les
limites de certains modèles comme T5 qui s’est avéré peu adapté à la classification de
labels courts : la moindre variation dans la sortie générée étant comptée comme une erreur,
et les versions « small » du modèle nécessitent davantage de données pour apprendre
efficacement. Pour finir sur l’apprentissage profond, nos tests sur les modèles BERT
montrent que CamemBERT dépasse Flaubert et XLM-RoBERTa dans le cadre de notre
classification. Ce résultat s’expliquant par la qualité et l’étendue de son pré-entraînement,
mieux aligné avec les spécificités du français et de son architecture efficiente, ce qui lui
permet de produire des représentations plus robustes pour la classification.

Malheureusement, les résultats obtenus, même après le fine-tuning et l’ajout de
features, n’étaient pas à la hauteur de nos espérances. Ainsi, il est difficile d’accepter notre
hypothèse selon laquelle une machine serait capable de différencier des textes en français
rédigés par des allophones, en se basant sur des indices orthographiques et syntaxiques.
Dans une perspective d’amélioration de nos résultats, il serait idéal de faire appel à des
linguistes spécialistes des langues mentionnées. En effet, aucune de nous ne parle
couramment les cinq langues sélectionnées, ce qui nous freine dans l’ajout de features
pertinentes. 
