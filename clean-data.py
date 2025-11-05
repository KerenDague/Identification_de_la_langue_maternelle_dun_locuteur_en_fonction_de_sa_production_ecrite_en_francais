"""
Script de nettoyage et de préparation du corpus pour le classifieur.

- Supprime toutes les balises HTML présentes dans un fichier CSV
- Supprime les bruits dans les textes types 'Chine', 'Japonais' qui pourraient influencer nos classifieurs
- Supprime les langues que l'on ne va pas analyser'
- Produit un fichier CSV nettoyé
"""

import pandas as pd
import re
import argparse

# Suppresion des balises
def nettoyer_html(texte):

    texte = re.sub(r"<[^>]*>", "", str(texte))
    texte = re.sub(r"\b(C|c)hin(e|ois)\b", "", str(texte))
    texte = re.sub(r"\b(A|a)ngl(eterre|ais)\b", "", str(texte))
    texte = re.sub(r"\b(E|e)spagn(e|ol)\b", "", str(texte))
    texte = re.sub(r"\b(P|p)ortuga(l|is)\b", "", str(texte))
    texte = re.sub(r"\b(R|r)uss(ie|e)\b", "", str(texte))
    texte = re.sub(r"\b(J|j)apon(ais)?\b", "", str(texte))
    texte = re.sub(r"\b(K|k)abyle\b", "", str(texte))
    texte = re.sub(r"\b(A|a)rabe\b", "", str(texte))

    return texte


def main():

    # 1. Créer un parser pour entrer le nom du fichier à traiter
    parser = argparse.ArgumentParser(description="Choix d'une table CSV")
    parser.add_argument("-i", "--fichierEntree", help="Entrez le nom du fichier csv pour l'input", required=True)
    parser.add_argument("-o", "--fichierSortie", help="Entrez le nom du fichier csv pour l'output", required=True)
    args = parser.parse_args()

    # 2.Paramètres
    colonne_texte = "Texte"
    colonne_langue = "Langue"
    fichier_entree = args.fichierEntree
    fichier_sortie = args.fichierSortie
    langues_a_supprimer = ["KABYLE", "ESPAGNOL", "JAPONAIS", "ANGLAIS", "RUSSE"]

    # 3.Lecture et nettoyage
    df = pd.read_csv(fichier_entree, encoding="utf-8")
    df[colonne_texte] = df[colonne_texte].apply(nettoyer_html)
    df = df[~df[colonne_langue].isin(langues_a_supprimer)]

    # 4.Sauvegarde
    df.to_csv(fichier_sortie, index=False, encoding="utf-8")



if __name__ == "__main__":
    main()
