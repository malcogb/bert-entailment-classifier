# BERT Entailment Classifier

Ce projet implémente un modèle de classification d'entailment basé sur BERT.

## Structure du projet

```
.
├── train.py              # Script d'entraînement
├── model.py              # Définition du modèle BERTClassifier
├── demo.py               # Interface Gradio pour tester le modèle
├── requirements.txt      # Dépendances Python
├── train_6.csv           # Le dataset d'entraînement
├── saved_model.pt        # Fichier pour sauvegarder les checkpoints
└── README.md             # Ce fichier explique ce projet
```

## Installation

```bash
git clone https://github.com/malcogb/bert-entailment-classifier.git
cd bert-entailment-classifier

# Création d'un environnement virtuel
python -m venv wandb_env
wandb_env\Scripts\activate    # Sur Windows

# Installation des dépendances
transformers

torch

scikit-learn

pandas

wandb

gradio

Installe-les avec : pip install -r requirements.txt

## Lancer l'entraînement
Place le fichier train_6.csv dans le dossier racine.

Exécutez :

```bash
python train.py
```

## Test via Gradio : Démo interactive

Lancez le fichier `demo.py` pour ouvrir l’interface utilisateur afin de tester le modèle entraîné via une interface Web

```bash
python demo.py
```

Aperçu de l'interface : `bert-entailment-classifier\Images  captures d'écran\Interface gradio.png`

## Détails du modèle

- Modèle utilisé : `bert-base-multilingual-cased`
- Tokenizer : `AutoTokenizer`
- max_length : `259` cette valeur constitue le max_length de notre dataset
- batch_size : `16`
- Optimiseur : `Adam`
- Loss : `CrossEntropyLoss`

## Données : Correspondance des labels

Le fichier `train_6.csv` est utilisé comme dataset principal avec trois labels :
- 0 : entailment
- 1 : contradiction
- 2 : neutral


## Suivi des performances  et résultats

Les métriques (accuracy, loss) sont suivies via [Weights & Biases](https://wandb.ai/).
Courbes obtenues dans wandb :`bert-entailment-classifier\Images  captures d'écran\Interface gradio.png`

Métriques après 10 epochs : `bert-entailment-classifier\Images  captures d'écran\Métriques après entraînement.png`

## Ce qui est fait :
 Entraînement du modèle

 Interface Gradio

 ## Ce qui reste à faire :

 Optimisation hyperparamètres

 Ajout de tests unitaires


## Auteur

Projet développé avec 💻 et ☕ par Malco GBAKPA.
