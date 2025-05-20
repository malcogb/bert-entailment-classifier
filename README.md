# BERT Entailment Classifier

Ce projet implémente un modèle de classification d'entailment basé sur BERT.

## Structure du projet

```
.
├── train.py              # Script d'entraînement
├── model.py              # Définition du modèle BERTClassifier
├── demo.py               # Interface Gradio pour tester le modèle
├── requirements.txt      # Dépendances Python
├── saved_model.pt        # Fichier pour sauvegarder les checkpoints
└── README.md             # Ce fichier explique ce projet
```

## Lancer l'entraînement

Après avoir installé les dépendances :

```bash
pip install -r requirements.txt
```

Puis exécutez :

```bash
python train.py
```

## Test via Gradio

Lancez le fichier `demo.py` pour ouvrir l’interface utilisateur :

```bash
python demo.py
```

## Détails du modèle

- Modèle utilisé : `bert-base-multilingual-cased`
- Tokenizer : `AutoTokenizer`
- max_length : `259` cette valeur constitue le max_length de notre dataset
- batch_size : `16`
- Optimiseur : `Adam`
- Loss : `CrossEntropyLoss`

## Suivi des performances

Les métriques (accuracy, loss) sont suivies via [Weights & Biases](https://wandb.ai/).

## Données

Le fichier `train_6.csv` est utilisé comme dataset principal avec trois labels :
- 0 : entailment
- 1 : contradiction
- 2 : neutral

## Auteur

Projet développé avec 💻 et ☕ par Malco GBAKPA.
