# DSLR (Data Science x Logistic Regression)

Ce projet vise à recréer le Choixpeau magique de Poudlard en utilisant des techniques de Data Science et de Machine Learning.

## Installation

### Prérequis
- Python 3.x
- Terminal

### Configuration de l'environnement virtuel

1. **Création de l'environnement virtuel**
```bash
python3 -m venv .venv
```

2. **Activation de l'environnement virtuel**
```bash
source .venv/bin/activate
```

3. **Installation des dépendances**
```bash
pip3 install -r requirements.txt
```

4. **Désactivation de l'environnement virtuel**
```bash
deactivate
```

### Commandes utiles pour venv

- **Voir les dépendances**
```bash
pip3 list
```

- **Mettre à jour le fichier requirements.txt**
```bash
pip3 freeze > requirements.txt
```

### Nettoyage

- **Nettoyer les fichiers .pyc, __pycache__, .DS_Store et .venv**
```bash
find . -name "*.pyc" -delete
rm -rf __pycache__
rm -rf .venv
```

