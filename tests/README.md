# Tests Unitaires

Ce dossier contient les tests unitaires pour le projet de détection de tumeurs cérébrales.

## Structure

```
tests/
├── test_model.py          # Tests pour la classe BrainTumorCNN
├── test_visualization.py  # Tests pour les utilitaires de visualisation
└── README.md             # Ce fichier
```

## Exécution des Tests

### Exécuter tous les tests

```bash
python -m unittest discover tests
```

### Exécuter un fichier de test spécifique

```bash
python -m unittest tests/test_model.py
python -m unittest tests/test_visualization.py
```

### Exécuter un test spécifique

```bash
python -m unittest tests.test_model.TestBrainTumorCNN.test_model_initialization
```

## Couverture des Tests

Pour générer un rapport de couverture des tests :

```bash
coverage run -m unittest discover tests
coverage report
coverage html  # Génère un rapport HTML détaillé
```

## Conventions de Test

1. Chaque fichier de test doit commencer par "test_"
2. Chaque classe de test doit hériter de `unittest.TestCase`
3. Chaque méthode de test doit commencer par "test_"
4. Utilisez `setUp()` pour la configuration commune à plusieurs tests
5. Utilisez `tearDown()` pour le nettoyage après les tests
6. Incluez des docstrings descriptifs pour chaque test

## Bonnes Pratiques

1. Les tests doivent être indépendants les uns des autres
2. Évitez les dépendances externes dans les tests
3. Utilisez des données de test représentatives mais simples
4. Testez les cas limites et les cas d'erreur
5. Maintenez les tests à jour avec le code 