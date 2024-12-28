# Deep Learning Tumor Detection

A deep learning project for detecting brain tumors in medical images.

## Project Structure

```
DeepLearning-Tumor-Detection/
├── src/                    # Code source principal
│   ├── data/              # Gestion des données
│   │   ├── augmentation/  # Data augmentation
│   │   └── preprocessing/ # Prétraitement des données
│   ├── models/            # Définitions des modèles
│   └── utils/             # Utilitaires et fonctions communes
├── configs/               # Fichiers de configuration
├── tests/                 # Tests unitaires
├── notebooks/            # Jupyter notebooks pour l'exploration
├── docs/                 # Documentation
└── requirements.txt      # Dépendances du projet
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepLearning-Tumor-Detection.git
cd DeepLearning-Tumor-Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Augmentation

The project includes a robust image augmentation pipeline for enhancing the training dataset:

```python
from src.data.augmentation.image_augmentation import ImageAugmentor

augmentor = ImageAugmentor(
    image_size=256,
    augmentation_factor=2
)

augmentor.process_directory(
    input_dir="path/to/input/images",
    output_dir="path/to/output/images"
)
```

## Features

- Image data augmentation with configurable parameters
- Support for various image formats and color spaces
- Automated preprocessing pipeline
- (More features to be added)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Camille Maslin