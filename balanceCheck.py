# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from collections import Counter
from prepdata import data_recovery
import logging
from sklearn.model_selection import train_test_split, KFold
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(filename='dataset_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Liste des datasets
dataset_names = [
    'abalone8', 'abalone17', 'autompg', 'australian', 'balance',
    'bupa', 'german', 'glass', 'hayes', 'heart', 'iono',
    'libras', 'newthyroid', 'pageblocks', 'pima', 'segmentation',
    'sonar', 'spambase', 'splice', 'vehicle', 'wdbc', 'wine', 'wine4', 'yeast3', 'yeast6'
]

# Paramètres globaux
THRESHOLD = 0.3
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_SPLITS = 5
RANDOM_STATE = 42


def load_and_analyze_dataset(dataset_name: str) -> Dict:
    """
    Charge un dataset et calcule ses caractéristiques principales
    """
    X, y = data_recovery(dataset_name)
    class_counts = Counter(y)
    total_samples = sum(class_counts.values())
    minority_ratio = min(class_counts.values()) / total_samples

    return {
        'name': dataset_name,
        'X': X,
        'y': y,
        'n_samples': total_samples,
        'n_features': X.shape[1],
        'n_classes': len(class_counts),
        'minority_ratio': minority_ratio,
        'is_balanced': minority_ratio >= THRESHOLD,
        'class_distribution': dict(class_counts)
    }


def categorize_datasets(dataset_names: List[str]) -> Tuple[Dict[str, Dict], List[str], List[str]]:
    """
    Charge et catégorise tous les datasets
    """
    datasets_info = {}
    balanced_datasets = []
    imbalanced_datasets = []

    for dataset in dataset_names:
        try:
            info = load_and_analyze_dataset(dataset)
            datasets_info[dataset] = info

            if info['is_balanced']:
                balanced_datasets.append(dataset)
            else:
                imbalanced_datasets.append(dataset)

            logging.info(f"Successfully processed dataset {dataset}")

        except Exception as e:
            logging.error(f"Error processing dataset {dataset}: {e}")
            continue

    return datasets_info, balanced_datasets, imbalanced_datasets


def create_dataset_summary(datasets_info: Dict) -> pd.DataFrame:
    """
    Crée un DataFrame résumant les caractéristiques des datasets
    """
    summary_data = []
    for name, info in datasets_info.items():
        summary_data.append({
            'Dataset': name,
            'N_samples': info['n_samples'],
            'N_features': info['n_features'],
            'N_classes': info['n_classes'],
            'Minority_ratio': info['minority_ratio'],
            'Is_balanced': info['is_balanced']
        })

    return pd.DataFrame(summary_data)


def split_dataset(X: np.ndarray,
                  y: np.ndarray,
                  test_size: float = 0.2,
                  val_size: float = 0.25,
                  n_splits: int = 5,
                  random_state: int = None) -> Dict:
    """
    Divise un dataset en ensembles train, validation et test, puis crée des folds
    pour la validation croisée.
    """
    # 1) Split train+val et test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 2) Split train et validation
    #    Note : si test_size=0.2, alors la taille de X_temp est 0.8 * X.
    #    val_size/(1 - test_size) signifie que X_val sera 0.25 de X_temp,
    #    donc 0.25 * 0.8 = 0.20 de X (soit 20 %). Ainsi train=60% / val=20% / test=20%.
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_temp
    )

    # 3) Création des folds pour la validation croisée (5 par défaut)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(kf.split(X_train))

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val,   'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'cv_splits': cv_splits
    }


# Exécution principale
if __name__ == "__main__":
    # 1. Chargement et catégorisation des datasets
    print("Chargement et analyse des datasets...")
    datasets_info, balanced_datasets, imbalanced_datasets = categorize_datasets(dataset_names)

    print("\nDatasets équilibrés:", balanced_datasets)
    print("\nDatasets déséquilibrés:", imbalanced_datasets)

    # 2. Création et affichage du résumé
    summary_df = create_dataset_summary(datasets_info)
    print("\nRésumé des datasets:")
    print(summary_df)
