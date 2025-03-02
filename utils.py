import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from prepdata import data_recovery  # Provided by the project resources

# Threshold to determine if a dataset is balanced
THRESHOLD = 0.3


def load_and_analyze_dataset(dataset_name):
    """
    Loads a dataset using the provided data_recovery function and computes
    its main characteristics: number of samples, features, classes, class balance,
    and the number of missing values.
    """
    X, y = data_recovery(dataset_name)
    class_counts = Counter(y)
    total_samples = sum(class_counts.values())
    minority_ratio = min(class_counts.values()) / total_samples

    # Comptage des valeurs manquantes dans X
    # pd.isnull fonctionne aussi bien pour un DataFrame que pour un array numpy.
    missing_count = pd.isnull(X).sum() if hasattr(X, 'shape') else 0
    # Si X est un tableau 2D, la somme donne le total sur tous les éléments
    if isinstance(missing_count, np.ndarray):
        missing_count = missing_count.sum()

    return {
        'name': dataset_name,
        'X': X,
        'y': y,
        'n_samples': total_samples,
        'n_features': X.shape[1],
        'n_classes': len(class_counts),
        'minority_ratio': minority_ratio,
        'is_balanced': minority_ratio >= THRESHOLD,
        'class_distribution': dict(class_counts),
        'missing_values': missing_count
    }


def prepare_datasets(dataset_names):
    """
    Iterates over a list of dataset names, loading and analyzing each one.
    Returns a dictionary of dataset information.
    """
    datasets_info = {}
    for name in dataset_names:
        try:
            info = load_and_analyze_dataset(name)
            datasets_info[name] = info
        except Exception as e:
            print(f"Error processing {name}: {e}")
    return datasets_info


def display_dataset_summary(datasets_info):
    """
    Creates and prints a summary table of all datasets with key characteristics.
    """
    summary = []
    for name, info in datasets_info.items():
        summary.append({
            'Dataset': name,
            'Samples': info['n_samples'],
            'Features': info['n_features'],
            'Classes': info['n_classes'],
            'Balanced': info['is_balanced'],
            'Missing Values': info['missing_values']
        })
    df_summary = pd.DataFrame(summary)
    print(df_summary)
    return df_summary


if __name__ == '__main__':
    # List of dataset names provided in the project
    dataset_names = [
        'abalone8', 'abalone17', 'autompg', 'australian', 'balance',
        'bupa', 'german', 'glass', 'hayes', 'heart', 'iono',
        'libras', 'newthyroid', 'pageblocks', 'pima', 'segmentation',
        'sonar', 'spambase', 'splice', 'vehicle', 'wdbc', 'wine', 'wine4',
        'yeast3', 'yeast6'
    ]

    # Prepare and analyze the datasets
    datasets_info = prepare_datasets(dataset_names)

    # Display the summary of dataset characteristics
    df_summary = display_dataset_summary(datasets_info)

    # Optional: Plot a bar chart for the number of samples per dataset
    df_summary.set_index('Dataset')['Samples'].plot(kind='bar', figsize=(10, 5),
                                                    title="Number of Samples per Dataset")
    plt.ylabel("Number of Samples")
    plt.show()
