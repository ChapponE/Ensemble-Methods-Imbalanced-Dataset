#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import time
import subprocess
import os
import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier, StackingClassifier,
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Importation des fonctions de récupération et d'analyse des datasets
from prepdata import data_recovery
from balanceCheck import categorize_datasets

# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The SAMME.R algorithm')

# Configuration du logging
logging.basicConfig(filename='all.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Configuration centrale des hyperparamètres, chargeurs de données, et modèles
# =============================================================================
HYPERPARAMS = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 3],
    "min_samples_split": [10, 30],
    "C": [0.01, 0.1, 1.0, 10.0],
    "C_stack": [0.01, 0.1, 1.]
}

dataset_list = [
    'abalone8', 'abalone17', 'autompg', 'australian', 'balance',
    'bupa', 'german', 'glass', 'hayes', 'heart', 'iono',
    'libras', 'newthyroid', 'pageblocks', 'pima', 'segmentation',
    'sonar', 'spambase', 'splice', 'vehicle', 'wdbc', 'wine', 'wine4',
    'yeast3', 'yeast6'
]
# dataset_list = [
#     'abalone8']

DATA_LOADERS = {
    "balanced": lambda ds: data_recovery(ds),
    "imbalanced": lambda ds: data_recovery(ds),
    "oversampling": lambda ds: SMOTE(random_state=42).fit_resample(*data_recovery(ds)),
    "undersampling": lambda ds: RandomUnderSampler(random_state=42).fit_resample(*data_recovery(ds)),
    "hybridsampling": lambda ds: SMOTEENN(random_state=42).fit_resample(*data_recovery(ds))
}

# -----------------------------------------------------------------------------
# Fonction pour sauvegarder un DataFrame en PDF via LaTeX
# -----------------------------------------------------------------------------
def save_table_to_pdf(table: pd.DataFrame, pdf_filename: str):
    latex_table = table.to_latex(index=False, escape=False)
    latex_document = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
\begin{flushleft}
\resizebox{\textwidth}{!}{%%
%s
}
\end{flushleft}
\end{document}
""" % latex_table

    tex_filename = pdf_filename.replace(".pdf", ".tex")
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(latex_document)
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename],
                       check=True, stdout=subprocess.DEVNULL)
        print(f"PDF '{pdf_filename}' généré avec succès.")
    except Exception as e:
        logging.error(f"Erreur pendant la compilation PDF de {tex_filename}: {e}")
        print(f"Erreur pendant la compilation PDF : {e}")
    for ext in [".aux", ".log", ".tex"]:
        aux_file = pdf_filename.replace(".pdf", ext)
        if os.path.exists(aux_file):
            os.remove(aux_file)

# -----------------------------------------------------------------------------
# Fonction d'évaluation pour tous les datasets (sans itérations multiples)
# -----------------------------------------------------------------------------
def evaluate_algorithm_all_datasets(dataset_names: List[str],
                                    data_loader,
                                    train_func) -> Dict:
    results = {}
    # On utilise un KFold fixe pour tous les datasets
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for dataset_name in dataset_names:
        print(f"\n[{train_func.__name__}] Traitement du dataset: {dataset_name}")
        try:
            X, y = data_loader(dataset_name)
            result = train_func(X, y, cv, is_balanced=True)
            results[dataset_name] = result
        except Exception as e:
            logging.error(f"Erreur sur {dataset_name} avec {train_func.__name__}: {e}")
    return results

# -----------------------------------------------------------------------------
# Création d'un tableau comparatif à partir des résultats (meilleur f1 score)
# -----------------------------------------------------------------------------
def create_comparison_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    
    # Vérifie si results_dict est vide
    if not results_dict:
        print("Aucun résultat disponible pour générer un tableau comparatif.")
        return pd.DataFrame(columns=["Dataset"] + [model["name"] for model in MODELS])
    
    # Identification des datasets communs
    datasets = set()
    for model_name, model_results in results_dict.items():
        datasets.update(model_results.keys())
        print('model_name', model_name)
        print('model_results.keys()', model_results.keys())
    common_datasets = sorted(datasets)
    
    # Si aucun dataset commun, retourne un DataFrame vide avec les colonnes appropriées
    if not common_datasets:
        print("Aucun dataset commun trouvé pour générer un tableau comparatif.")
        return pd.DataFrame(columns=["Dataset"] + [model["name"] for model in MODELS])
        
    for dataset in common_datasets:
        scores = {}
        for model in MODELS:
            mname = model["name"]
            if mname in results_dict and dataset in results_dict[mname]:
                res = results_dict[mname][dataset]
                # Utilisation des scores de test
                best_f1 = res["best_model"]["test"]["mean_f1"]
                best_std = res["best_model"]["test"]["std_f1"]
                scores[mname] = (best_f1, best_std)
        
        if scores:
            max_score = max(mean for mean, _ in scores.values())
            row = {"Dataset": dataset}
            for mname, (mean, std) in scores.items():
                formatted = f"{mean * 100:.1f} ± {std * 100:.1f}"
                if mean == max_score:
                    formatted = f"\\textbf{{{formatted}}}"
                row[mname] = formatted
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=["Dataset"] + [model["name"] for model in MODELS])
    
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Fonction pour collecter les métriques de performance et d'overfitting
# -----------------------------------------------------------------------------
def collect_fitting_metrics(model, X, y, cv, param_grid):
    """
    Collecte des métriques détaillées sur l'évolution du modèle et son comportement
    d'overfitting/underfitting selon son hyperparamètre principal.
    
    Returns:
        dict: Dictionnaire contenant les métriques d'évolution et d'overfitting
    """
    model_type = type(model).__name__
    metrics = {
        'hyperparameter': None,
        'values': [],
        'train_scores': [],
        'val_scores': [],
        'train_val_gap': [],
        'fitting_status': []
    }
    
    # Définir le paramètre principal à faire varier selon le type de modèle
    if model_type in ['RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'BaggingClassifier']:
        metrics['hyperparameter'] = 'n_estimators'
        param_values = sorted([p for k, p in param_grid.items() if 'n_estimators' in k][0]) if any('n_estimators' in k for k in param_grid) else [50, 100, 150, 200]
        
    elif model_type == 'LogisticRegression':
        metrics['hyperparameter'] = 'C'
        param_values = sorted([p for k, p in param_grid.items() if 'C' in k][0]) if any('C' in k for k in param_grid) else [0.01, 0.1, 1.0, 10.0]
        
    elif model_type == 'DecisionTreeClassifier':
        metrics['hyperparameter'] = 'max_depth'
        param_values = sorted([p for k, p in param_grid.items() if 'max_depth' in k][0]) if any('max_depth' in k for k in param_grid) else [2, 3, 5, 10]
        
    elif 'StackingClassifier' in model_type:
        metrics['hyperparameter'] = 'meta_C'
        param_values = sorted([p for k, p in param_grid.items() if 'final_estimator__C' in k][0]) if any('final_estimator__C' in k for k in param_grid) else [0.01, 0.1, 1.0, 10.0]
    
    else:
        # Modèle non reconnu, retourner des métriques vides
        return metrics
    
    # Initialiser le splitter CV
    cv_splitter = cv if hasattr(cv, 'split') else KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Pour chaque valeur du paramètre principal
    for param_value in param_values:
        # Configurer le modèle avec ce paramètre
        if metrics['hyperparameter'] == 'n_estimators':
            if model_type == 'RandomForestClassifier':
                current_model = RandomForestClassifier(n_estimators=param_value, random_state=42)
            elif model_type == 'AdaBoostClassifier':
                current_model = AdaBoostClassifier(n_estimators=param_value, random_state=42)
            elif model_type == 'GradientBoostingClassifier':
                current_model = GradientBoostingClassifier(n_estimators=param_value, random_state=42)
            elif model_type == 'BaggingClassifier':
                current_model = BaggingClassifier(n_estimators=param_value, random_state=42)
        
        elif metrics['hyperparameter'] == 'C':
            current_model = LogisticRegression(C=param_value, max_iter=1000, random_state=42)
        
        elif metrics['hyperparameter'] == 'max_depth':
            current_model = DecisionTreeClassifier(max_depth=param_value, random_state=42)
        
        elif metrics['hyperparameter'] == 'meta_C':
            # Copier le modèle de stacking mais mettre à jour le C du final_estimator
            final_estimator = LogisticRegression(C=param_value, max_iter=1000, random_state=42)
            estimators = [(name, est) for name, est in model.estimators]
            current_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
        
        # Évaluer le modèle sur chaque fold
        train_scores, val_scores = [], []
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Standardiser les données
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Entraîner le modèle
            current_model.fit(X_train_scaled, y_train)
            
            # Calculer les scores
            y_train_pred = current_model.predict(X_train_scaled)
            y_val_pred = current_model.predict(X_val_scaled)
            
            train_f1 = f1_score(y_train, y_train_pred, average='macro')
            val_f1 = f1_score(y_val, y_val_pred, average='macro')
            
            train_scores.append(train_f1)
            val_scores.append(val_f1)
        
        # Calculer les moyennes
        mean_train = np.mean(train_scores)
        mean_val = np.mean(val_scores)
        gap = mean_train - mean_val
        
        # Déterminer le statut d'overfitting/underfitting
        if mean_train < 0.6 and mean_val < 0.6:
            status = "underfitting"
        elif gap > 0.15:
            status = "overfitting"
        elif gap < 0.05 and mean_val > 0.7:
            status = "optimal"
        else:
            status = "balanced"
        
        # Stocker les résultats
        metrics['values'].append(param_value)
        metrics['train_scores'].append(float(mean_train))
        metrics['val_scores'].append(float(mean_val))
        metrics['train_val_gap'].append(float(gap))
        metrics['fitting_status'].append(status)
    
    return metrics

def train_model_with_grid_search(model, param_grid: Dict, X: np.ndarray, y: np.ndarray,
                               cv, is_balanced: bool) -> Dict:
    # Mesure du temps de début
    start_time = time.time()
    
    # Création d'un pipeline avec standardisation et le modèle
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    
    # Configuration de GridSearchCV avec return_train_score=True pour récupérer les scores d'entraînement
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=cv,
                               scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
                               refit='f1',
                               return_train_score=True,
                               n_jobs=-1)
    grid_search.fit(X, y)
    cv_results = grid_search.cv_results_
    
    # Fonction pour extraire le nom réel du paramètre, même imbriqué
    def get_param_name(param_key):
        parts = param_key.split('__')
        if len(parts) > 2:  # Paramètre imbriqué comme 'clf__estimator__max_depth'
            return parts[-1]  # Retourne 'max_depth'
        else:  # Paramètre simple comme 'clf__n_estimators'
            return parts[-1]  # Retourne 'n_estimators'
    
    # Extraction des résultats de la recherche par grille avec des clés plus lisibles
    hyper_results = {}
    for i, params in enumerate(cv_results['params']):
        # Construction de la clé à partir des hyperparamètres, avec des noms plus significatifs
        key = '_'.join([f"{get_param_name(k)}_{params[k]}" for k in sorted(params)])
        hyper_results[key] = {
            'train': {
                'mean_accuracy': cv_results['mean_train_accuracy'][i],
                'std_accuracy': cv_results['std_train_accuracy'][i],
                'mean_f1': cv_results['mean_train_f1'][i],
                'std_f1': cv_results['std_train_f1'][i]
            },
            'test': {
                'mean_accuracy': cv_results['mean_test_accuracy'][i],
                'std_accuracy': cv_results['std_test_accuracy'][i],
                'mean_f1': cv_results['mean_test_f1'][i],
                'std_f1': cv_results['std_test_f1'][i]
            }
        }
    
    best_index = grid_search.best_index_
    best_params = grid_search.best_params_
    best_model_scores = {
        'train': {
            'mean_accuracy': cv_results['mean_train_accuracy'][best_index],
            'std_accuracy': cv_results['std_train_accuracy'][best_index],
            'mean_f1': cv_results['mean_train_f1'][best_index],
            'std_f1': cv_results['std_train_f1'][best_index]
        },
        'test': {
            'mean_accuracy': cv_results['mean_test_accuracy'][best_index],
            'std_accuracy': cv_results['std_test_accuracy'][best_index],
            'mean_f1': cv_results['mean_test_f1'][best_index],
            'std_f1': cv_results['std_test_f1'][best_index]
        }
    }
    
    # Collecter les métriques d'overfitting
    fitting_metrics = collect_fitting_metrics(model, X, y, cv, param_grid)
    
    # Calculer des métriques supplémentaires utiles pour l'analyse
    additional_metrics = {
        'convergence_rate': best_model_scores['test']['mean_f1'] / best_model_scores['train']['mean_f1'],
        'train_test_ratio': best_model_scores['test']['mean_f1'] / best_model_scores['train']['mean_f1'],
        'train_test_gap': best_model_scores['train']['mean_f1'] - best_model_scores['test']['mean_f1']
    }
    
    # Ajouter des informations sur la complexité du modèle
    complexity_metrics = {}
    model_type = type(model).__name__
    if model_type == 'RandomForestClassifier':
        complexity_metrics = {
            'n_trees': best_params.get('clf__n_estimators', None),
            'max_depth': best_params.get('clf__max_depth', None)
        }
    elif model_type == 'DecisionTreeClassifier':
        complexity_metrics = {
            'max_depth': best_params.get('clf__max_depth', None)
        }
    elif model_type in ['AdaBoostClassifier', 'GradientBoostingClassifier', 'BaggingClassifier']:
        complexity_metrics = {
            'n_estimators': best_params.get('clf__n_estimators', None)
        }
    elif model_type == 'LogisticRegression':
        complexity_metrics = {
            'regularization': 1.0 / best_params.get('clf__C', 1.0)  # Plus C est petit, plus la régularisation est forte
        }
    
    # Calcul du temps d'exécution
    execution_time = time.time() - start_time
    
    return {
        'hyperparameter_results': hyper_results,
        'best_model': {
            'params': best_params,
            **best_model_scores
        },
        'fitting_metrics': fitting_metrics,
        'additional_metrics': additional_metrics,
        'complexity_metrics': complexity_metrics,
        'dataset_size': len(X),
        'n_features': X.shape[1],
        'class_distribution': {str(label): int(sum(y == label)) for label in np.unique(y)},
        'execution_time': execution_time
    }


# -----------------------------------------------------------------------------
# Fonctions d'entraînement pour chaque classifieur
# Chaque fonction reçoit l'ensemble du dataset X, y et le splitter cv
# -----------------------------------------------------------------------------
def train_logistic_regression(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    model = LogisticRegression(max_iter=1000, class_weight='balanced' if is_balanced else None)
    param_grid = {'clf__C': HYPERPARAMS["C"]}
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)

def train_random_forest(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    model = RandomForestClassifier(class_weight='balanced' if is_balanced else None, random_state=42)
    param_grid = {
        'clf__n_estimators': HYPERPARAMS["n_estimators"],
        'clf__max_depth': HYPERPARAMS["max_depth"],
        'clf__min_samples_split': HYPERPARAMS["min_samples_split"]
    }
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)

def train_bagging_LR(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    # Utiliser un estimateur minimal sans configuration avancée
    model = BaggingClassifier(
        estimator=LogisticRegression(),
        random_state=42
    )
    param_grid = {
        'clf__n_estimators': HYPERPARAMS["n_estimators"],
        'clf__estimator__C': HYPERPARAMS["C"],  # Changé de base_estimator à estimator
        'clf__estimator__max_iter': [1000],
        'clf__estimator__class_weight': ['balanced' if is_balanced else None],
        'clf__estimator__random_state': [42]
    }
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)

def train_adaboost_LR(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    # Utiliser un estimateur minimal sans configuration avancée
    model = AdaBoostClassifier(
        estimator=LogisticRegression(),
        algorithm='SAMME', 
        random_state=42
    )
    param_grid = {
        'clf__n_estimators': HYPERPARAMS["n_estimators"],
        'clf__estimator__C': HYPERPARAMS["C"],  # Changé de base_estimator à estimator
        'clf__estimator__max_iter': [1000],
        'clf__estimator__class_weight': ['balanced' if is_balanced else None],
        'clf__estimator__random_state': [42]
    }
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)

def train_adaboost_tree(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    # Utiliser un estimateur minimal sans configuration avancée
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        algorithm='SAMME',
        random_state=42
    )
    param_grid = {
        'clf__n_estimators': HYPERPARAMS["n_estimators"],
        'clf__estimator__max_depth': HYPERPARAMS["max_depth"],  # Changé de base_estimator à estimator
        'clf__estimator__min_samples_split': HYPERPARAMS["min_samples_split"],
        'clf__estimator__class_weight': ['balanced' if is_balanced else None],
        'clf__estimator__random_state': [42]
    }
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)


def train_decision_tree(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    model = DecisionTreeClassifier(class_weight='balanced' if is_balanced else None, random_state=42)
    param_grid = {
        'clf__max_depth': HYPERPARAMS["max_depth"],
        'clf__min_samples_split': HYPERPARAMS["min_samples_split"]
    }
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)

def train_gradient_boosting(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'clf__n_estimators': HYPERPARAMS["n_estimators"],
        'clf__max_depth': HYPERPARAMS["max_depth"],
        'clf__min_samples_split': HYPERPARAMS["min_samples_split"],
    }
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)

def train_stacking(X: np.ndarray, y: np.ndarray, cv, is_balanced: bool) -> Dict:

    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]
    
    # Définir l'estimateur final
    final_estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    # Créer le modèle de stacking
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=4,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1,
        verbose=0
    )
    
    # Paramètres pour la recherche par grille avec hyperparamètres pour tous les estimateurs
    param_grid = {
        # Hyperparamètres pour l'estimateur final
        'clf__final_estimator__C': HYPERPARAMS["C"],
        'clf__final_estimator__class_weight': ['balanced' if is_balanced else None],
        
        # Hyperparamètres pour LogisticRegression
        'clf__lr__C': HYPERPARAMS["C"], 
        'clf__lr__class_weight': ['balanced' if is_balanced else None],
        
        # Hyperparamètres pour DecisionTree
        'clf__dt__max_depth': HYPERPARAMS["max_depth"],
        'clf__dt__min_samples_split': HYPERPARAMS["min_samples_split"],
        'clf__dt__class_weight': ['balanced' if is_balanced else None]
    }
    
    return train_model_with_grid_search(model, param_grid, X, y, cv, is_balanced)


# -----------------------------------------------------------------------------
# Liste des modèles avec leurs noms d'affichage et fonctions d'entraînement
# L'ordre détermine l'ordre dans le tableau comparatif
# -----------------------------------------------------------------------------
MODELS = [
    {"name": "Logistic Regression", "train_func": train_logistic_regression},
    {"name": "Bagging LR", "train_func": train_bagging_LR},
    {"name": "AdaBoost LR", "train_func": train_adaboost_LR},
    {"name": "Decision Tree", "train_func": train_decision_tree},
    {"name": "Random Forest", "train_func": train_random_forest},
    {"name": "AdaBoost Tree", "train_func": train_adaboost_tree},
    {"name": "Gradient Boosting", "train_func": train_gradient_boosting},
    {"name": "Stacking", "train_func": train_stacking}
]

# -----------------------------------------------------------------------------
# Fonction d'évaluation pour tous les datasets (sans itérations multiples)
# -----------------------------------------------------------------------------
def evaluate_algorithm_all_datasets(dataset_names: List[str],
                                    data_loader,
                                    train_func) -> Dict:
    results = {}
    # On utilise un KFold fixe pour tous les datasets
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for dataset_name in dataset_names:
        print(f"\n[{train_func.__name__}] Traitement du dataset: {dataset_name}")
        try:
            X, y = data_loader(dataset_name)
            result = train_func(X, y, cv, is_balanced=True)
            results[dataset_name] = result
        except Exception as e:
            logging.error(f"Erreur sur {dataset_name} avec {train_func.__name__}: {e}")
    return results

# -----------------------------------------------------------------------------
# Création d'un tableau comparatif à partir des résultats (meilleur f1 score)
# -----------------------------------------------------------------------------
def create_comparison_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    
    # Vérifie si results_dict est vide
    if not results_dict:
        print("Aucun résultat disponible pour générer un tableau comparatif.")
        return pd.DataFrame(columns=["Dataset"] + [model["name"] for model in MODELS])
    
    # Identification des datasets communs
    datasets = set()
    for model_name, model_results in results_dict.items():
        datasets.update(model_results.keys())
        print('model_name', model_name)
        print('model_results.keys()', model_results.keys())
    common_datasets = sorted(datasets)
    
    # Si aucun dataset commun, retourne un DataFrame vide avec les colonnes appropriées
    if not common_datasets:
        print("Aucun dataset commun trouvé pour générer un tableau comparatif.")
        return pd.DataFrame(columns=["Dataset"] + [model["name"] for model in MODELS])
        
    for dataset in common_datasets:
        scores = {}
        for model in MODELS:
            mname = model["name"]
            if mname in results_dict and dataset in results_dict[mname]:
                res = results_dict[mname][dataset]
                # Utilisation des scores de test
                best_f1 = res["best_model"]["test"]["mean_f1"]
                best_std = res["best_model"]["test"]["std_f1"]
                scores[mname] = (best_f1, best_std)
        
        if scores:
            max_score = max(mean for mean, _ in scores.values())
            row = {"Dataset": dataset}
            for mname, (mean, std) in scores.items():
                formatted = f"{mean * 100:.1f} ± {std * 100:.1f}"
                if mean == max_score:
                    formatted = f"\\textbf{{{formatted}}}"
                row[mname] = formatted
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=["Dataset"] + [model["name"] for model in MODELS])
    
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Bloc principal d'exécution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 all.py [balanced] [imbalanced] [oversampling] [undersampling] [hybridsampling]")
        sys.exit(1)
    
    workflows = set(sys.argv[1:])
    
    # Récupération des datasets et catégorisation
    datasets_info, balanced_datasets, imbalanced_datasets = categorize_datasets(dataset_list)
    
    print(f"\nNombre de datasets équilibrés: {len(balanced_datasets)}")
    print("Datasets équilibrés:", balanced_datasets)
    print(f"\nNombre de datasets déséquilibrés: {len(imbalanced_datasets)}")
    print("Datasets déséquilibrés:", imbalanced_datasets)
    
    workflow_results = {}
    
    for wf in workflows:
        if wf not in DATA_LOADERS:
            print(f"Workflow inconnu: {wf}")
            continue
        print(f"\n--- Évaluation du workflow: {wf} ---")
        datasets = balanced_datasets if wf == "balanced" else imbalanced_datasets
        data_loader = DATA_LOADERS[wf]
        results_by_model = {}
        for model in MODELS:
            model_name = model["name"]
            train_func = model["train_func"]
            print(f"\nTraitement du modèle: {model_name}")
            results_by_model[model_name] = evaluate_algorithm_all_datasets(datasets, data_loader, train_func)
        workflow_results[wf] = results_by_model
    
    # Création des tableaux comparatifs et sauvegarde des résultats pour chaque workflow
    for wf, results_dict in workflow_results.items():
        print(f"\n--- Tableau comparatif pour le workflow: {wf} ---")
        table = create_comparison_table(results_dict)
        print(table)
        os.makedirs("output", exist_ok=True)
        csv_filename = f"output/comparison_{wf}.csv"
        pdf_filename = f"comparison_{wf}.pdf"
        table.to_csv(csv_filename, index=False)
        save_table_to_pdf(table, pdf_filename)
    
    # Sauvegarde détaillée des résultats dans un fichier JSON
    try:
        if os.path.exists("results_merged.json") and os.path.getsize("results_merged.json") > 0:
            try:
                with open("results_merged.json", "r", encoding="utf-8") as f:
                    results_merged = json.load(f)
            except json.JSONDecodeError:
                print("Le fichier results_merged.json existe mais est corrompu. Création d'un nouveau fichier.")
                results_merged = {}
        else:
            results_merged = {}
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier results_merged.json: {e}")
        results_merged = {}

    results_merged.update(workflow_results)

    try:
        with open("results_merged.json", "w", encoding="utf-8") as f:
            json.dump(results_merged, f, indent=4)
        print("Les résultats ont été enregistrés dans 'results_merged.json'.")
    except Exception as e:
        print(f"Erreur lors de l'écriture dans results_merged.json: {e}")