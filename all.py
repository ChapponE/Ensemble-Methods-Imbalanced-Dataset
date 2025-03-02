#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import time
from typing import Dict, List, Tuple
import subprocess
import os
import random

import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier, StackingClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # Baseline : SVM linéaire

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Importation des fonctions de récupération et d'analyse des datasets
from prepdata import data_recovery
from balanceCheck import categorize_datasets, split_dataset

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The SAMME.R algorithm')

# Configuration du logging
logging.basicConfig(filename='all.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# Helper function to save a DataFrame as a PDF using LaTeX
# -----------------------------------------------------------------------------
def save_table_to_pdf(table: pd.DataFrame, pdf_filename: str):
    """
    Convertit un DataFrame en tableau LaTeX, crée un document LaTeX complet
    et le compile en PDF avec pdflatex. Le tableau occupe toute la largeur de la page
    et est aligné à gauche.
    """
    latex_document = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
\begin{flushleft}
\resizebox{\textwidth}{!}{%%
%s
}
\end{flushleft}
\end{document}
""" % table.to_latex(index=False, escape=False)

    tex_filename = pdf_filename.replace(".pdf", ".tex")

    with open(tex_filename, "w") as f:
        f.write(latex_document)

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename],
                       check=True, stdout=subprocess.DEVNULL)
        print(f"PDF '{pdf_filename}' généré avec succès.")
    except Exception as e:
        logging.error(f"Erreur pendant la compilation PDF de {tex_filename}: {e}")
        print(f"Erreur pendant la compilation PDF : {e}")

    # Suppression des fichiers auxiliaires générés par pdflatex
    for ext in [".aux", ".log", ".tex"]:
        aux_file = pdf_filename.replace(".pdf", ext)
        if os.path.exists(aux_file):
            os.remove(aux_file)


# -----------------------------------------------------------------------------
# Fonctions de chargement des données (selon la méthode d'échantillonnage)
# -----------------------------------------------------------------------------
def get_balanced_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    return data_recovery(dataset_name)


def get_oversampled_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = data_recovery(dataset_name)
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


def get_undersampled_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = data_recovery(dataset_name)
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X, y)


def get_hybridsampled_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = data_recovery(dataset_name)
    sampler = SMOTEENN(random_state=42)
    return sampler.fit_resample(X, y)


# -----------------------------------------------------------------------------
# Fonctions d'entraînement pour chaque classifieur
# -----------------------------------------------------------------------------
# def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
#                               X_val: np.ndarray, y_val: np.ndarray,
#                               cv_splits: List, is_balanced: bool) -> Dict:
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
#     model = LogisticRegression(max_iter=1000,
#                                class_weight='balanced' if is_balanced else None)
#     grid_search = GridSearchCV(estimator=model,
#                                param_grid=param_grid,
#                                cv=cv_splits,
#                                scoring='f1',
#                                n_jobs=-1)
#     grid_search.fit(X_train_scaled, y_train)
#     best_model = grid_search.best_estimator_
#     y_pred = best_model.predict(X_val_scaled)
#     validation_scores = {
#         'f1': f1_score(y_val, y_pred, average='macro'),
#         'accuracy': accuracy_score(y_val, y_pred),
#         'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
#         'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
#     }
#     return {'best_model': best_model,
#             'best_params': grid_search.best_params_,
#             'validation_scores': validation_scores}


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        cv_splits: List, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    model = RandomForestClassifier(class_weight='balanced' if is_balanced else None,
                                   random_state=42)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv_splits,
                               scoring='f1',
                               n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': grid_search.best_params_,
            'validation_scores': validation_scores}


def train_adaboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   cv_splits, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    base_estimator = SVC(random_state=42, kernel='linear', class_weight='balanced' if is_balanced else None)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    model = AdaBoostClassifier(estimator=base_estimator,
                               algorithm='SAMME',
                               random_state=42)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv_splits,
                               scoring='f1',
                               n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': grid_search.best_params_,
            'validation_scores': validation_scores}


def train_bagging_SVC(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  cv_splits: List, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    base_estimator = SVC(random_state=42, kernel='linear', class_weight='balanced' if is_balanced else None)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 1.0]
    }
    model = BaggingClassifier(estimator=base_estimator,
                              random_state=42)
    try:
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=cv_splits,
                                   scoring='f1',
                                   n_jobs=-1,
                                   error_score=np.nan)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    except Exception as e:
        logging.error(f"Grid search failed for bagging: {e}")
        best_model = BaggingClassifier(estimator=base_estimator,
                                       n_estimators=50,
                                       max_samples=1.0,
                                       max_features=1.0,
                                       random_state=42)
        best_model.fit(X_train_scaled, y_train)
        best_params = {"n_estimators": 50, "max_samples": 1.0, "max_features": 1.0}
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': best_params,
            'validation_scores': validation_scores}

def train_bagging_tree(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       cv_splits: List, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    base_estimator = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced' if is_balanced else None
    )

    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 1.0],
        'base_estimator__max_depth': [None, 5, 10, 15],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__min_samples_leaf': [1, 2, 4]}
    
    model = BaggingClassifier(estimator=base_estimator, random_state=42)
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_splits,
            scoring='f1',
            n_jobs=-1,
            error_score=np.nan
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    except Exception as e:
        logging.error(f"Grid search failed for bagging tree: {e}")
        best_model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=50,
            max_samples=1.0,
            max_features=1.0,
            random_state=42)
        best_model.fit(X_train_scaled, y_train)
        best_params = {"n_estimators": 50, "max_samples": 1.0, "max_features": 1.0}
    
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)}
    return {'best_model': best_model,
            'best_params': best_params,
            'validation_scores': validation_scores}

def train_stacking(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   cv_splits: List, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Définition des base learners avec AdaBoost en SAMME
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced' if is_balanced else None)),
        ('dt', DecisionTreeClassifier(class_weight='balanced' if is_balanced else None, random_state=42)),
        ('svm', SVC(random_state=42, kernel='linear', class_weight='balanced' if is_balanced else None))
    ]
    final_estimator = LogisticRegression(max_iter=1000,
                                         class_weight='balanced' if is_balanced else None)
    model = StackingClassifier(estimators=estimators,
                               final_estimator=final_estimator,
                               n_jobs=-1,
                               cv=5)
    param_grid = {'final_estimator__C': [0.01, 0.1, 1, 10, 100]}
    try:
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='f1',
                                   n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    except Exception as e:
        logging.error(f"Grid search failed for stacking: {e}")
        best_model = model
        best_model.fit(X_train_scaled, y_train)
        best_params = {}
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': best_params,
            'validation_scores': validation_scores}


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            cv_splits: List, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [1, 3, 5, 7]
    }
    model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv_splits,
                               scoring='f1',
                               n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': grid_search.best_params_,
            'validation_scores': validation_scores}


# -----------------------------------------------------------------------------
# Fonction de la baseline : SVM linéaire
# -----------------------------------------------------------------------------
def train_linear_svm(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     cv_splits: List, is_balanced: bool) -> Dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    model = SVC(kernel='linear', class_weight='balanced' if is_balanced else None)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv_splits,
                               scoring='f1',
                               n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': grid_search.best_params_,
            'validation_scores': validation_scores}


# -----------------------------------------------------------------------------
# Nouvelle baseline : Decision Tree
# -----------------------------------------------------------------------------
def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        cv_splits: List, is_balanced: bool) -> Dict:
    # Pour un arbre de décision, la normalisation n'est pas indispensable,
    # mais nous l'appliquons ici pour conserver la cohérence avec les autres modèles.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = DecisionTreeClassifier(class_weight='balanced' if is_balanced else None,
                                   random_state=42)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=cv_splits,
                               scoring='f1',
                               n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    validation_scores = {
        'f1': f1_score(y_val, y_pred, average='macro'),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='macro', zero_division=0)
    }
    return {'best_model': best_model,
            'best_params': grid_search.best_params_,
            'validation_scores': validation_scores}


# -----------------------------------------------------------------------------
# Fonction générique d'évaluation sur l'ensemble des datasets (parallélisée)
# -----------------------------------------------------------------------------
def evaluate_algorithm_all_datasets(dataset_names: List[str],
                                    data_loader,
                                    train_func) -> Dict:
    from joblib import Parallel, delayed

    base_seed = 42  # graine de départ pour cette méthode

    # Fonction interne pour traiter un dataset
    def process_dataset(dataset_name: str):
        print(f"\n[{train_func.__name__}] Traitement du dataset: {dataset_name}")
        try:
            X, y = data_loader(dataset_name)
            iteration_results = []
            n_iterations = 10
            for i in range(n_iterations):
                current_seed = base_seed + i
                random.seed(current_seed)
                np.random.seed(current_seed)
                splits = split_dataset(X, y, test_size=0.2, val_size=0.25,
                                       n_splits=5, random_state=current_seed)
                result = train_func(
                    X_train=splits['X_train'],
                    y_train=splits['y_train'],
                    X_val=splits['X_val'],
                    y_val=splits['y_val'],
                    cv_splits=splits['cv_splits'],
                    is_balanced=True
                )
                iteration_results.append(result['validation_scores'])
            mean_scores = {metric: np.mean([res[metric] for res in iteration_results])
                           for metric in iteration_results[0]}
            std_scores = {metric: np.std([res[metric] for res in iteration_results])
                          for metric in iteration_results[0]}
            return dataset_name, mean_scores, std_scores, result['best_params']
        except Exception as e:
            logging.error(f"Erreur sur {dataset_name} avec {train_func.__name__}: {e}")
            return dataset_name, None, None, None

    # Parallélisation sur tous les datasets
    results = Parallel(n_jobs=-1)(delayed(process_dataset)(ds) for ds in dataset_names)

    all_results = {}
    for dataset_name, mean_scores, std_scores, best_params in results:
        if mean_scores is not None:
            all_results[dataset_name] = {
                'mean_scores': mean_scores,
                'std_scores': std_scores,
                'best_params': best_params
            }
    return all_results


def create_comparison_table(results_lr: Dict, results_rf: Dict, results_adb: Dict,
                            results_bag: Dict, results_stack: Dict,
                            results_gb: Dict, results_svm: Dict, results_dt: Dict) -> pd.DataFrame:
    rows = []
    all_datasets = sorted(set(results_lr.keys()).intersection(
        results_rf.keys(), results_adb.keys(),
        results_bag.keys(), results_stack.keys(), results_gb.keys(),
        results_svm.keys(), results_dt.keys()))
    for dataset_name in all_datasets:
        lr_mean_f1 = results_lr[dataset_name]['mean_scores']['f1']
        lr_std_f1 = results_lr[dataset_name]['std_scores']['f1']
        rf_mean_f1 = results_rf[dataset_name]['mean_scores']['f1']
        rf_std_f1 = results_rf[dataset_name]['std_scores']['f1']
        adb_mean_f1 = results_adb[dataset_name]['mean_scores']['f1']
        adb_std_f1 = results_adb[dataset_name]['std_scores']['f1']
        bag_mean_f1 = results_bag[dataset_name]['mean_scores']['f1']
        bag_std_f1 = results_bag[dataset_name]['std_scores']['f1']
        stack_mean_f1 = results_stack[dataset_name]['mean_scores']['f1']
        stack_std_f1 = results_stack[dataset_name]['std_scores']['f1']
        gb_mean_f1 = results_gb[dataset_name]['mean_scores']['f1']
        gb_std_f1 = results_gb[dataset_name]['std_scores']['f1']
        svm_mean_f1 = results_svm[dataset_name]['mean_scores']['f1']
        svm_std_f1 = results_svm[dataset_name]['std_scores']['f1']
        dt_mean_f1 = results_dt[dataset_name]['mean_scores']['f1']
        dt_std_f1 = results_dt[dataset_name]['std_scores']['f1']

        # Dictionnaire récapitulatif des scores moyens pour chaque modèle
        scores = {
            'Logistic Regression': lr_mean_f1,
            'Random Forest': rf_mean_f1,
            'AdaBoost': adb_mean_f1,
            'Bagging': bag_mean_f1,
            'Stacking': stack_mean_f1,
            'Gradient Boosting': gb_mean_f1,
            'SVM Linéaire': svm_mean_f1,
            'Decision Tree': dt_mean_f1
        }
        # Détermine le score maximum dans la ligne
        max_score = max(scores.values())

        # Fonction de formatage d'un résultat, en gras s'il correspond au score maximum
        def format_result(mean, std):
            formatted = f"{mean * 100:.1f} ± {std * 100:.1f}"
            if mean == max_score:
                return r"\textbf{" + formatted + "}"
            else:
                return formatted

        row = {
            'Dataset': dataset_name,
            'Logistic Regression': format_result(lr_mean_f1, lr_std_f1),
            'Random Forest': format_result(rf_mean_f1, rf_std_f1),
            'AdaBoost': format_result(adb_mean_f1, adb_std_f1),
            'Bagging': format_result(bag_mean_f1, bag_std_f1),
            'Stacking': format_result(stack_mean_f1, stack_std_f1),
            'Gradient Boosting': format_result(gb_mean_f1, gb_std_f1),
            'SVM Linéaire': format_result(svm_mean_f1, svm_std_f1),
            'Decision Tree': format_result(dt_mean_f1, dt_std_f1)
        }
        rows.append(row)
    df_comparison = pd.DataFrame(rows)
    return df_comparison


# -----------------------------------------------------------------------------
# Bloc principal
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Vérification des arguments
    if len(sys.argv) < 2:
        print("Usage: python3 all.py [balanced] [oversampling] [undersampling] [hybridsampling]")
        sys.exit(1)

    # Récupération des arguments passés en ligne de commande
    workflows = set(sys.argv[1:])

    # Fixation de la graine globale pour la reproductibilité (pour le reste du code)
    random.seed(42)
    np.random.seed(42)

    # Définition de la liste complète des datasets
    dataset_list = [
        'abalone8', 'abalone17', 'autompg', 'australian', 'balance',
        'bupa', 'german', 'glass', 'hayes', 'heart', 'iono',
        'libras', 'newthyroid', 'pageblocks', 'pima', 'segmentation',
        'sonar', 'spambase', 'splice', 'vehicle', 'wdbc', 'wine', 'wine4',
        'yeast3', 'yeast6'
    ]

    # Séparation des datasets en équilibrés et déséquilibrés
    datasets_info, balanced_datasets, imbalanced_datasets = categorize_datasets(dataset_list)

    print(f"\nNombre de datasets équilibrés: {len(balanced_datasets)}")
    print("Datasets équilibrés:", balanced_datasets)

    print(f"\nNombre de datasets déséquilibrés: {len(imbalanced_datasets)}")
    print("Datasets déséquilibrés:", imbalanced_datasets)

    # -------------------------------------------------------------------------
    # 1) Workflow sur les datasets équilibrés
    if "balanced" in workflows:
        print("\nÉvaluation sur les datasets équilibrés")
        # results_lr_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_logistic_regression)
        results_rf_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_random_forest)
        results_adb_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_adaboost)
        results_bagsvc_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_bagging_SVC)
        results_bagtree_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_bagging_SVC)
        results_stack_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_stacking)
        results_gb_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_gradient_boosting)
        results_svm_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_linear_svm)
        results_dt_bal = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_decision_tree)

        table_balanced = create_comparison_table(results_rf_bal, results_adb_bal, results_bagsvc_bal, 
                                                 results_bagtree_bal, results_stack_bal, results_gb_bal, results_svm_bal, results_dt_bal)
        print("\nTableau comparatif pour les datasets équilibrés:")
        print(table_balanced)
        table_balanced.to_csv("output/comparison_balanced.csv", index=False)
        save_table_to_pdf(table_balanced, "comparison_balanced.pdf")

    # -------------------------------------------------------------------------
    # 2) Workflow sur les datasets oversamplés
    if "oversampling" in workflows:
        print("\nÉvaluation sur les datasets oversamplés")
        #results_lr_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_logistic_regression)
        results_rf_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_random_forest)
        results_adb_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_adaboost)
        results_bagsvc_over = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_bagging_SVC)
        results_bagtree_over = evaluate_algorithm_all_datasets(balanced_datasets, get_balanced_data, train_bagging_tree)
        results_stack_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_stacking)
        results_gb_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_gradient_boosting)
        results_svm_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_linear_svm)
        results_dt_over = evaluate_algorithm_all_datasets(imbalanced_datasets, get_oversampled_data, train_decision_tree)

        table_oversampled = create_comparison_table(results_rf_over, results_adb_over, results_bagsvc_over, 
                                                    results_bagtree_over, results_stack_over, results_gb_over, results_svm_over, results_dt_over)
        print("\nTableau comparatif pour les datasets oversamplés:")
        print(table_oversampled)
        table_oversampled.to_csv("output/comparison_oversampled.csv", index=False)
        save_table_to_pdf(table_oversampled, "comparison_oversampled.pdf")

    # -------------------------------------------------------------------------
    # 3) Workflow sur les datasets undersamplés
    if "undersampling" in workflows:
        print("\nÉvaluation sur les datasets undersamplés")
        #results_lr_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_logistic_regression)
        results_rf_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_random_forest)
        results_adb_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_adaboost)
        results_bagsvc_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_bagging_SVC)
        results_bagtree_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_bagging_tree)
        results_stack_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_stacking)
        results_gb_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_gradient_boosting)
        results_svm_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_linear_svm)
        results_dt_under = evaluate_algorithm_all_datasets(imbalanced_datasets, get_undersampled_data, train_decision_tree)

        table_undersampled = create_comparison_table(results_rf_under, results_adb_under, results_bagsvc_under, 
                                                     results_bagtree_under, results_stack_under, results_gb_under, results_svm_under, results_dt_under)
        print("\nTableau comparatif pour les datasets undersamplés:")
        print(table_undersampled)
        table_undersampled.to_csv("output/comparison_undersampled.csv", index=False)
        save_table_to_pdf(table_undersampled, "comparison_undersampled.pdf")

    # -------------------------------------------------------------------------
    # 4) Workflow sur les datasets hybridsamplés
    if "hybridsampling" in workflows:
        print("\nÉvaluation sur les datasets hybridsamplés")
        #results_lr_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_logistic_regression)
        results_rf_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_random_forest)
        results_adb_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_adaboost)
        results_bagsvc_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_bagging_SVC)
        results_bagtree_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_bagging_tree)
        results_stack_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_stacking)
        results_gb_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_gradient_boosting)
        results_svm_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_linear_svm)
        results_dt_hybrid = evaluate_algorithm_all_datasets(imbalanced_datasets, get_hybridsampled_data, train_decision_tree)

        table_hybridsampled = create_comparison_table(results_rf_hybrid, results_adb_hybrid, results_bagsvc_hybrid, 
                                                      results_bagtree_hybrid, results_stack_hybrid, results_gb_hybrid, results_svm_hybrid, results_dt_hybrid)
        print("\nTableau comparatif pour les datasets hybridsamplés:")
        print(table_hybridsampled)
        table_hybridsampled.to_csv("output/comparison_hybridsampled.csv", index=False)
        save_table_to_pdf(table_hybridsampled, "comparison_hybridsampled.pdf")
