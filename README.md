# Imbalanced Classification Project

This project explores various machine learning algorithms and sampling techniques for handling imbalanced datasets. It implements and evaluates different classification models with various balancing strategies to improve performance on minority classes.

## Project Structure

- `main.py`: Core implementation of the machine learning pipeline, including model training, hyperparameter tuning, and evaluation
- `results.ipynb`: Jupyter notebook for analyzing and visualizing results
- `prepdata.py`: Functions for data loading and preprocessing
- `balanceCheck.py`: Functions for analyzing and categorizing datasets based on class imbalance
- `datasets`: Directory containing the datasets used in the project (find the datasets on the github releases)
- `results_merged.json`: File containing the results of the project

## Features

- **Multiple Classification Algorithms**:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - AdaBoost (with both Decision Tree and Logistic Regression base estimators)
  - Gradient Boosting
  - Bagging (with Logistic Regression)
  - Stacking Ensemble

- **Sampling Techniques**:
  - Undersampling (RandomUnderSampler)
  - Oversampling (SMOTE)
  - Hybrid sampling (SMOTEENN)

- **Comprehensive Evaluation**:
  - F1 score optimization
  - Cross-validation
  - Overfitting/underfitting analysis
  - Hyperparameter impact assessment

## Results

The project generates detailed performance comparisons across different datasets and models, with visualizations including:
- Performance tables with F1 scores
- Overfitting/underfitting trend analysis
- Hyperparameter sensitivity analysis

Results are saved in both CSV and PDF formats in the `output` directory, with detailed model information stored in `results_merged.json`.

## Usage

1. Run the main pipeline:
   ```
   python main.py
   ```

2. Analyze results using the Jupyter notebook:
   ```
   jupyter notebook results.ipynb
   ```

## Dependencies

- scikit-learn
- imbalanced-learn
- numpy
- pandas
- matplotlib
- jupyter

## Authors

- Edouard Chappon
- Antoine Sirvent

