# Credit Card Fraud Detection

This repository contains code, notebooks, and assets for building and evaluating models to detect credit card fraud using transaction data. The goal is to provide reproducible experiments, clear training/inference scripts, and example notebooks for exploration.

## Project structure (suggested)

- data/                       # raw and processed datasets (gitignored)
- notebooks/                  # exploratory analysis and experiments (Jupyter notebooks)
- src/                        # training, evaluation and inference scripts
  - train.py                  # training pipeline entrypoint
  - evaluate.py               # evaluation scripts / metrics calculation
  - predict.py                # inference example to run a saved model
- models/                     # saved model artifacts (gitignored)
- requirements.txt            # Python dependencies
- README.md                   # this file

> Note: The exact filenames above are suggestions — adjust to match the repo's layout if they differ.

## Dataset

This project typically uses the "Credit Card Fraud Detection" dataset from Kaggle (European cardholders):
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the original CSV under data/ (e.g. `data/creditcard.csv`). The dataset is imbalanced; expect a very small fraction of transactions to be fraudulent.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, common packages used in this repo include:

```bash
pandas scikit-learn xgboost matplotlib seaborn joblib jupyterlab
```

## Quickstart: Training

A typical training command (example):

```bash
python src/train.py --data data/creditcard.csv --output models/creditcard_model.pkl --seed 42
```

What a training script commonly does:
- Load data and basic preprocessing (scaling, imputation if needed)
- Create train / validation splits using stratification to preserve class ratio
- Handle class imbalance via class weighting, undersampling, oversampling (SMOTE), or specialized algorithms
- Train one or more models (e.g., Logistic Regression, Random Forest, XGBoost)
- Save the best model and evaluation results to `models/` or `artifacts/`

## Quickstart: Inference

Example of running inference with a saved model:

```bash
python src/predict.py --model models/creditcard_model.pkl --input data/sample_transactions.csv --output results/predictions.csv
```

## Evaluation & Metrics

Because of the strong class imbalance, accuracy is not a good metric. Prefer metrics such as:
- Precision, Recall
- F1-score
- Area under the Precision-Recall curve (AUPRC)
- ROC-AUC (useful but can be optimistic for heavily imbalanced data)

Also consider evaluating model performance at specific operating points (thresholds) and reporting confusion matrices for clarity.

## Notebooks

Use the notebooks/ directory for exploratory data analysis and to reproduce experiments. Typical notebooks include:
- notebooks/01-exploration.ipynb
- notebooks/02-feature-engineering.ipynb
- notebooks/03-modeling.ipynb

## Reproducibility

- Pin package versions in `requirements.txt`.
- Seed random number generators when training and splitting data.
- Save preprocessing pipelines (e.g., scikit-learn Pipelines) together with the model so inference uses the same transforms.

## Tips for improving performance

- Try tree-based models (XGBoost, LightGBM) with careful hyperparameter tuning.
- Use cross-validation with stratified folds.
- Experiment with different imbalance strategies (class weights, SMOTE, ADASYN, undersampling).
- Use feature engineering to derive time- and user-based aggregation features if additional metadata is available.

## CI / Tests

Add unit tests for data loaders and small integration tests for training and inference. Consider adding GitHub Actions workflows to run tests and linting on each PR.

## Contributing

Contributions are welcome — please open an issue or a pull request describing proposed changes. Follow the repository's coding style and add tests where appropriate.

## License

This project is provided under the MIT License. Update as appropriate for your project.

## Contact

Created by sakshyasinha. For questions or collaboration, open an issue or contact the maintainer on GitHub.
