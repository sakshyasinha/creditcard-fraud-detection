# Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions using the [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## 📊 Dataset
- 284,807 transactions
- 492 fraud cases (highly imbalanced)
- Features are anonymized (PCA-transformed), with `Amount` and `Time` included

## 🛠️ Tools & Tech
- Python
- Pandas, Seaborn, Matplotlib
- Scikit-learn
- Random Forest Classifier

## 💡 Approach
1. Loaded and preprocessed data
2. Visualized class imbalance
3. Normalized `Amount` column
4. Trained a `RandomForestClassifier`
5. Evaluated with classification report and confusion matrix
