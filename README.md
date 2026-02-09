# Customer Churn Prediction using Machine Learning

This project implements a customer churn prediction system for the telecommunication industry using various machine learning algorithms, based on the research paper "Prediction of Customer Churn Behavior in the Telecommunication Industry Using Machine Learning Models" (Chang et al., 2024).

## 📊 Overview

Customer churn is a significant concern in the telecommunications industry, with annual churn rates exceeding 30%. This project uses ensemble learning models to analyze and forecast customer churn, providing accurate predictions with explainable AI (XAI) techniques.

## 🚀 Features

- **5 ML Models**: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest
- **Hyperparameter Tuning**: GridSearchCV for Random Forest optimization
- **SMOTE**: Handles imbalanced data
- **Explainable AI**: LIME and SHAP visualizations
- **Comprehensive Visualizations**: Confusion matrices, ROC curves, feature importance plots

## 📁 Project Structure

```
Customer_Churn_Prediction_ML/
├── dataset/
│   └── telecom_customer_churn.csv
├── main/
│   ├── main.py                 # Main experiment runner
│   ├── data_loader.py          # Data loading utilities
│   ├── data_preprocessing.py   # Data preprocessing pipeline
│   ├── model_training.py       # Model training functions
│   ├── evaluation.py           # Model evaluation metrics
│   ├── visualization.py        # Plotting functions
│   ├── explainability.py       # LIME and SHAP explanations
│   └── results/                # Generated plots and results
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/doanthetin193/Customer_Churn_Prediction_ML.git
cd Customer_Churn_Prediction_ML
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Usage

Run the main experiment:
```bash
cd main
python main.py
```

This will:
1. Load and preprocess the telecom customer churn dataset
2. Train 5 different ML models
3. Perform hyperparameter tuning on Random Forest
4. Generate evaluation metrics and visualizations
5. Create LIME and SHAP explanations

## 📊 Results

The Random Forest model achieved the best performance:
- **AUC**: ~0.89
- **Accuracy**: ~82%
- **Sensitivity**: ~72%
- **Specificity**: ~86%

### Generated Visualizations (in `main/results/`):
- `confusion_matrices.png` - Confusion matrices for all models
- `roc_curves.png` - ROC curves comparison
- `metrics_comparison.png` - Model performance comparison
- `lime_sample_*.png` - LIME explanations
- `shap_summary.png` - SHAP feature importance (beeswarm plot)
- `shap_bar.png` - Mean SHAP values bar chart

## 📚 Dataset

The dataset contains customer information including:
- Demographics (Age, Gender, Dependents)
- Service usage (Internet, Phone, Streaming)
- Account information (Contract, Monthly Charges, Tenure)
- Churn status

## 🔬 Technologies Used

- Python 3.x
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- LIME, SHAP
- Imbalanced-learn (SMOTE)

## 📖 Reference

Chang, V., Hall, K., Xu, Q.A., Amao, F.O., Ganatra, M.A., & Benson, V. (2024). Prediction of Customer Churn Behavior in the Telecommunication Industry Using Machine Learning Models. *Algorithms*, 17(6), 231.

## 👤 Author

- **Đoàn Thế Tín** - [GitHub](https://github.com/doanthetin193)

## 📄 License

This project is for educational purposes.
