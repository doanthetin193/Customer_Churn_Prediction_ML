"""
ML Models Module - OPTIMIZED VERSION
Implement 5 classification algorithms with hyperparameter tuning
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def get_models() -> dict:
    """Return dictionary of all 5 models with optimized hyperparameters"""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=0.5,  # Regularization
            solver='lbfgs',
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,  # Optimized from 5
            weights='distance',  # Weight by distance
            metric='minkowski'
        ),
        'Naive Bayes': GaussianNB(
            var_smoothing=1e-8  # Reduced smoothing
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15,  # Increased from 10
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='entropy',  # Information gain
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=15,  # Increased from 10
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def get_tuned_random_forest(X_train, y_train):
    """Perform GridSearchCV to find best Random Forest parameters"""
    print("Tuning Random Forest with GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Train a single model"""
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train: np.ndarray, y_train: np.ndarray, tune_rf: bool = True) -> dict:
    """Train all 5 models and return fitted models"""
    models = get_models()
    trained_models = {}
    
    print("\n" + "="*50)
    print("TRAINING MODELS (OPTIMIZED)")
    print("="*50)
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        
        # Special handling for Random Forest with tuning
        if name == 'Random Forest' and tune_rf:
            print()  # New line before tuning output
            trained_model = get_tuned_random_forest(X_train, y_train)
        else:
            trained_model = train_model(model, X_train, y_train)
        
        trained_models[name] = trained_model
        if name != 'Random Forest' or not tune_rf:
            print("✓")
    
    print(f"\n✓ All {len(trained_models)} models trained successfully!")
    return trained_models

def predict(model, X: np.ndarray) -> tuple:
    """Get predictions and probabilities from a model"""
    y_pred = model.predict(X)
    
    # Get probability of positive class (Churn=1)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred  # Fallback for models without predict_proba
    
    return y_pred, y_prob

def predict_all_models(trained_models: dict, X: np.ndarray) -> dict:
    """Get predictions from all models"""
    predictions = {}
    
    for name, model in trained_models.items():
        y_pred, y_prob = predict(model, X)
        predictions[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    return predictions

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessing import preprocess_data
    
    df = load_data()
    data = preprocess_data(df)
    
    trained_models = train_all_models(data['X_train'], data['y_train'])
    predictions = predict_all_models(trained_models, data['X_test'])
    
    print("\nPredictions generated for all models!")
