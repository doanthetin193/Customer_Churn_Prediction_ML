"""
Explainability Module
LIME and SHAP explanations for model interpretability
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import os

# LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Run: pip install lime")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Run: pip install shap")

def ensure_results_dir(results_dir: str = 'results'):
    """Create results directory if not exists"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def create_lime_explainer(X_train: np.ndarray, feature_names: list, class_names: list = None):
    """Create LIME explainer"""
    if not LIME_AVAILABLE:
        return None
    
    # Paper uses 'Good' for class 0 (Not Churned) and 'Bad' for class 1 (Churned)
    if class_names is None:
        class_names = ['Good', 'Bad']
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    return explainer

def explain_with_lime(model, explainer, X_sample: np.ndarray, 
                      sample_idx: int = 0, num_features: int = 10,
                      results_dir: str = 'results'):
    """Generate LIME explanation for a single sample"""
    if not LIME_AVAILABLE or explainer is None:
        print("LIME not available")
        return None
    
    ensure_results_dir(results_dir)
    
    # Get explanation - use labels=[0] to explain class "Good" like in the paper
    exp = explainer.explain_instance(
        X_sample[sample_idx],
        model.predict_proba,
        num_features=num_features,
        labels=[0]  # Explain class 0 (Good) to match paper's Figure 8
    )
    
    # Save as figure
    fig = exp.as_pyplot_figure(label=0)  # Show explanation for class 0 (Good)
    fig.suptitle(f'LIME Explanation - Sample {sample_idx} (Figure 8)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'lime_sample_{sample_idx}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved lime_sample_{sample_idx}.png")
    return exp

def explain_with_shap(model, X_train: np.ndarray, X_test: np.ndarray,
                      feature_names: list, results_dir: str = 'results',
                      max_samples: int = 100):
    """Generate SHAP explanations matching paper's Figure 9 and 10"""
    if not SHAP_AVAILABLE:
        print("SHAP not available")
        return None
    
    ensure_results_dir(results_dir)
    
    print("Calculating SHAP values (this may take a while)...")
    
    # Use subset for speed
    X_subset = X_test[:max_samples] if len(X_test) > max_samples else X_test
    
    # Create explainer based on model type
    model_name = type(model).__name__
    
    try:
        if model_name == 'RandomForestClassifier':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_subset)
            # For binary classification, use positive class
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values
        else:
            # Use KernelExplainer for other models
            background = shap.sample(X_train, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_subset)
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None
    
    # Ensure feature_names matches X_subset columns
    feature_names = list(feature_names)
    n_features = X_subset.shape[1]
    
    print(f"  SHAP values shape: {shap_values_plot.shape}")
    print(f"  X_subset shape: {X_subset.shape}")
    print(f"  Feature names count: {len(feature_names)}")
    
    if len(feature_names) < n_features:
        feature_names = feature_names + [f'Feature_{i}' for i in range(len(feature_names), n_features)]
    elif len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
    
    # Calculate mean absolute SHAP values for feature importance
    # Only take mean of the 2D array (samples x features)
    n_shap_features = shap_values_plot.shape[1] if len(shap_values_plot.shape) > 1 else 1
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    
    # Ensure 1D and limit to valid feature count
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    mean_abs_shap = mean_abs_shap[:n_shap_features]  # Limit to actual features
    
    # Get top 10 features by importance (limit to available features)
    n_top = min(10, len(mean_abs_shap), n_shap_features)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n_top]
    # Ensure indices are within valid range
    top_indices = top_indices[top_indices < n_shap_features]
    
    # ============ FIGURE 9: Custom Beeswarm Plot ============
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, idx in enumerate(top_indices):
        # Use numpy-safe indexing
        shap_col = np.asarray(shap_values_plot[:, idx]).flatten()
        feature_col = np.asarray(X_subset[:, idx]).flatten()
        
        # Ensure same size
        n_samples = min(len(shap_col), len(feature_col))
        shap_col = shap_col[:n_samples]
        feature_col = feature_col[:n_samples]
        
        # Normalize feature values for coloring (0 to 1)
        if feature_col.max() != feature_col.min():
            norm_values = (feature_col - feature_col.min()) / (feature_col.max() - feature_col.min())
        else:
            norm_values = np.ones(n_samples) * 0.5
        
        # Add y-jitter for visibility
        y_pos = len(top_indices) - 1 - i + np.random.uniform(-0.2, 0.2, n_samples)
        
        # Plot scatter with colormap (blue=low, red=high)
        scatter = ax.scatter(shap_col, y_pos, c=norm_values, cmap='coolwarm', 
                           alpha=0.7, s=20, edgecolors='none')
    
    # Configure axes
    ax.set_yticks(range(len(top_indices)))
    # Flatten and convert to plain Python ints
    top_indices_flat = np.ravel(top_indices).astype(int).tolist()
    feature_labels = [feature_names[i] for i in top_indices_flat]
    ax.set_yticklabels(feature_labels[::-1], fontsize=10)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
    ax.set_title('Summary plots for Random Forest classifier (Figure 9)', 
                fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label('Feature value', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', '', 'High'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved shap_summary.png")
    
    # ============ FIGURE 10: Bar Plot ============
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:10]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = mean_abs_shap[sorted_idx]
    
    # Horizontal bar chart (reversed for top at top)
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importance[::-1], color='#1976D2', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features[::-1], fontsize=10)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=11)
    ax.set_title('SHAP Feature Importance (Figure 10)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # Top feature at top
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved shap_bar.png")
    
    return shap_values

def generate_explanations(model, X_train: np.ndarray, X_test: np.ndarray,
                         feature_names: list, results_dir: str = 'results'):
    """Generate both LIME and SHAP explanations"""
    print("\n" + "="*50)
    print("GENERATING XAI EXPLANATIONS")
    print("="*50)
    
    # LIME for 2 sample cases (similar to Figure 8 in paper)
    if LIME_AVAILABLE:
        explainer = create_lime_explainer(X_train, feature_names)
        
        # Find a churner and non-churner sample
        explain_with_lime(model, explainer, X_test, sample_idx=0, results_dir=results_dir)
        explain_with_lime(model, explainer, X_test, sample_idx=1, results_dir=results_dir)
    
    # SHAP explanations
    if SHAP_AVAILABLE:
        explain_with_shap(model, X_train, X_test, feature_names, results_dir)
    
    print(f"\n✓ XAI explanations saved to '{results_dir}/' folder")

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessing import preprocess_data
    from models import train_all_models
    
    df = load_data()
    data = preprocess_data(df)
    
    trained_models = train_all_models(data['X_train'], data['y_train'])
    
    # Use Random Forest for explanations (best model)
    rf_model = trained_models['Random Forest']
    
    generate_explanations(
        rf_model,
        data['X_train'],
        data['X_test'],
        data['feature_names']
    )
