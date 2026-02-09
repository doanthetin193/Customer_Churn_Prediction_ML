"""
Evaluation Module
Metrics, confusion matrix, and ROC curve visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, recall_score
)
import os

def ensure_results_dir(results_dir: str = 'results'):
    """Create results directory if not exists"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate all evaluation metrics"""
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    
    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC': roc_auc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'fpr': fpr,
        'tpr': tpr
    }

def evaluate_all_models(y_test: np.ndarray, predictions: dict) -> dict:
    """Evaluate all models and return metrics"""
    results = {}
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    for name, pred in predictions.items():
        metrics = calculate_metrics(y_test, pred['y_pred'], pred['y_prob'])
        results[name] = metrics
        
        print(f"\n{name}:")
        print(f"  Accuracy:    {metrics['Accuracy']*100:.2f}%")
        print(f"  Sensitivity: {metrics['Sensitivity']*100:.2f}%")
        print(f"  Specificity: {metrics['Specificity']*100:.2f}%")
        print(f"  AUC:         {metrics['AUC']:.4f}")
    
    return results

def print_results_table(results: dict):
    """Print comparison table of all models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'AUC':>8}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Accuracy']*100:>9.2f}% {metrics['Sensitivity']*100:>11.2f}% {metrics['Specificity']*100:>11.2f}% {metrics['AUC']:>8.4f}")
    
    print("-"*70)
    
    # Find best model by AUC
    best_model = max(results.items(), key=lambda x: x[1]['AUC'])
    print(f"\n🏆 Best Model: {best_model[0]} (AUC = {best_model[1]['AUC']:.4f})")

def plot_confusion_matrices(y_test: np.ndarray, predictions: dict, results_dir: str = 'results'):
    """Plot confusion matrices for all models"""
    ensure_results_dir(results_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, pred['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'])
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    # Hide empty subplot
    axes[-1].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models (Figure 5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion_matrices.png")

def plot_roc_curves(results: dict, results_dir: str = 'results'):
    """Plot ROC curves for all models"""
    ensure_results_dir(results_dir)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for idx, (name, metrics) in enumerate(results.items()):
        plt.plot(metrics['fpr'], metrics['tpr'], 
                 color=colors[idx], lw=2,
                 label=f"{name} (AUC = {metrics['AUC']:.4f})")
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models (Figure 6)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved roc_curves.png")

def plot_metrics_comparison(results: dict, results_dir: str = 'results'):
    """Plot bar chart comparing metrics across models (Figure 7)"""
    ensure_results_dir(results_dir)
    
    models = list(results.keys())
    metrics_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    for i, metric in enumerate(metrics_names):
        values = [results[m][metric] for m in models]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Evaluation Comparison (Figure 7)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics_comparison.png")

def generate_all_plots(y_test: np.ndarray, predictions: dict, results: dict, results_dir: str = 'results'):
    """Generate all evaluation plots"""
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    plot_confusion_matrices(y_test, predictions, results_dir)
    plot_roc_curves(results, results_dir)
    plot_metrics_comparison(results, results_dir)
    
    print(f"\n✓ All plots saved to '{results_dir}/' folder")

if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessing import preprocess_data
    from models import train_all_models, predict_all_models
    
    df = load_data()
    data = preprocess_data(df)
    
    trained_models = train_all_models(data['X_train'], data['y_train'])
    predictions = predict_all_models(trained_models, data['X_test'])
    
    results = evaluate_all_models(data['y_test'], predictions)
    print_results_table(results)
    generate_all_plots(data['y_test'], predictions, results)
