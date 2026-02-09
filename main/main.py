"""
Main Entry Point
Churn Prediction ML Experiment - Complete Pipeline

Based on: "Predicting Customer Churn in the Telecommunications Industry"
Implements: 5 ML Algorithms + Evaluation + XAI (LIME/SHAP)
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, explore_data
from data_preprocessing import preprocess_data
from models import train_all_models, predict_all_models
from evaluation import evaluate_all_models, print_results_table, generate_all_plots
from explainability import generate_explanations

def main():
    """Run complete ML experiment pipeline"""
    print("="*60)
    print("   CUSTOMER CHURN PREDICTION - ML EXPERIMENT")
    print("   Based on Research Paper Implementation")
    print("="*60)
    
    # Setup results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    # Step 1: Load data
    print("\n📁 STEP 1: LOADING DATA")
    df = load_data()
    explore_data(df)
    
    # Step 2: Preprocess data
    print("\n🔧 STEP 2: PREPROCESSING DATA")
    data = preprocess_data(df, use_smote=True)  # Enable SMOTE for better Sensitivity
    
    # Step 3: Train models
    print("\n🤖 STEP 3: TRAINING MODELS")
    trained_models = train_all_models(data['X_train'], data['y_train'])
    
    # Step 4: Make predictions
    print("\n📊 STEP 4: MAKING PREDICTIONS")
    predictions = predict_all_models(trained_models, data['X_test'])
    
    # Step 5: Evaluate models
    print("\n📈 STEP 5: EVALUATING MODELS")
    results = evaluate_all_models(data['y_test'], predictions)
    print_results_table(results)
    
    # Step 6: Generate plots
    print("\n🎨 STEP 6: GENERATING PLOTS")
    generate_all_plots(data['y_test'], predictions, results, results_dir)
    
    # Step 7: XAI Explanations
    print("\n🔍 STEP 7: XAI EXPLANATIONS")
    # Use Random Forest (best model) for explanations
    rf_model = trained_models['Random Forest']
    generate_explanations(
        rf_model,
        data['X_train'],
        data['X_test'],
        data['feature_names'],
        results_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("   ✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Results saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - confusion_matrices.png (Figure 5)")
    print("  - roc_curves.png (Figure 6)")
    print("  - metrics_comparison.png (Figure 7)")
    print("  - lime_sample_*.png (Figure 8)")
    print("  - shap_summary.png (Figure 9)")
    print("  - shap_bar.png (Figure 10)")
    
    return results

if __name__ == "__main__":
    results = main()
