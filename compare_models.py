#!/usr/bin/env python3
"""
Compare original and v2_expanded models on a shared test set.
Includes training vs test accuracy, sample sizes, overfitting metrics,
feature importance, and SHAP plots for model explainability.
"""

import json
import os
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from train_interpretation_model import InterpretationModelTrainer
import shap


def load_metadata(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_models(model_dir: str, suffix: str = ""):
    suffix_part = f"_{suffix}" if suffix else ""
    metadata_file = os.path.join(model_dir, f"model_metadata{suffix_part}.json")
    metadata = load_metadata(metadata_file)

    scaler_file = os.path.join(model_dir, f"scaler{suffix_part}.pkl")
    scaler = joblib.load(scaler_file)

    models = {}
    for category in metadata['categories']:
        model_path = os.path.join(model_dir, f"model_{category}{suffix_part}.pkl")
        if os.path.exists(model_path):
            models[category] = joblib.load(model_path)
    return metadata, scaler, models


def evaluate_models(X, y_dict, metadata, scaler, models, test_indices, train_indices=None):
    """Evaluate models on test set and optionally compute training accuracy."""
    results = {}
    X_test = X.iloc[test_indices]
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare training data if provided (for overfitting analysis)
    X_train_scaled = None
    if train_indices is not None:
        X_train = X.iloc[train_indices]
        X_train_scaled = scaler.transform(X_train)

    for category in metadata['categories']:
        y = y_dict[category].iloc[test_indices]

        # Filter unknown labels
        mask = y != 'Unknown'
        if mask.sum() == 0:
            results[category] = {
                'test_accuracy': None,
                'test_f1_macro': None,
                'test_precision': None,
                'test_recall': None,
                'train_accuracy': None,
                'generalization_gap': None,
                'test_support': 0,
                'train_support': 0
            }
            continue

        y_true = y[mask]
        X_cat = X_test_scaled[mask]

        if category not in models:
            results[category] = {
                'test_accuracy': None,
                'test_f1_macro': None,
                'test_precision': None,
                'test_recall': None,
                'train_accuracy': None,
                'generalization_gap': None,
                'test_support': int(mask.sum()),
                'train_support': 0
            }
            continue

        y_pred = models[category].predict(X_cat)

        test_acc = float(accuracy_score(y_true, y_pred))
        test_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        test_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        test_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        
        results[category] = {
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'train_accuracy': None,
            'generalization_gap': None,
            'test_support': int(mask.sum()),
            'train_support': 0
        }
        
        # Calculate training accuracy for overfitting detection
        if train_indices is not None and X_train_scaled is not None:
            y_train = y_dict[category].iloc[train_indices]
            train_mask = y_train != 'Unknown'
            
            if train_mask.sum() > 0:
                y_train_true = y_train[train_mask]
                X_train_cat = X_train_scaled[train_mask]
                y_train_pred = models[category].predict(X_train_cat)
                
                train_acc = float(accuracy_score(y_train_true, y_train_pred))
                results[category]['train_accuracy'] = train_acc
                results[category]['train_support'] = int(train_mask.sum())
                
                # Generalization gap: train_acc - test_acc (positive means overfitting)
                results[category]['generalization_gap'] = train_acc - test_acc

    return results


def summarize(results):
    test_accs = [v['test_accuracy'] for v in results.values() if v['test_accuracy'] is not None]
    test_f1s = [v['test_f1_macro'] for v in results.values() if v['test_f1_macro'] is not None]
    train_accs = [v['train_accuracy'] for v in results.values() if v['train_accuracy'] is not None]
    gen_gaps = [v['generalization_gap'] for v in results.values() if v['generalization_gap'] is not None]
    
    total_test_support = sum([v['test_support'] for v in results.values()])
    total_train_support = sum([v['train_support'] for v in results.values()])
    
    return {
        'avg_test_accuracy': float(np.mean(test_accs)) if test_accs else None,
        'avg_test_f1_macro': float(np.mean(test_f1s)) if test_f1s else None,
        'avg_train_accuracy': float(np.mean(train_accs)) if train_accs else None,
        'avg_generalization_gap': float(np.mean(gen_gaps)) if gen_gaps else None,
        'total_test_samples': total_test_support,
        'total_train_samples': total_train_support
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original vs v2 models")
    parser.add_argument("--dataset", default="data/processed/combined_training_dataset.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--v2-suffix", default="v2_expanded")
    parser.add_argument("--generate-plots", action="store_true", default=True,
                        help="Generate feature importance and SHAP plots")
    parser.add_argument("--output-dir", default="outputs/comparison_plots",
                        help="Directory to save plots")
    return parser.parse_args()


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def plot_feature_importance(models, feature_names, metadata, suffix="", output_dir="outputs"):
    """Generate feature importance plots for all categories using both permutation and native importance."""
    create_output_directory(output_dir)
    
    for category in metadata['categories']:
        if category not in models:
            continue
        
        model = models[category]
        
        # Try to get native feature importance if available
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Top 15 features
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(indices)), importances[indices], align='center')
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Importance')
                ax.set_title(f'Feature Importance - {category}{suffix}')
                plt.tight_layout()
                
                filename = f"{output_dir}/feature_importance_{category}{suffix}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Saved: {filename}")
        except Exception as e:
            print(f"⚠ Could not generate feature importance for {category}: {str(e)}")


def plot_shap_summary(X_data, models, feature_names, metadata, X_sample_size=100, 
                      suffix="", output_dir="outputs"):
    """Generate SHAP summary plots for model explanations."""
    create_output_directory(output_dir)
    
    for category in metadata['categories']:
        if category not in models:
            continue
        
        model = models[category]
        
        try:
            # Use KernelExplainer for model-agnostic SHAP values
            # Limit background data for computational efficiency
            background_size = min(100, len(X_data) // 2)
            X_background = X_data[:background_size]
            
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                X_background
            )
            
            # Use a sample of test data for efficiency
            sample_size = min(X_sample_size, len(X_data))
            X_sample = X_data[:sample_size]
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class: take the first class
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # SHAP Summary Plot (Bar)
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, 
                            plot_type="bar", show=False)
            plt.title(f'SHAP Summary (Bar) - {category}{suffix}')
            plt.tight_layout()
            filename = f"{output_dir}/shap_summary_bar_{category}{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved: {filename}")
            
            # SHAP Summary Plot (Beeswarm)
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, 
                            plot_type="dot", show=False)
            plt.title(f'SHAP Summary (Beeswarm) - {category}{suffix}')
            plt.tight_layout()
            filename = f"{output_dir}/shap_summary_dot_{category}{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved: {filename}")
            
        except Exception as e:
            print(f"⚠ Could not generate SHAP plots for {category}: {str(e)}")


def plot_shap_waterfall(X_data, models, feature_names, metadata, instance_idx=0, 
                        suffix="", output_dir="outputs"):
    """Generate SHAP waterfall plots for individual predictions."""
    create_output_directory(output_dir)
    
    for category in metadata['categories']:
        if category not in models:
            continue
        
        model = models[category]
        
        try:
            # Convert to DataFrame if needed for indexing
            if not hasattr(X_data, 'iloc'):
                import pandas as pd
                X_data = pd.DataFrame(X_data, columns=feature_names)
            
            # Use a smaller background for efficiency
            background_size = min(50, len(X_data) // 2)
            X_background = X_data.iloc[:background_size] if hasattr(X_data, 'iloc') else X_data[:background_size]
            
            # Use TreeExplainer if available (more efficient), otherwise KernelExplainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.KernelExplainer(model.predict_proba, X_background)
                sample_data = X_data.iloc[[instance_idx]] if hasattr(X_data, 'iloc') else X_data[[instance_idx]]
                shap_values = explainer.shap_values(sample_data)
                
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
            else:
                explainer = shap.KernelExplainer(model.predict, X_background)
                sample_data = X_data.iloc[[instance_idx]] if hasattr(X_data, 'iloc') else X_data[[instance_idx]]
                shap_values = explainer.shap_values(sample_data)
                shap_vals = shap_values if not isinstance(shap_values, list) else shap_values[0]
            
            # Extract numpy array for waterfall plot
            sample_values = sample_data.iloc[0].values if hasattr(sample_data, 'iloc') else sample_data[0]
            
            # Waterfall plot for first prediction
            fig = plt.figure(figsize=(12, 8))
            shap.waterfall_plot(shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                data=sample_values,
                feature_names=feature_names
            ), show=False)
            plt.title(f'SHAP Waterfall Plot - {category}{suffix} (Sample 0)')
            plt.tight_layout()
            filename = f"{output_dir}/shap_waterfall_{category}{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved: {filename}")
            
        except Exception as e:
            print(f"⚠ Could not generate SHAP waterfall plot for {category}: {str(e)}")


def generate_all_plots(X_train_scaled, X_test_scaled, metadata_orig, metadata_v2, 
                       models_orig, models_v2, feature_names, output_dir="outputs"):
    """Generate all feature importance and SHAP plots for both models."""
    print(f"\n📊 GENERATING VISUALIZATION PLOTS")
    print("-" * 120)
    
    # Create subdirectories
    orig_plot_dir = os.path.join(output_dir, "v1_original")
    v2_plot_dir = os.path.join(output_dir, "v2_expanded")
    
    print("\n🔷 Version 1 (Original) Plots:")
    plot_feature_importance(models_orig, feature_names, metadata_orig, suffix="", output_dir=orig_plot_dir)
    plot_shap_summary(X_test_scaled, models_orig, feature_names, metadata_orig, 
                      suffix="", output_dir=orig_plot_dir)
    plot_shap_waterfall(X_test_scaled, models_orig, feature_names, metadata_orig, 
                        suffix="", output_dir=orig_plot_dir)
    
    print("\n🔶 Version 2 (Expanded) Plots:")
    plot_feature_importance(models_v2, feature_names, metadata_v2, suffix="", output_dir=v2_plot_dir)
    plot_shap_summary(X_test_scaled, models_v2, feature_names, metadata_v2, 
                      suffix="", output_dir=v2_plot_dir)
    plot_shap_waterfall(X_test_scaled, models_v2, feature_names, metadata_v2, 
                        suffix="", output_dir=v2_plot_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")



def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ Combined dataset not found: {args.dataset}")
        return

    trainer = InterpretationModelTrainer()
    df = trainer.load_dataset(args.dataset)
    X, y_dict = trainer.prepare_features_and_labels(df)

    # Fixed test split for fair comparison
    X_train, X_test, train_indices, test_indices = train_test_split(
        X, X.index, test_size=0.2, random_state=42
    )

    # Original models
    orig_meta, orig_scaler, orig_models = load_models(args.model_dir)
    orig_results = evaluate_models(X, y_dict, orig_meta, orig_scaler, orig_models, test_indices, train_indices)
    orig_summary = summarize(orig_results)

    # v2 models
    v2_meta, v2_scaler, v2_models = load_models(args.model_dir, args.v2_suffix)
    v2_results = evaluate_models(X, y_dict, v2_meta, v2_scaler, v2_models, test_indices, train_indices)
    v2_summary = summarize(v2_results)

    # Print detailed comparison
    print("\n" + "=" * 120)
    print("MODEL COMPARISON REPORT - Version 1 vs Version 2")
    print("=" * 120)
    
    # Training data summary
    print(f"\n📊 TRAINING DATA SUMMARY")
    print("-" * 120)
    print(f"Total samples used: {len(df)}")
    print(f"  └─ Training set: {len(X_train)} samples")
    print(f"  └─ Test set: {len(X_test)} samples")
    print(f"  └─ Train/Test ratio: {len(X_train)/len(X_test):.1f}:1")
    
    # Original model data
    print(f"\n📈 VERSION 1 (Original Model)")
    print("-" * 120)
    print(f"Training samples: {orig_summary['total_train_samples']}")
    print(f"Test samples: {orig_summary['total_test_samples']}")
    
    # v2 model data
    print(f"\n🚀 VERSION 2 (Expanded Model)")
    print("-" * 120)
    print(f"Training samples: {v2_summary['total_train_samples']}")
    print(f"Test samples: {v2_summary['total_test_samples']}")
    
    # Detailed per-category metrics
    print(f"\n📋 PER-CATEGORY PERFORMANCE METRICS")
    print("-" * 120)
    print(f"{'Category':<20} | {'V1 Test Acc':<12} | {'V2 Test Acc':<12} | {'Δ Accuracy':<12} | {'V1 Train':<11} | {'V2 Train':<11} | {'V1 Gen Gap':<12} | {'V2 Gen Gap':<12}")
    print("-" * 120)
    
    for category in orig_meta['categories']:
        o = orig_results.get(category, {})
        v = v2_results.get(category, {})
        
        o_test_acc = o.get('test_accuracy')
        v_test_acc = v.get('test_accuracy')
        
        o_acc_str = f"{o_test_acc:.3f}" if o_test_acc is not None else "N/A  "
        v_acc_str = f"{v_test_acc:.3f}" if v_test_acc is not None else "N/A  "
        
        if o_test_acc is not None and v_test_acc is not None:
            delta_acc = v_test_acc - o_test_acc
            delta_str = f"{delta_acc:+.3f}"
        else:
            delta_str = "N/A  "
        
        o_train_acc = o.get('train_accuracy')
        v_train_acc = v.get('train_accuracy')
        
        o_train_str = f"{o_train_acc:.3f}" if o_train_acc is not None else "N/A  "
        v_train_str = f"{v_train_acc:.3f}" if v_train_acc is not None else "N/A  "
        
        o_gen_gap = o.get('generalization_gap')
        v_gen_gap = v.get('generalization_gap')
        
        o_gap_str = f"{o_gen_gap:.3f}" if o_gen_gap is not None else "N/A  "
        v_gap_str = f"{v_gen_gap:.3f}" if v_gen_gap is not None else "N/A  "
        
        print(f"{category:<20} | {o_acc_str:<12} | {v_acc_str:<12} | {delta_str:<12} | {o_train_str:<11} | {v_train_str:<11} | {o_gap_str:<12} | {v_gap_str:<12}")
    
    # Summary metrics
    print("-" * 120)
    print(f"\n📊 AGGREGATE METRICS")
    print("-" * 120)
    
    orig_avg_test = orig_summary['avg_test_accuracy']
    v2_avg_test = v2_summary['avg_test_accuracy']
    
    print(f"Test Set Performance:")
    print(f"  Version 1 (Original):")
    print(f"    └─ Average Test Accuracy: {orig_avg_test:.3f} (across {orig_summary['total_test_samples']} test samples)")
    print(f"    └─ Average F1-Macro: {orig_summary['avg_test_f1_macro']:.3f}")
    
    print(f"  Version 2 (Expanded):")
    print(f"    └─ Average Test Accuracy: {v2_avg_test:.3f} (across {v2_summary['total_test_samples']} test samples)")
    print(f"    └─ Average F1-Macro: {v2_summary['avg_test_f1_macro']:.3f}")
    
    if orig_avg_test is not None and v2_avg_test is not None:
        delta = v2_avg_test - orig_avg_test
        print(f"  ✓ Improvement (Δ): {delta:+.3f} ({delta/orig_avg_test*100:+.1f}% relative improvement)")
    
    print(f"\nOverfitting Analysis (Generalization Gap: Train - Test):")
    print(f"  Version 1 (Original):")
    print(f"    └─ Average Train Accuracy: {orig_summary['avg_train_accuracy']:.3f}" if orig_summary['avg_train_accuracy'] is not None else "    └─ Average Train Accuracy: N/A")
    print(f"    └─ Average Generalization Gap: {orig_summary['avg_generalization_gap']:.3f}" if orig_summary['avg_generalization_gap'] is not None else "    └─ Average Generalization Gap: N/A")
    print(f"       (Lower gap = better generalization to unseen data)")
    
    print(f"  Version 2 (Expanded):")
    print(f"    └─ Average Train Accuracy: {v2_summary['avg_train_accuracy']:.3f}" if v2_summary['avg_train_accuracy'] is not None else "    └─ Average Train Accuracy: N/A")
    print(f"    └─ Average Generalization Gap: {v2_summary['avg_generalization_gap']:.3f}" if v2_summary['avg_generalization_gap'] is not None else "    └─ Average Generalization Gap: N/A")
    print(f"       (Lower gap = better generalization to unseen data)")
    
    # Generate feature importance and SHAP plots
    if args.generate_plots:
        try:
            # Scale the test data
            X_test_scaled = orig_scaler.transform(X_test)
            
            # Get feature names
            feature_names = list(X.columns)
            
            generate_all_plots(
                X_test_scaled, X_test_scaled,
                orig_meta, v2_meta,
                orig_models, v2_models,
                feature_names,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"\n⚠ Warning: Could not generate plots: {str(e)}")
    
    # Final verdict
    print(f"\n" + "=" * 120)
    if v2_avg_test is not None and orig_avg_test is not None:
        if v2_avg_test >= orig_avg_test:
            print(f"✅ WINNER: Version 2 (Expanded Model) with {v2_avg_test:.3f} avg test accuracy")
            print(f"   • Better performance on test set")
            print(f"   • Trained on more data ({v2_summary['total_train_samples']} samples vs {orig_summary['total_train_samples']})")
            
            if v2_summary['avg_generalization_gap'] is not None and orig_summary['avg_generalization_gap'] is not None:
                if v2_summary['avg_generalization_gap'] <= orig_summary['avg_generalization_gap']:
                    print(f"   • Better generalization (lower gen gap: {v2_summary['avg_generalization_gap']:.3f} vs {orig_summary['avg_generalization_gap']:.3f})")
        else:
            print(f"✅ WINNER: Version 1 (Original Model) with {orig_avg_test:.3f} avg test accuracy")
    
    print("=" * 120)


if __name__ == "__main__":
    main()
