"""
Model Explainability Module
Provides feature importance analysis, SHAP values, and PDP plots
for understanding model predictions and behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

from sklearn.inspection import partial_dependence, PartialDependenceDisplay


class ModelExplainer:
    """
    Provides comprehensive model explainability tools:
    - SHAP values for feature importance
    - Partial Dependence Plots (PDP)
    - Individual Conditional Expectation (ICE) plots
    - Feature interaction analysis
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize explainer with trained models.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.explainers = {}  # SHAP explainers
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scaler."""
        try:
            # Load scaler
            scaler_path = self.model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load models
            for model_file in self.model_dir.glob('model_*.pkl'):
                category = model_file.stem.replace('model_', '')
                self.models[category] = joblib.load(model_file)
                print(f"✓ Loaded model: {category}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def compute_shap_values(self, 
                           X: pd.DataFrame, 
                           category: str,
                           background_samples: int = 100) -> Optional[Any]:
        """
        Compute SHAP values for model predictions.
        
        Args:
            X: Feature data
            category: Model category to explain
            background_samples: Number of samples for SHAP background
            
        Returns:
            SHAP values object or None if SHAP not available
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if category not in self.models:
            print(f"Model {category} not found")
            return None
        
        model = self.models[category]
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Create or retrieve SHAP explainer
        if category not in self.explainers:
            # Use TreeExplainer for tree-based models (Random Forest)
            try:
                self.explainers[category] = shap.TreeExplainer(model)
            except Exception:
                # Fallback to KernelExplainer if TreeExplainer fails
                background = shap.sample(X_scaled, min(background_samples, len(X_scaled)))
                self.explainers[category] = shap.KernelExplainer(
                    model.predict, 
                    background
                )
        
        # Compute SHAP values
        explainer = self.explainers[category]
        shap_values = explainer.shap_values(X_scaled)
        
        return shap_values
    
    def plot_shap_summary(self, 
                         X: pd.DataFrame, 
                         category: str,
                         plot_type: str = 'bar',
                         max_display: int = 10,
                         save_path: Optional[str] = None):
        """
        Create SHAP summary plot showing feature importance.
        
        Args:
            X: Feature data
            category: Model category to explain
            plot_type: 'bar', 'dot', or 'violin'
            max_display: Maximum number of features to display
            save_path: Path to save plot (optional)
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return
        
        shap_values = self.compute_shap_values(X, category)
        if shap_values is None:
            return
        
        # Scale features for plotting
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        plt.figure(figsize=(10, 6))
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # For multi-class, average across classes
            shap_values_avg = np.mean(np.abs(shap_values), axis=0)
            shap.summary_plot(
                shap_values_avg, 
                X_scaled, 
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_scaled, 
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
        
        plt.title(f'Feature Importance - {category}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved SHAP plot to {save_path}")
        
        plt.show()
    
    def plot_shap_waterfall(self,
                           X: pd.DataFrame,
                           category: str,
                           instance_idx: int = 0,
                           save_path: Optional[str] = None):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X: Feature data
            category: Model category
            instance_idx: Index of instance to explain
            save_path: Path to save plot (optional)
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return
        
        shap_values = self.compute_shap_values(X, category)
        if shap_values is None:
            return
        
        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Create waterfall plot
        if isinstance(shap_values, list):
            # Multi-class: use first class
            shap_values = shap_values[0]
        
        explainer = self.explainers[category]
        
        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0],
            data=X_scaled.iloc[instance_idx].values,
            feature_names=X_scaled.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall - {category} (Instance {instance_idx})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved waterfall plot to {save_path}")
        
        plt.show()
    
    def plot_shap_dependence(self,
                            X: pd.DataFrame,
                            category: str,
                            feature: str,
                            interaction_feature: Optional[str] = 'auto',
                            save_path: Optional[str] = None):
        """
        Create SHAP dependence plot showing how feature affects predictions.
        
        Args:
            X: Feature data
            category: Model category
            feature: Feature to plot
            interaction_feature: Feature for interaction coloring
            save_path: Path to save plot (optional)
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return
        
        shap_values = self.compute_shap_values(X, category)
        if shap_values is None:
            return
        
        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap.dependence_plot(
            feature,
            shap_values,
            X_scaled,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence - {feature} ({category})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dependence plot to {save_path}")
        
        plt.show()
    
    def plot_partial_dependence(self,
                               X: pd.DataFrame,
                               category: str,
                               features: List[str],
                               save_path: Optional[str] = None):
        """
        Create Partial Dependence Plots (PDP).
        
        Args:
            X: Feature data
            category: Model category
            features: List of features to plot
            save_path: Path to save plot (optional)
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return
        
        model = self.models[category]
        
        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Get feature indices
        feature_indices = [X_scaled.columns.get_loc(f) for f in features if f in X_scaled.columns]
        
        if not feature_indices:
            print("No valid features found")
            return
        
        # Create PDP plot
        fig, ax = plt.subplots(figsize=(12, 4 * ((len(feature_indices) + 2) // 3)))
        
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_scaled,
            feature_indices,
            feature_names=X_scaled.columns.tolist(),
            ax=ax,
            n_cols=3,
            grid_resolution=50
        )
        
        fig.suptitle(f'Partial Dependence Plots - {category}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved PDP plot to {save_path}")
        
        plt.show()
    
    def plot_ice_curves(self,
                       X: pd.DataFrame,
                       category: str,
                       feature: str,
                       n_samples: int = 50,
                       save_path: Optional[str] = None):
        """
        Create Individual Conditional Expectation (ICE) plots.
        
        Args:
            X: Feature data
            category: Model category
            feature: Feature to plot
            n_samples: Number of individual curves to plot
            save_path: Path to save plot (optional)
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return
        
        model = self.models[category]
        
        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        if feature not in X_scaled.columns:
            print(f"Feature {feature} not found")
            return
        
        # Sample instances
        n_samples = min(n_samples, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
        X_sample = X_scaled.iloc[sample_indices]
        
        # Get feature index
        feature_idx = X_scaled.columns.get_loc(feature)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot ICE curves
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_sample,
            [feature_idx],
            kind='both',  # Shows both ICE and PDP
            feature_names=X_scaled.columns.tolist(),
            ax=axes,
            n_cols=2,
            grid_resolution=50
        )
        
        fig.suptitle(f'ICE Plots - {feature} ({category})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved ICE plot to {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, category: str, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get feature importance from Random Forest model.
        
        Args:
            category: Model category
            X: Optional feature data for permutation importance fallback
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return pd.DataFrame()
        
        model = self.models[category]
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.scaler.feature_names_in_ if self.scaler else [f"Feature_{i}" for i in range(len(importances))]
            if np.sum(importances) > 0:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                return importance_df
            
            # Fallback to permutation importance if provided data
            if X is not None and not X.empty:
                try:
                    from sklearn.inspection import permutation_importance
                    if self.scaler is not None:
                        X_scaled = pd.DataFrame(
                            self.scaler.transform(X),
                            columns=X.columns,
                            index=X.index
                        )
                    else:
                        X_scaled = X
                    perm = permutation_importance(model, X_scaled, model.predict(X_scaled), n_repeats=10, random_state=42)
                    perm_importances = np.maximum(perm.importances_mean, 0)
                    importance_df = pd.DataFrame({
                        'Feature': X_scaled.columns.tolist(),
                        'Importance': perm_importances
                    }).sort_values('Importance', ascending=False)
                    return importance_df
                except Exception:
                    pass
            
            return pd.DataFrame()
        else:
            print("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
    
    def plot_feature_importance(self,
                               category: str,
                               top_n: int = 10,
                               save_path: Optional[str] = None,
                               X: Optional[pd.DataFrame] = None):
        """
        Plot feature importance from Random Forest.
        
        Args:
            category: Model category
            top_n: Number of top features to display
            save_path: Path to save plot (optional)
            X: Optional feature data for permutation importance fallback
        """
        importance_df = self.get_feature_importance(category, X=X)
        
        if importance_df.empty:
            print("⚠ Feature importance not available (model may be single-class or degenerate)")
            return
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {category}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved importance plot to {save_path}")
        
        plt.show()
        
        return importance_df
    
    def generate_explanation_report(self,
                                   X: pd.DataFrame,
                                   category: str,
                                   instance_idx: int = 0) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report for a single prediction.
        
        Args:
            X: Feature data
            category: Model category
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary containing various explanations
        """
        report = {
            'category': category,
            'instance_idx': instance_idx,
            'feature_values': {},
            'prediction': None,
            'feature_importance': None,
            'shap_values': None,
            'top_contributors': []
        }
        
        if category not in self.models:
            return report
        
        model = self.models[category]
        
        # Get instance data
        instance = X.iloc[instance_idx:instance_idx+1]
        report['feature_values'] = instance.to_dict('records')[0]
        
        # Scale features
        if self.scaler is not None:
            instance_scaled = self.scaler.transform(instance)
        else:
            instance_scaled = instance.values
        
        # Get prediction
        try:
            prediction = model.predict(instance_scaled)[0]
            report['prediction'] = prediction
        except Exception as e:
            print(f"Error getting prediction: {e}")
        
        # Get feature importance
        importance_df = self.get_feature_importance(category)
        if not importance_df.empty:
            report['feature_importance'] = importance_df.to_dict('records')
        
        # Get SHAP values
        if SHAP_AVAILABLE:
            try:
                shap_values = self.compute_shap_values(instance, category)
                if shap_values is not None:
                    # Normalize SHAP output shape to (n_features,)
                    if isinstance(shap_values, list):
                        # Multi-class: pick class with highest probability
                        class_idx = 0
                        try:
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(instance_scaled)
                                if proba.ndim == 2:
                                    class_idx = int(np.argmax(proba[0]))
                        except Exception:
                            class_idx = 0
                        shap_values = shap_values[class_idx]

                    if isinstance(shap_values, np.ndarray):
                        if shap_values.ndim == 3:
                            # (n_samples, n_classes, n_features)
                            class_idx = 0
                            try:
                                if hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(instance_scaled)
                                    if proba.ndim == 2:
                                        class_idx = int(np.argmax(proba[0]))
                            except Exception:
                                class_idx = 0
                            shap_values = shap_values[0, class_idx]
                        elif shap_values.ndim == 2:
                            # (n_samples, n_features)
                            shap_values = shap_values[0]

                    # Get top contributors
                    feature_names = instance.columns.tolist()
                    shap_row = shap_values
                    shap_dict = {feature_names[i]: float(shap_row[i])
                                for i in range(len(feature_names))}
                    
                    sorted_shap = sorted(shap_dict.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
                    
                    report['shap_values'] = shap_dict
                    report['top_contributors'] = sorted_shap[:5]
            except Exception as e:
                print(f"Error computing SHAP values: {e}")
        
        return report


if __name__ == "__main__":
    print("Model Explainability Module")
    print("=" * 50)
    print("Available features:")
    print("- SHAP value computation")
    print("- Feature importance plots")
    print("- Partial Dependence Plots (PDP)")
    print("- Individual Conditional Expectation (ICE) plots")
    print("- Feature interaction analysis")
