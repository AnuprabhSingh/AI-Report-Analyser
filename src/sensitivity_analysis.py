"""
Sensitivity Analysis Module
Tests model robustness by varying input parameters and quantifying uncertainty.
Helps understand how measurement errors and variations affect predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Callable
import joblib
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    Performs comprehensive sensitivity analysis on clinical prediction models:
    - Parameter perturbation analysis
    - Uncertainty quantification
    - Robustness testing
    - Monte Carlo simulations
    - Measurement error propagation
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize sensitivity analyzer.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        
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

    def _predict_score(self, model, X_scaled: np.ndarray) -> float:
        """Return a numeric prediction score using probabilities when available."""
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                if proba.ndim == 2:
                    return float(np.max(proba, axis=1)[0])
            return float(model.predict(X_scaled)[0])
        except Exception:
            return np.nan
    
    def one_at_a_time_sensitivity(self,
                                  base_instance: pd.DataFrame,
                                  category: str,
                                  features: Optional[List[str]] = None,
                                  variation_range: Tuple[float, float] = (-0.2, 0.2),
                                  n_steps: int = 20) -> Dict[str, pd.DataFrame]:
        """
        One-at-a-time (OAT) sensitivity analysis.
        Varies each feature independently while keeping others constant.
        
        Args:
            base_instance: Base case (single row DataFrame)
            category: Model category to analyze
            features: List of features to analyze (None = all)
            variation_range: (min_percent, max_percent) relative variation
            n_steps: Number of variation steps
            
        Returns:
            Dictionary mapping feature names to sensitivity DataFrames
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return {}
        
        model = self.models[category]
        
        if features is None:
            features = base_instance.columns.tolist()
        
        results = {}
        
        for feature in features:
            if feature not in base_instance.columns:
                continue
            
            base_value = base_instance[feature].values[0]
            
            # Create variation range
            min_val = base_value * (1 + variation_range[0])
            max_val = base_value * (1 + variation_range[1])
            values = np.linspace(min_val, max_val, n_steps)
            
            predictions = []
            
            for val in values:
                # Create modified instance
                instance = base_instance.copy()
                instance[feature] = val
                
                # Scale and predict
                if self.scaler is not None:
                    instance_scaled = self.scaler.transform(instance)
                else:
                    instance_scaled = instance.values
                
                pred = self._predict_score(model, instance_scaled)
                predictions.append(pred)
            
            # Store results
            results[feature] = pd.DataFrame({
                f'{feature}_value': values,
                'prediction': predictions,
                'variation_percent': (values - base_value) / base_value * 100
            })
        
        return results
    
    def plot_oat_sensitivity(self,
                            sensitivity_results: Dict[str, pd.DataFrame],
                            category: str,
                            save_path: Optional[str] = None):
        """
        Plot one-at-a-time sensitivity results.
        
        Args:
            sensitivity_results: Output from one_at_a_time_sensitivity
            category: Model category name
            save_path: Path to save plot (optional)
        """
        n_features = len(sensitivity_results)
        if n_features == 0:
            return
        
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, (feature, df) in enumerate(sensitivity_results.items()):
            ax = axes[idx]
            
            # Plot sensitivity curve
            ax.plot(df['variation_percent'], df['prediction'], 
                   marker='o', linewidth=2, markersize=4)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Base value')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f'{feature} Variation (%)', fontsize=10)
            ax.set_ylabel('Prediction', fontsize=10)
            ax.set_title(f'Sensitivity to {feature}', fontsize=11, fontweight='bold')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'One-at-a-Time Sensitivity Analysis - {category}', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved sensitivity plot to {save_path}")
        
        plt.show()
    
    def monte_carlo_simulation(self,
                              base_instance: pd.DataFrame,
                              category: str,
                              features: Optional[List[str]] = None,
                              error_std: float = 0.05,
                              n_simulations: int = 1000,
                              distribution: str = 'normal') -> Dict[str, Any]:
        """
        Monte Carlo simulation for uncertainty quantification.
        Simulates measurement errors and propagates uncertainty.
        
        Args:
            base_instance: Base case (single row DataFrame)
            category: Model category
            features: Features to perturb (None = all)
            error_std: Standard deviation of measurement error (as fraction)
            n_simulations: Number of simulations
            distribution: 'normal', 'uniform', or 'lognormal'
            
        Returns:
            Dictionary with simulation results and statistics
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return {}
        
        model = self.models[category]
        
        if features is None:
            features = base_instance.columns.tolist()
        
        # Store all predictions
        predictions = []
        perturbed_features = []
        
        for i in range(n_simulations):
            # Create perturbed instance
            instance = base_instance.copy()
            perturbation = {}
            
            for feature in features:
                if feature not in instance.columns:
                    continue
                
                base_value = base_instance[feature].values[0]
                
                # Generate random perturbation
                if distribution == 'normal':
                    noise = np.random.normal(0, error_std)
                elif distribution == 'uniform':
                    noise = np.random.uniform(-error_std, error_std)
                elif distribution == 'lognormal':
                    noise = np.random.lognormal(0, error_std) - 1
                else:
                    noise = 0
                
                perturbed_value = base_value * (1 + noise)
                instance[feature] = perturbed_value
                perturbation[feature] = perturbed_value
            
            # Scale and predict
            if self.scaler is not None:
                instance_scaled = self.scaler.transform(instance)
            else:
                instance_scaled = instance.values
            
            pred = self._predict_score(model, instance_scaled)
            if not np.isnan(pred):
                predictions.append(pred)
                perturbed_features.append(perturbation)
        
        predictions = np.array(predictions)
        
        # Compute statistics
        results = {
            'predictions': predictions,
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'confidence_interval_95': (np.percentile(predictions, 2.5), 
                                      np.percentile(predictions, 97.5)),
            'coefficient_of_variation': np.std(predictions) / np.mean(predictions) if np.mean(predictions) != 0 else np.inf,
            'n_simulations': len(predictions),
            'perturbed_features': perturbed_features
        }
        
        return results
    
    def plot_monte_carlo_results(self,
                                mc_results: Dict[str, Any],
                                category: str,
                                save_path: Optional[str] = None):
        """
        Plot Monte Carlo simulation results.
        
        Args:
            mc_results: Output from monte_carlo_simulation
            category: Model category name
            save_path: Path to save plot (optional)
        """
        predictions = mc_results['predictions']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        ax1 = axes[0]
        ax1.hist(predictions, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        # Fit and plot normal distribution
        mu, sigma = mc_results['mean'], mc_results['std']
        x = np.linspace(predictions.min(), predictions.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
        
        ax1.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}')
        ax1.axvline(mc_results['confidence_interval_95'][0], color='orange', linestyle=':', linewidth=2, label='95% CI')
        ax1.axvline(mc_results['confidence_interval_95'][1], color='orange', linestyle=':', linewidth=2)
        
        ax1.set_xlabel('Prediction', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Prediction Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        box_data = [predictions]
        bp = ax2.boxplot(box_data, labels=['Predictions'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        
        # Add statistics text
        stats_text = f"Mean: {mc_results['mean']:.2f}\n"
        stats_text += f"Median: {mc_results['median']:.2f}\n"
        stats_text += f"Std: {mc_results['std']:.2f}\n"
        stats_text += f"CV: {mc_results['coefficient_of_variation']:.2%}\n"
        stats_text += f"95% CI: [{mc_results['confidence_interval_95'][0]:.2f}, {mc_results['confidence_interval_95'][1]:.2f}]"
        
        ax2.text(1.15, np.median(predictions), stats_text, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_ylabel('Prediction', fontsize=12)
        ax2.set_title('Prediction Variability', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Monte Carlo Uncertainty Quantification - {category}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Monte Carlo plot to {save_path}")
        
        plt.show()
    
    def global_sensitivity_analysis(self,
                                   X_sample: pd.DataFrame,
                                   category: str,
                                   n_samples: int = 500,
                                   method: str = 'sobol') -> pd.DataFrame:
        """
        Global sensitivity analysis using Sobol indices or correlation-based methods.
        
        Args:
            X_sample: Sample data for analysis
            category: Model category
            n_samples: Number of samples for analysis
            method: 'sobol' or 'correlation'
            
        Returns:
            DataFrame with sensitivity indices
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return pd.DataFrame()
        
        model = self.models[category]
        
        # Sample data
        if len(X_sample) > n_samples:
            X_sample = X_sample.sample(n_samples)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_sample),
                columns=X_sample.columns,
                index=X_sample.index
            )
        else:
            X_scaled = X_sample
        
        # Get predictions
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled.values)
                if proba.ndim == 2:
                    predictions = np.max(proba, axis=1)
                else:
                    predictions = model.predict(X_scaled.values)
            else:
                predictions = model.predict(X_scaled.values)
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return pd.DataFrame()

        if np.nanstd(predictions) == 0:
            print("Predictions are constant; global sensitivity not informative.")
            return pd.DataFrame()
        
        # Compute sensitivity indices
        if method == 'correlation':
            # Correlation-based sensitivity
            # Drop zero-variance features for correlation computation
            variances = X_scaled.var(axis=0)
            X_corr = X_scaled.loc[:, variances > 0]
            if X_corr.empty:
                print("All features have zero variance; cannot compute global sensitivity.")
                return pd.DataFrame()

            sensitivity_indices = []
            
            for col in X_corr.columns:
                # Pearson correlation
                try:
                    corr_pearson, _ = stats.pearsonr(X_corr[col], predictions)
                except Exception:
                    corr_pearson = 0
                
                # Spearman correlation (rank-based)
                try:
                    corr_spearman, _ = stats.spearmanr(X_corr[col], predictions)
                except Exception:
                    corr_spearman = 0
                
                # Partial correlation (normalized importance)
                sensitivity_indices.append({
                    'Feature': col,
                    'Pearson_Correlation': abs(corr_pearson),
                    'Spearman_Correlation': abs(corr_spearman),
                    'Mean_Sensitivity': (abs(corr_pearson) + abs(corr_spearman)) / 2
                })
            
            sensitivity_df = pd.DataFrame(sensitivity_indices)
            sensitivity_df = sensitivity_df.dropna()
            if sensitivity_df.empty:
                print("Global sensitivity results are NaN; skipping.")
                return pd.DataFrame()
            sensitivity_df = sensitivity_df.sort_values('Mean_Sensitivity', ascending=False)
            
        else:
            print(f"Method {method} not fully implemented. Using correlation method.")
            return self.global_sensitivity_analysis(X_sample, category, n_samples, 'correlation')
        
        return sensitivity_df
    
    def plot_global_sensitivity(self,
                               sensitivity_df: pd.DataFrame,
                               save_path: Optional[str] = None):
        """
        Plot global sensitivity analysis results.
        
        Args:
            sensitivity_df: Output from global_sensitivity_analysis
            save_path: Path to save plot (optional)
        """
        if sensitivity_df.empty:
            print("⚠ Global sensitivity results are empty; skipping plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of mean sensitivity
        ax1 = axes[0]
        top_features = sensitivity_df.head(10)
        sns.barplot(data=top_features, x='Mean_Sensitivity', y='Feature', 
                   palette='viridis', ax=ax1)
        ax1.set_xlabel('Mean Sensitivity Index', fontsize=12)
        ax1.set_ylabel('Feature', fontsize=12)
        ax1.set_title('Top 10 Most Sensitive Features', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Comparison of Pearson vs Spearman
        ax2 = axes[1]
        ax2.scatter(sensitivity_df['Pearson_Correlation'], 
                   sensitivity_df['Spearman_Correlation'],
                   alpha=0.6, s=100)
        
        # Add diagonal line
        max_val = max(sensitivity_df['Pearson_Correlation'].max(), 
                     sensitivity_df['Spearman_Correlation'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        # Annotate points
        for idx, row in sensitivity_df.head(5).iterrows():
            ax2.annotate(row['Feature'], 
                        (row['Pearson_Correlation'], row['Spearman_Correlation']),
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Pearson Correlation', fontsize=12)
        ax2.set_ylabel('Spearman Correlation', fontsize=12)
        ax2.set_title('Linear vs. Rank-Based Sensitivity', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved global sensitivity plot to {save_path}")
        
        plt.show()
    
    def feature_interaction_sensitivity(self,
                                       base_instance: pd.DataFrame,
                                       category: str,
                                       feature1: str,
                                       feature2: str,
                                       n_points: int = 20,
                                       variation_range: Tuple[float, float] = (-0.2, 0.2)) -> pd.DataFrame:
        """
        Analyze sensitivity to interactions between two features.
        
        Args:
            base_instance: Base case (single row DataFrame)
            category: Model category
            feature1: First feature name
            feature2: Second feature name
            n_points: Number of grid points per dimension
            variation_range: (min_percent, max_percent) relative variation
            
        Returns:
            DataFrame with interaction results
        """
        if category not in self.models:
            print(f"Model {category} not found")
            return pd.DataFrame()
        
        model = self.models[category]
        
        # Create grid of feature values
        base_val1 = base_instance[feature1].values[0]
        base_val2 = base_instance[feature2].values[0]
        
        min_val1 = base_val1 * (1 + variation_range[0])
        max_val1 = base_val1 * (1 + variation_range[1])
        min_val2 = base_val2 * (1 + variation_range[0])
        max_val2 = base_val2 * (1 + variation_range[1])
        
        values1 = np.linspace(min_val1, max_val1, n_points)
        values2 = np.linspace(min_val2, max_val2, n_points)
        
        # Create meshgrid
        V1, V2 = np.meshgrid(values1, values2)
        predictions = np.zeros_like(V1)
        
        for i in range(n_points):
            for j in range(n_points):
                instance = base_instance.copy()
                instance[feature1] = V1[i, j]
                instance[feature2] = V2[i, j]
                
                # Scale and predict
                if self.scaler is not None:
                    instance_scaled = self.scaler.transform(instance)
                else:
                    instance_scaled = instance.values
                
                pred = self._predict_score(model, instance_scaled)
                predictions[i, j] = pred
        
        # Create result DataFrame
        results = pd.DataFrame({
            feature1: V1.flatten(),
            feature2: V2.flatten(),
            'prediction': predictions.flatten()
        })
        
        return results
    
    def plot_interaction_sensitivity(self,
                                    interaction_results: pd.DataFrame,
                                    feature1: str,
                                    feature2: str,
                                    category: str,
                                    save_path: Optional[str] = None):
        """
        Plot feature interaction sensitivity as heatmap.
        
        Args:
            interaction_results: Output from feature_interaction_sensitivity
            feature1: First feature name
            feature2: Second feature name
            category: Model category name
            save_path: Path to save plot (optional)
        """
        # Pivot data for heatmap
        pivot_data = interaction_results.pivot(
            index=feature2, 
            columns=feature1, 
            values='prediction'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, cmap='viridis', cbar_kws={'label': 'Prediction'})
        plt.xlabel(feature1, fontsize=12)
        plt.ylabel(feature2, fontsize=12)
        plt.title(f'Feature Interaction Sensitivity: {feature1} × {feature2} ({category})',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved interaction plot to {save_path}")
        
        plt.show()
    
    def generate_sensitivity_report(self,
                                   base_instance: pd.DataFrame,
                                   category: str,
                                   features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive sensitivity analysis report.
        
        Args:
            base_instance: Base case for analysis
            category: Model category
            features: Features to analyze (None = all)
            
        Returns:
            Dictionary containing all sensitivity analyses
        """
        report = {
            'category': category,
            'base_values': base_instance.to_dict('records')[0],
            'oat_sensitivity': None,
            'monte_carlo': None,
            'robustness_score': None
        }
        
        # One-at-a-time sensitivity
        print("Running OAT sensitivity analysis...")
        oat_results = self.one_at_a_time_sensitivity(base_instance, category, features)
        report['oat_sensitivity'] = oat_results
        
        # Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(base_instance, category, features)
        report['monte_carlo'] = mc_results
        
        # Compute robustness score (lower is more robust)
        if mc_results:
            cv = mc_results.get('coefficient_of_variation', np.inf)
            report['robustness_score'] = 1 / (1 + cv) if cv != np.inf else 0
            report['robustness_interpretation'] = self._interpret_robustness(report['robustness_score'])
        
        return report
    
    def _interpret_robustness(self, score: float) -> str:
        """Interpret robustness score."""
        if score > 0.9:
            return "Excellent - Model predictions are highly robust to measurement errors"
        elif score > 0.7:
            return "Good - Model shows good robustness with moderate sensitivity"
        elif score > 0.5:
            return "Fair - Model has noticeable sensitivity to input variations"
        else:
            return "Poor - Model predictions are highly sensitive to measurement errors"


if __name__ == "__main__":
    print("Sensitivity Analysis Module")
    print("=" * 50)
    print("Available analyses:")
    print("- One-at-a-time (OAT) sensitivity")
    print("- Monte Carlo uncertainty quantification")
    print("- Global sensitivity analysis")
    print("- Feature interaction analysis")
    print("- Robustness scoring")
