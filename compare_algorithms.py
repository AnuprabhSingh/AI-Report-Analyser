#!/usr/bin/env python3
"""
Compare Multiple ML Algorithms for Medical Interpretation

Tests the following algorithms:
1. Random Forest (current)
2. Gradient Boosting
3. XGBoost
4. Support Vector Machine (SVM)
5. Logistic Regression
6. K-Nearest Neighbors (KNN)
7. Decision Tree
8. Neural Network (MLP)

Compares accuracy, precision, recall, F1-score, and training time.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# Import all algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Neural Network removed - requires larger dataset
# from sklearn.neural_network import MLPClassifier

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    # Don't print error here, will show in main if needed


class AlgorithmComparator:
    """Compare multiple ML algorithms for interpretation prediction."""
    
    def __init__(self):
        self.key_parameters = [
            'EF', 'FS', 'LVID_D', 'LVID_S', 'IVS_D', 'LVPW_D',
            'LA_DIMENSION', 'AORTIC_ROOT', 'MV_E_A', 'LV_MASS'
        ]
        
        self.feature_names = ['age', 'sex'] + self.key_parameters
        
        # Define algorithms to test
        self.algorithms = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            # Neural Network removed - requires more data (>1000 samples)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.algorithms['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
    
    def load_dataset(self, dataset_path: str):
        """Load and prepare dataset."""
        print(f"Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} samples\n")
        
        # Convert to dataframe
        samples = []
        for item in data:
            measurements = item['measurements']
            patient = item.get('patient', {})
            interpretations = item['interpretations']
            
            sample = {
                'age': patient.get('age', 0),
                'sex': 1 if patient.get('sex') == 'M' else 0,
            }
            
            for param in self.key_parameters:
                sample[param] = measurements.get(param, 0)
            
            sample['labels'] = self._extract_labels(interpretations)
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def _extract_labels(self, interpretations: dict) -> dict:
        """Extract classification labels from interpretation text."""
        labels = {}
        
        # LV_HYPERTROPHY
        ivs_text = interpretations.get('Interventricular Septum', '')
        if 'Normal' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'None'
        elif 'Mild' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'Mild'
        elif 'Moderate' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'Moderate'
        elif 'Severe' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'Severe'
        else:
            labels['LV_HYPERTROPHY'] = 'Unknown'
        
        # LA_SIZE
        la_text = interpretations.get('Left Atrium', '')
        if 'Normal' in la_text:
            labels['LA_SIZE'] = 'Normal'
        elif 'enlarge' in la_text.lower():
            labels['LA_SIZE'] = 'Enlarged'
        else:
            labels['LA_SIZE'] = 'Unknown'
        
        # DIASTOLIC_FUNCTION
        diastolic_text = interpretations.get('Diastolic Function', '')
        if 'Normal' in diastolic_text:
            labels['DIASTOLIC_FUNCTION'] = 'Normal'
        else:
            labels['DIASTOLIC_FUNCTION'] = 'Abnormal'
        
        return labels
    
    def prepare_data(self, df: pd.DataFrame, category: str, test_size: float = 0.2):
        """Prepare features and labels for a specific category."""
        
        # Features
        X = df[self.feature_names].copy()
        
        # Handle missing values
        for col in self.feature_names:
            if col not in ['age', 'sex']:
                median_val = X[X[col] > 0][col].median()
                X[col] = X[col].replace(0, median_val)
        
        X = X.fillna(0)
        
        # Labels
        y = df['labels'].apply(lambda x: x.get(category, 'Unknown'))
        
        # Remove unknown labels
        mask = y != 'Unknown'
        X = X[mask]
        y = y[mask]
        
        if len(y) < 10:
            return None, None, None, None, None
        
        # Check if stratification is possible (need at least 2 samples per class)
        class_counts = y.value_counts()
        can_stratify = all(count >= 2 for count in class_counts)
        
        # Split
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Don't stratify if classes have too few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def compare_algorithms(self, dataset_path: str, categories: list = None):
        """Compare all algorithms on specified categories."""
        
        if categories is None:
            categories = ['LV_HYPERTROPHY', 'LA_SIZE', 'DIASTOLIC_FUNCTION']
        
        print("=" * 80)
        print("ML ALGORITHM COMPARISON")
        print("=" * 80)
        print()
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        # Store results
        all_results = {}
        
        for category in categories:
            print("\n" + "=" * 80)
            print(f"CATEGORY: {category}")
            print("=" * 80)
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(df, category)
            
            if X_train is None:
                print(f"⚠️  Insufficient data for {category}, skipping...\n")
                continue
            
            print(f"Training samples: {len(y_train)}")
            print(f"Testing samples: {len(y_test)}")
            print(f"Classes: {sorted(y_train.unique())}")
            print()
            
            results = {}
            
            # Test each algorithm
            for algo_name, model in self.algorithms.items():
                try:
                    # Train
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    # Predict
                    start_time = time.time()
                    y_pred = model.predict(X_test)
                    predict_time = time.time() - start_time
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average='weighted', zero_division=0
                    )
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    results[algo_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'train_time': train_time,
                        'predict_time': predict_time,
                        'predictions': y_pred,
                        'actual': y_test
                    }
                    
                    print(f"✓ {algo_name:25s}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Time={train_time:.3f}s")
                    
                except Exception as e:
                    print(f"✗ {algo_name:25s}: ERROR - {str(e)}")
                    results[algo_name] = None
            
            all_results[category] = results
        
        return all_results
    
    def display_comparison_table(self, results: dict):
        """Display comparison table for all algorithms."""
        
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON TABLE")
        print("=" * 80)
        
        for category, algo_results in results.items():
            if not algo_results:
                continue
            
            print(f"\n{'=' * 80}")
            print(f"CATEGORY: {category}")
            print('=' * 80)
            
            # Header
            print(f"\n{'Algorithm':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} "
                  f"{'F1-Score':>10s} {'CV-Mean':>10s} {'Train(s)':>10s}")
            print('-' * 95)
            
            # Sort by accuracy
            sorted_results = sorted(
                [(name, res) for name, res in algo_results.items() if res is not None],
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            # Display results
            for algo_name, res in sorted_results:
                print(f"{algo_name:<25s} "
                      f"{res['accuracy']:>10.3f} "
                      f"{res['precision']:>10.3f} "
                      f"{res['recall']:>10.3f} "
                      f"{res['f1_score']:>10.3f} "
                      f"{res['cv_mean']:>10.3f} "
                      f"{res['train_time']:>10.3f}")
            
            # Best algorithm
            best_algo = sorted_results[0]
            print(f"\n🏆 BEST: {best_algo[0]} with {best_algo[1]['accuracy']:.3f} accuracy")
    
    def display_summary(self, results: dict):
        """Display overall summary."""
        
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        # Calculate average accuracy for each algorithm
        algo_avg_scores = {}
        
        for category, algo_results in results.items():
            for algo_name, res in algo_results.items():
                if res is not None:
                    if algo_name not in algo_avg_scores:
                        algo_avg_scores[algo_name] = []
                    algo_avg_scores[algo_name].append(res['accuracy'])
        
        # Calculate averages
        algo_averages = {
            name: np.mean(scores) 
            for name, scores in algo_avg_scores.items()
        }
        
        # Sort by average accuracy
        sorted_algos = sorted(algo_averages.items(), key=lambda x: x[1], reverse=True)
        
        print("\nAverage Accuracy Across All Categories:")
        print('-' * 50)
        
        for rank, (algo_name, avg_acc) in enumerate(sorted_algos, 1):
            badge = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{badge} {rank}. {algo_name:<25s}: {avg_acc:.3f} ({avg_acc*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        best_algo = sorted_algos[0][0]
        best_acc = sorted_algos[0][1]
        
        print(f"""
🏆 BEST OVERALL: {best_algo}
   Average Accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)

TRADE-OFFS:
- Random Forest: Good accuracy, interpretable, robust
- Gradient Boosting: High accuracy, slower training
- XGBoost: Excellent accuracy, requires tuning
- SVM: Good for small datasets, slow on large data
- Logistic Regression: Fast, interpretable, works for linear patterns
- KNN: Simple, slow prediction, sensitive to scale
- Decision Tree: Fast, interpretable, prone to overfitting

Note: Neural Networks removed (need >1000 samples for deep learning)

RECOMMENDATION FOR YOUR PROJECT:
- Use {best_algo} for best accuracy
- Consider Random Forest for interpretability + performance balance
- Use Gradient Boosting if accuracy is critical
""")


def main():
    """Main comparison pipeline."""
    
    print("\n" + "=" * 80)
    print("ML ALGORITHM COMPARISON TOOL")
    print("=" * 80)
    
    if not XGBOOST_AVAILABLE:
        print("\n⚠️  Note: XGBoost not available (requires OpenMP library)")
        print("    To install: brew install libomp (on macOS)")
        print("    Testing with 6 algorithms instead of 7...\n")
    else:
        print("\nTesting 7 algorithms on your medical interpretation dataset...\n")
    
    print()
    
    # Initialize comparator
    comparator = AlgorithmComparator()
    
    # Dataset path
    dataset_path = 'data/processed/training_dataset.json'
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Categories to test
    categories = ['LV_HYPERTROPHY', 'LA_SIZE', 'DIASTOLIC_FUNCTION']
    
    # Compare algorithms
    results = comparator.compare_algorithms(dataset_path, categories)
    
    # Display results
    comparator.display_comparison_table(results)
    comparator.display_summary(results)
    
    print("\n" + "=" * 80)
    print("✓ COMPARISON COMPLETE")
    print("=" * 80)
    print("\nYou can now choose the best algorithm for your models!")
    print("To retrain with a different algorithm, edit train_interpretation_model.py")


if __name__ == '__main__':
    main()
