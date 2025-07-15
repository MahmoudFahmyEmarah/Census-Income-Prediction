"""
Phase 3: Model Development & Selection
Advanced model development with XGBoost, LightGBM, Neural Networks, and comprehensive evaluation.
Includes proper cross-validation, hyperparameter tuning, and overfitting detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import logging
import joblib
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import time
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelDeveloper:
    """Advanced model development with multiple algorithms and comprehensive evaluation."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.cv_results = {}
        self.best_models = {}
        self.feature_importance = {}
        self.training_curves = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with their configurations."""
        
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                'param_grid': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                },
                'search_type': 'grid'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced', None]
                },
                'search_type': 'random'
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'search_type': 'random'
            },
            
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    force_col_wise=True
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, -1],
                    'learning_rate': [0.1, 0.2],
                    'num_leaves': [31, 50],
                    'subsample': [0.8, 1.0]
                },
                'search_type': 'random'
            },
            
            'neural_network': {
                'model': MLPClassifier(
                    random_state=RANDOM_STATE,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.1
                ),
                'param_grid': {
                    'hidden_layer_sizes': [(100,), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'search_type': 'random'
            }
        }
        
        logger.info(f"Initialized {len(self.model_configs)} models for development")
        return self.model_configs
    
    def perform_cross_validation(self, X: pd.DataFrame, y: np.ndarray, 
                                cv_folds: int = 3) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive cross-validation for all models."""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        # Metrics to evaluate
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        self.cv_results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Cross-validating {model_name}...")
            
            model = config['model']
            results = {}
            
            # Evaluate each metric
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
            
            self.cv_results[model_name] = results
            
            # Log results
            logger.info(f"{model_name} CV results:")
            for metric, values in results.items():
                logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std']*2:.4f})")
        
        return self.cv_results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning for all models."""
        logger.info("Starting hyperparameter tuning...")
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        self.best_models = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            model = config['model']
            param_grid = config['param_grid']
            search_type = config['search_type']
            
            # Choose search strategy
            if search_type == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=cv, scoring='roc_auc',
                    n_jobs=-1, verbose=0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    model, param_grid, cv=cv, scoring='roc_auc',
                    n_iter=20, n_jobs=-1, verbose=0, random_state=RANDOM_STATE
                )
            
            # Fit the search
            start_time = time.time()
            search.fit(X, y)
            end_time = time.time()
            
            # Store results
            self.best_models[model_name] = {
                'model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'training_time': end_time - start_time
            }
            
            logger.info(f"{model_name} best score: {search.best_score_:.4f}")
            logger.info(f"{model_name} best params: {search.best_params_}")
            logger.info(f"{model_name} training time: {end_time - start_time:.2f}s")
        
        return self.best_models
    
    def detect_overfitting(self, X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Detect overfitting by comparing train vs validation performance."""
        logger.info("Detecting overfitting...")
        
        overfitting_analysis = {}
        
        for model_name, model_info in self.best_models.items():
            model = model_info['model']
            
            # Train predictions
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            
            # Validation predictions
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred),
                'recall': recall_score(y_train, y_train_pred),
                'f1': f1_score(y_train, y_train_pred),
                'roc_auc': roc_auc_score(y_train, y_train_proba)
            }
            
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1': f1_score(y_val, y_val_pred),
                'roc_auc': roc_auc_score(y_val, y_val_proba)
            }
            
            # Calculate overfitting indicators
            overfitting_scores = {}
            for metric in train_metrics:
                diff = train_metrics[metric] - val_metrics[metric]
                overfitting_scores[f'{metric}_diff'] = diff
                overfitting_scores[f'{metric}_ratio'] = val_metrics[metric] / train_metrics[metric] if train_metrics[metric] > 0 else 0
            
            # Overall overfitting score (average of differences)
            avg_diff = np.mean([overfitting_scores[f'{m}_diff'] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
            overfitting_scores['overall_overfitting'] = avg_diff
            
            overfitting_analysis[model_name] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'overfitting_scores': overfitting_scores,
                'is_overfitting': avg_diff > 0.05  # Threshold for overfitting
            }
            
            logger.info(f"{model_name} overfitting analysis:")
            logger.info(f"  Overall overfitting score: {avg_diff:.4f}")
            logger.info(f"  Is overfitting: {avg_diff > 0.05}")
        
        return overfitting_analysis
    
    def extract_feature_importance(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract feature importance from tree-based models."""
        logger.info("Extracting feature importance...")
        
        self.feature_importance = {}
        
        for model_name, model_info in self.best_models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[model_name] = importance_df
                
            elif hasattr(model, 'coef_'):
                # Linear models
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[model_name] = importance_df
            
            else:
                logger.warning(f"Cannot extract feature importance for {model_name}")
        
        return self.feature_importance
    
    def create_learning_curves(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        """Create learning curves to analyze model performance vs training size."""
        logger.info("Creating learning curves...")
        
        from sklearn.model_selection import learning_curve
        
        self.training_curves = {}
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for model_name, model_info in self.best_models.items():
            model = model_info['model']
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=3,
                scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
            )
            
            self.training_curves[model_name] = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        
        return self.training_curves
    
    def feature_selection(self, X: pd.DataFrame, y: np.ndarray, 
                         method: str = 'importance', top_k: int = 50) -> pd.DataFrame:
        """Perform feature selection based on importance or statistical tests."""
        logger.info(f"Performing feature selection using {method} method...")
        
        if method == 'importance':
            # Use feature importance from best performing model
            best_model_name = max(self.best_models.keys(), 
                                key=lambda x: self.best_models[x]['best_score'])
            
            if best_model_name in self.feature_importance:
                top_features = self.feature_importance[best_model_name].head(top_k)['feature'].tolist()
                selected_X = X[top_features]
                
                logger.info(f"Selected top {len(top_features)} features using {best_model_name} importance")
                return selected_X
        
        elif method == 'statistical':
            # Use statistical tests (chi-square for categorical, f-test for numerical)
            from sklearn.feature_selection import SelectKBest, f_classif
            
            selector = SelectKBest(score_func=f_classif, k=top_k)
            X_selected = selector.fit_transform(X, y)
            
            selected_features = X.columns[selector.get_support()].tolist()
            selected_X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            logger.info(f"Selected {len(selected_features)} features using statistical tests")
            return selected_X
        
        return X
    
    def generate_model_comparison_report(self) -> None:
        """Generate comprehensive model comparison report."""
        logger.info("Generating model comparison report...")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, model_info in self.best_models.items():
            row = {
                'Model': model_name,
                'Best_CV_Score': model_info['best_score'],
                'Training_Time': model_info['training_time']
            }
            
            # Add CV results if available
            if model_name in self.cv_results:
                cv_res = self.cv_results[model_name]
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    row[f'CV_{metric}_mean'] = cv_res[metric]['mean']
                    row[f'CV_{metric}_std'] = cv_res[metric]['std']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        report_content = f"""
# Model Development & Selection Report

## Model Comparison Summary

{comparison_df.to_string(index=False)}

## Best Models and Hyperparameters

"""
        
        for model_name, model_info in self.best_models.items():
            report_content += f"""
### {model_name.replace('_', ' ').title()}

**Best CV Score (ROC-AUC):** {model_info['best_score']:.4f}
**Training Time:** {model_info['training_time']:.2f} seconds

**Best Hyperparameters:**
"""
            for param, value in model_info['best_params'].items():
                report_content += f"- {param}: {value}\n"
        
        # Add feature importance section
        if self.feature_importance:
            report_content += "\n## Feature Importance Analysis\n\n"
            
            for model_name, importance_df in self.feature_importance.items():
                report_content += f"### Top 10 Features - {model_name.replace('_', ' ').title()}\n\n"
                report_content += importance_df.head(10).to_string(index=False)
                report_content += "\n\n"
        
        # Add recommendations
        report_content += """
## Recommendations

1. **Best Overall Model**: Based on cross-validation scores and training time
2. **Feature Selection**: Consider using top features for model simplification
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Hyperparameter Refinement**: Further tune top-performing models
5. **Overfitting Monitoring**: Regular validation on unseen data

## Next Steps

1. Evaluate models on test set
2. Perform bias analysis
3. Create model interpretability analysis
4. Prepare for production deployment
"""
        
        # Save report
        with open(REPORTS_DIR / "model_development_report.md", "w") as f:
            f.write(report_content)
        
        logger.info(f"Model development report saved to {REPORTS_DIR / 'model_development_report.md'}")
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations for model comparison."""
        logger.info("Creating model comparison visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        models = list(self.cv_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            
            means = [self.cv_results[model][metric]['mean'] for model in models]
            stds = [self.cv_results[model][metric]['std'] for model in models]
            
            bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance Comparison (for tree-based models)
        if self.feature_importance:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
            
            tree_models = [name for name in self.feature_importance.keys() 
                          if name in ['random_forest', 'xgboost', 'lightgbm']]
            
            for i, model_name in enumerate(tree_models[:4]):
                ax = axes[i//2, i%2]
                
                importance_df = self.feature_importance[model_name].head(15)
                
                bars = ax.barh(range(len(importance_df)), importance_df['importance'])
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df['feature'], fontsize=8)
                ax.set_xlabel('Importance')
                ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
                ax.invert_yaxis()
                
                # Add value labels
                for j, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                           f'{importance:.3f}', ha='left', va='center', fontsize=7)
            
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Training Time vs Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        training_times = [self.best_models[model]['training_time'] for model in models]
        best_scores = [self.best_models[model]['best_score'] for model in models]
        
        scatter = ax.scatter(training_times, best_scores, s=100, alpha=0.7)
        
        for i, model in enumerate(models):
            ax.annotate(model.replace('_', '\n'), 
                       (training_times[i], best_scores[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left')
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Best CV Score (ROC-AUC)')
        ax.set_title('Training Time vs Performance Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "training_time_vs_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison visualizations saved to reports directory")
    
    def save_models(self) -> None:
        """Save all trained models and results."""
        logger.info("Saving trained models and results...")
        
        # Save individual models
        for model_name, model_info in self.best_models.items():
            model_path = MODELS_DIR / f"{model_name}_best_model.pkl"
            joblib.dump(model_info['model'], model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save all results
        results_data = {
            'model_configs': self.model_configs,
            'cv_results': self.cv_results,
            'best_models': self.best_models,
            'feature_importance': self.feature_importance,
            'training_curves': self.training_curves
        }
        
        results_path = MODELS_DIR / "model_development_results.pkl"
        joblib.dump(results_data, results_path)
        logger.info(f"Saved all results to {results_path}")

def run_phase3():
    """Run complete Phase 3: Model Development & Selection."""
    logger.info("Starting Phase 3: Model Development & Selection")
    
    # Load processed data
    processed_data = joblib.load(MODELS_DIR / "processed_data.pkl")
    X_train = processed_data['X_train_balanced']
    y_train = processed_data['y_train_balanced']
    
    # Clean column names for XGBoost compatibility
    X_train.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                       for col in X_train.columns]
    
    # Split for overfitting detection
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Initialize model developer
    developer = ModelDeveloper()
    
    # Initialize models
    developer.initialize_models()
    
    # Perform cross-validation
    cv_results = developer.perform_cross_validation(X_train_split, y_train_split)
    
    # Hyperparameter tuning
    best_models = developer.hyperparameter_tuning(X_train_split, y_train_split)
    
    # Detect overfitting
    overfitting_analysis = developer.detect_overfitting(
        X_train_split, y_train_split, X_val_split, y_val_split
    )
    
    # Extract feature importance
    feature_importance = developer.extract_feature_importance(X_train_split)
    
    # Create learning curves
    training_curves = developer.create_learning_curves(X_train_split, y_train_split)
    
    # Feature selection
    X_selected = developer.feature_selection(X_train_split, y_train_split, method='importance', top_k=50)
    
    # Generate reports and visualizations
    developer.generate_model_comparison_report()
    developer.create_visualizations()
    
    # Save models
    developer.save_models()
    
    # Prepare results
    results = {
        'models_trained': len(best_models),
        'best_model': max(best_models.keys(), key=lambda x: best_models[x]['best_score']),
        'best_score': max(model_info['best_score'] for model_info in best_models.values()),
        'overfitting_detected': sum(1 for analysis in overfitting_analysis.values() if analysis['is_overfitting']),
        'feature_importance_extracted': len(feature_importance),
        'phase_status': 'completed'
    }
    
    logger.info("Phase 3 completed successfully!")
    return results

if __name__ == "__main__":
    # Run Phase 3
    results = run_phase3()
    print("Phase 3: Model Development & Selection completed successfully!")
    print(f"Models trained: {results['models_trained']}")
    print(f"Best model: {results['best_model']}")
    print(f"Best CV score: {results['best_score']:.4f}")
    print(f"Overfitting detected in: {results['overfitting_detected']} models")
    print(f"Feature importance extracted for: {results['feature_importance_extracted']} models")

