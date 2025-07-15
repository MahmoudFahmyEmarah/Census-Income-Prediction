"""
Phase 2: Data Preprocessing & Feature Engineering
Production-ready preprocessing pipeline with strategic missing value handling,
feature engineering, and duplicate detection.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
from pathlib import Path
import logging
import joblib
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DuplicateHandler(BaseEstimator, TransformerMixin):
    """Handle duplicate records with configurable strategies."""
    
    def __init__(self, strategy='remove', subset=None):
        """
        Initialize duplicate handler.
        
        Args:
            strategy: 'remove', 'keep_first', 'keep_last', or 'mark'
            subset: Columns to consider for duplicate detection
        """
        self.strategy = strategy
        self.subset = subset
        self.duplicate_indices_ = None
        
    def fit(self, X, y=None):
        """Fit the duplicate handler."""
        return self
    
    def transform(self, X):
        """Transform data by handling duplicates."""
        X_copy = X.copy()
        
        # Detect duplicates
        if self.subset:
            duplicates = X_copy.duplicated(subset=self.subset, keep=False)
        else:
            duplicates = X_copy.duplicated(keep=False)
        
        self.duplicate_indices_ = X_copy[duplicates].index.tolist()
        
        if self.strategy == 'remove':
            # Remove all duplicates
            X_copy = X_copy[~duplicates]
        elif self.strategy == 'keep_first':
            # Keep first occurrence
            X_copy = X_copy.drop_duplicates(subset=self.subset, keep='first')
        elif self.strategy == 'keep_last':
            # Keep last occurrence
            X_copy = X_copy.drop_duplicates(subset=self.subset, keep='last')
        elif self.strategy == 'mark':
            # Mark duplicates with a new column
            X_copy['is_duplicate'] = duplicates
        
        logger.info(f"Duplicate handling: {len(self.duplicate_indices_)} duplicates detected")
        return X_copy

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering with domain knowledge."""
    
    def __init__(self):
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """Fit the feature engineer."""
        return self
    
    def transform(self, X):
        """Transform data with advanced feature engineering."""
        X_copy = X.copy()
        
        # 1. Age-based features
        if 'age' in X_copy.columns:
            X_copy['age_group'] = pd.cut(X_copy['age'], 
                                       bins=[0, 25, 35, 45, 55, 65, 100],
                                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            X_copy['is_senior'] = (X_copy['age'] >= 65).astype(int)
            X_copy['is_young_adult'] = (X_copy['age'] <= 25).astype(int)
        
        # 2. Education-based features
        if 'education' in X_copy.columns:
            # Create education level hierarchy
            education_hierarchy = {
                'Children': 0, 'Less than 1st grade': 1, '1st 2nd 3rd or 4th grade': 2,
                '5th or 6th grade': 3, '7th and 8th grade': 4, '9th grade': 5,
                '10th grade': 6, '11th grade': 7, '12th grade no diploma': 8,
                'High school graduate': 9, 'Some college but no degree': 10,
                'Associates degree-occup /vocational': 11, 'Associates degree-academic program': 12,
                'Bachelors degree(BA AB BS)': 13, 'Masters degree(MA MS MEng MEd MSW MBA)': 14,
                'Prof school degree (MD DDS DVM LLB JD)': 15, 'Doctorate degree(PhD EdD)': 16
            }
            X_copy['education_level'] = X_copy['education'].map(education_hierarchy).fillna(0)
            X_copy['has_college_degree'] = (X_copy['education_level'] >= 13).astype(int)
            X_copy['has_advanced_degree'] = (X_copy['education_level'] >= 14).astype(int)
        
        # 3. Work-related features
        if 'class_of_worker' in X_copy.columns and 'employment_status' in X_copy.columns:
            X_copy['is_self_employed'] = X_copy['class_of_worker'].str.contains('Self-employed', na=False).astype(int)
            X_copy['is_government_worker'] = X_copy['class_of_worker'].str.contains('Government', na=False).astype(int)
            X_copy['is_unemployed'] = X_copy['employment_status'].str.contains('Unemployed', na=False).astype(int)
        
        # 4. Financial features
        if 'capital_gains' in X_copy.columns:
            X_copy['has_capital_gains'] = (X_copy['capital_gains'] > 0).astype(int)
            X_copy['capital_gains_log'] = np.log1p(X_copy['capital_gains'])
        
        if 'capital_losses' in X_copy.columns:
            X_copy['has_capital_losses'] = (X_copy['capital_losses'] > 0).astype(int)
            X_copy['capital_losses_log'] = np.log1p(X_copy['capital_losses'])
        
        if 'dividends' in X_copy.columns:
            X_copy['has_dividends'] = (X_copy['dividends'] > 0).astype(int)
            X_copy['dividends_log'] = np.log1p(X_copy['dividends'])
        
        # 5. Family and household features
        if 'marital_status' in X_copy.columns:
            X_copy['is_married'] = X_copy['marital_status'].str.contains('Married', na=False).astype(int)
            X_copy['is_divorced_separated'] = X_copy['marital_status'].str.contains('Divorced|Separated', na=False).astype(int)
        
        if 'family_members_under_18' in X_copy.columns:
            # Convert to numeric first, handling any non-numeric values
            family_members = pd.to_numeric(X_copy['family_members_under_18'], errors='coerce').fillna(0)
            X_copy['has_children'] = (family_members > 0).astype(int)
        
        # 6. Geographic and migration features
        migration_cols = ['migration_msa', 'migration_reg', 'migration_within_reg', 'live_here_1_year']
        available_migration = [col for col in migration_cols if col in X_copy.columns]
        if available_migration:
            # Create migration indicator
            X_copy['is_migrant'] = 0
            for col in available_migration:
                X_copy['is_migrant'] += X_copy[col].notna().astype(int)
            X_copy['is_migrant'] = (X_copy['is_migrant'] > 0).astype(int)
        
        # 7. Work intensity features
        if 'weeks_worked_in_year' in X_copy.columns:
            # Convert to numeric first
            weeks_worked = pd.to_numeric(X_copy['weeks_worked_in_year'], errors='coerce').fillna(0)
            X_copy['work_intensity'] = pd.cut(weeks_worked,
                                            bins=[0, 13, 26, 39, 52],
                                            labels=['part_year', 'half_year', 'most_year', 'full_year'])
            X_copy['is_full_year_worker'] = (weeks_worked >= 50).astype(int)
        
        # 8. Interaction features
        if 'age' in X_copy.columns and 'education_level' in X_copy.columns:
            X_copy['age_education_interaction'] = X_copy['age'] * X_copy['education_level']
        
        if 'is_married' in X_copy.columns and 'has_children' in X_copy.columns:
            X_copy['married_with_children'] = X_copy['is_married'] * X_copy['has_children']
        
        logger.info(f"Feature engineering completed. New shape: {X_copy.shape}")
        return X_copy

class StrategicMissingValueHandler(BaseEstimator, TransformerMixin):
    """Strategic missing value handling based on feature types and patterns."""
    
    def __init__(self):
        self.imputers_ = {}
        self.missing_indicators_ = {}
        
    def fit(self, X, y=None):
        """Fit missing value imputers."""
        # Analyze missing patterns
        missing_patterns = X.isnull().sum() / len(X)
        
        for column in X.columns:
            missing_pct = missing_patterns[column]
            
            if missing_pct == 0:
                # No missing values
                continue
            elif missing_pct > 0.5:
                # High missing rate - use mode or create missing indicator
                if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                    self.imputers_[column] = SimpleImputer(strategy='constant', fill_value='Unknown')
                else:
                    self.imputers_[column] = SimpleImputer(strategy='median')
                self.missing_indicators_[column] = True
            elif missing_pct > 0.1:
                # Moderate missing rate - use KNN imputation for numerical, mode for categorical
                if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                    self.imputers_[column] = SimpleImputer(strategy='most_frequent')
                else:
                    # Use KNN imputation for numerical features
                    self.imputers_[column] = KNNImputer(n_neighbors=5)
                self.missing_indicators_[column] = True
            else:
                # Low missing rate - use simple strategies
                if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                    self.imputers_[column] = SimpleImputer(strategy='most_frequent')
                else:
                    self.imputers_[column] = SimpleImputer(strategy='median')
        
        # Fit imputers
        for column, imputer in self.imputers_.items():
            if isinstance(imputer, KNNImputer):
                # For KNN, we need numerical data only
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if column in numeric_cols and len(numeric_cols) > 1:
                    imputer.fit(X[numeric_cols])
            else:
                imputer.fit(X[[column]])
        
        return self
    
    def transform(self, X):
        """Transform data by handling missing values."""
        X_copy = X.copy()
        
        # Create missing indicators
        for column in self.missing_indicators_:
            if column in X_copy.columns:
                X_copy[f'{column}_was_missing'] = X_copy[column].isnull().astype(int)
        
        # Apply imputation
        for column, imputer in self.imputers_.items():
            if column in X_copy.columns:
                if isinstance(imputer, KNNImputer):
                    # For KNN, impute using all numerical columns
                    numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
                    if column in numeric_cols and len(numeric_cols) > 1:
                        X_imputed = imputer.transform(X_copy[numeric_cols])
                        col_idx = list(numeric_cols).index(column)
                        X_copy[column] = X_imputed[:, col_idx]
                else:
                    X_copy[column] = imputer.transform(X_copy[[column]]).ravel()
        
        logger.info(f"Missing value handling completed. Shape: {X_copy.shape}")
        return X_copy

class RareCategoryHandler(BaseEstimator, TransformerMixin):
    """Handle rare categories in categorical features."""
    
    def __init__(self, min_frequency=0.01, max_categories=50):
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.category_mappings_ = {}
        
    def fit(self, X, y=None):
        """Fit rare category handler."""
        for column in X.select_dtypes(include=['object']).columns:
            if column == TARGET_COLUMN:
                continue
                
            value_counts = X[column].value_counts()
            total_count = len(X)
            
            # Identify rare categories
            rare_categories = value_counts[value_counts / total_count < self.min_frequency].index.tolist()
            
            # If too many categories, keep only top N
            if len(value_counts) > self.max_categories:
                top_categories = value_counts.head(self.max_categories - 1).index.tolist()
                rare_categories.extend([cat for cat in value_counts.index if cat not in top_categories])
            
            self.category_mappings_[column] = rare_categories
        
        return self
    
    def transform(self, X):
        """Transform data by handling rare categories."""
        X_copy = X.copy()
        
        for column, rare_categories in self.category_mappings_.items():
            if column in X_copy.columns:
                X_copy[column] = X_copy[column].replace(rare_categories, 'Other')
        
        logger.info(f"Rare category handling completed for {len(self.category_mappings_)} columns")
        return X_copy

class DataPreprocessor:
    """Complete data preprocessing pipeline."""
    
    def __init__(self):
        self.pipeline = None
        self.label_encoder = None
        self.feature_names_ = None
        self.preprocessing_stats_ = {}
        
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create comprehensive preprocessing pipeline."""
        
        # Define preprocessing steps
        preprocessing_steps = [
            ('duplicate_handler', DuplicateHandler(strategy='keep_first')),
            ('feature_engineer', AdvancedFeatureEngineer()),
            ('missing_handler', StrategicMissingValueHandler()),
            ('rare_category_handler', RareCategoryHandler(
                min_frequency=MIN_CATEGORY_FREQUENCY,
                max_categories=MAX_CATEGORIES
            ))
        ]
        
        # Create pipeline
        self.pipeline = Pipeline(preprocessing_steps)
        
        logger.info("Preprocessing pipeline created with steps:")
        for name, step in preprocessing_steps:
            logger.info(f"  - {name}: {step.__class__.__name__}")
        
        return self.pipeline
    
    def fit_transform_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit preprocessing pipeline and transform data."""
        logger.info("Fitting preprocessing pipeline...")
        
        # Clean target variable consistently
        train_data_clean = train_data.copy()
        test_data_clean = test_data.copy()
        
        # Standardize target variable format
        train_data_clean[TARGET_COLUMN] = train_data_clean[TARGET_COLUMN].str.strip()
        test_data_clean[TARGET_COLUMN] = test_data_clean[TARGET_COLUMN].str.strip()
        
        # Handle inconsistent formats in both datasets
        train_data_clean[TARGET_COLUMN] = train_data_clean[TARGET_COLUMN].str.replace('50000+.', '50000+')
        train_data_clean[TARGET_COLUMN] = train_data_clean[TARGET_COLUMN].str.replace('-50000.', '-50000')
        
        test_data_clean[TARGET_COLUMN] = test_data_clean[TARGET_COLUMN].str.replace('- 50000', '-50000')
        test_data_clean[TARGET_COLUMN] = test_data_clean[TARGET_COLUMN].str.replace('-50000.', '-50000')
        test_data_clean[TARGET_COLUMN] = test_data_clean[TARGET_COLUMN].str.replace('50000+.', '50000+')
        
        # Separate features and target
        X_train = train_data_clean.drop(columns=[TARGET_COLUMN] + IGNORE_FEATURES, errors='ignore')
        y_train = train_data_clean[TARGET_COLUMN].copy()
        
        X_test = test_data_clean.drop(columns=[TARGET_COLUMN] + IGNORE_FEATURES, errors='ignore')
        y_test = test_data_clean[TARGET_COLUMN].copy()
        
        # Create and fit pipeline
        if self.pipeline is None:
            self.create_preprocessing_pipeline()
        
        # Fit on training data
        X_train_processed = self.pipeline.fit_transform(X_train)
        
        # Transform test data
        X_test_processed = self.pipeline.transform(X_test)
        
        # Align target variables with processed data (after duplicate removal)
        y_train_aligned = y_train.loc[X_train_processed.index]
        y_test_aligned = y_test.loc[X_test_processed.index]
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train_aligned)
        y_test_encoded = self.label_encoder.transform(y_test_aligned)
        
        # Final encoding for categorical features
        X_train_final, X_test_final = self._final_encoding(X_train_processed, X_test_processed)
        
        # Store preprocessing statistics
        self._compute_preprocessing_stats(X_train, X_train_final, y_train_aligned, y_train_encoded)
        
        logger.info(f"Preprocessing completed:")
        logger.info(f"  Training shape: {X_train.shape} -> {X_train_final.shape}")
        logger.info(f"  Test shape: {X_test.shape} -> {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train_encoded, y_test_encoded
    
    def _final_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply final encoding to categorical features."""
        
        # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform
        X_train_encoded = preprocessor.fit_transform(X_train)
        X_test_encoded = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = []
        
        # Numerical feature names
        feature_names.extend(numerical_cols)
        
        # Categorical feature names
        if len(categorical_cols) > 0:
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        
        # Convert to DataFrame
        X_train_final = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)
        
        self.feature_names_ = feature_names
        
        return X_train_final, X_test_final
    
    def _compute_preprocessing_stats(self, X_original: pd.DataFrame, X_processed: pd.DataFrame, 
                                   y_original: pd.Series, y_processed: np.ndarray) -> None:
        """Compute preprocessing statistics."""
        
        self.preprocessing_stats_ = {
            'original_features': X_original.shape[1],
            'processed_features': X_processed.shape[1],
            'feature_increase': X_processed.shape[1] - X_original.shape[1],
            'original_samples': len(X_original),
            'processed_samples': len(X_processed),
            'target_classes': len(np.unique(y_processed)),
            'class_distribution': pd.Series(y_processed).value_counts().to_dict()
        }
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: np.ndarray, 
                             strategy: str = 'smote') -> Tuple[pd.DataFrame, np.ndarray]:
        """Handle class imbalance using various strategies."""
        logger.info(f"Handling class imbalance using {strategy}...")
        
        original_distribution = pd.Series(y).value_counts()
        logger.info(f"Original class distribution: {original_distribution.to_dict()}")
        
        if strategy == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif strategy == 'undersampling':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            
        elif strategy == 'smoteenn':
            # SMOTE + Edited Nearest Neighbours
            smoteenn = SMOTEENN(random_state=RANDOM_STATE)
            X_resampled, y_resampled = smoteenn.fit_resample(X, y)
            
        elif strategy == 'balanced_sample':
            # Balanced sampling (combination of over and under sampling)
            minority_class = pd.Series(y).value_counts().idxmin()
            majority_class = pd.Series(y).value_counts().idxmax()
            
            minority_count = (y == minority_class).sum()
            target_count = min(minority_count * 3, (y == majority_class).sum())
            
            # Oversample minority class
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_temp, y_temp = smote.fit_resample(X, y)
            
            # Undersample majority class
            undersampler = RandomUnderSampler(
                sampling_strategy={majority_class: target_count, minority_class: target_count},
                random_state=RANDOM_STATE
            )
            X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
            
        else:
            # No resampling
            X_resampled, y_resampled = X, y
        
        new_distribution = pd.Series(y_resampled).value_counts()
        logger.info(f"New class distribution: {new_distribution.to_dict()}")
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        
        # Convert back to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        
        return X_resampled, y_resampled
    
    def save_pipeline(self, filepath: str) -> None:
        """Save preprocessing pipeline."""
        pipeline_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names_,
            'preprocessing_stats': self.preprocessing_stats_
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Preprocessing pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load preprocessing pipeline."""
        pipeline_data = joblib.load(filepath)
        
        self.pipeline = pipeline_data['pipeline']
        self.label_encoder = pipeline_data['label_encoder']
        self.feature_names_ = pipeline_data['feature_names']
        self.preprocessing_stats_ = pipeline_data['preprocessing_stats']
        
        logger.info(f"Preprocessing pipeline loaded from {filepath}")
    
    def generate_preprocessing_report(self) -> None:
        """Generate comprehensive preprocessing report."""
        logger.info("Generating preprocessing report...")
        
        report_content = f"""
# Data Preprocessing Report

## Pipeline Overview

The preprocessing pipeline consists of the following steps:
1. **Duplicate Handling**: Remove duplicate records
2. **Feature Engineering**: Create domain-specific features
3. **Missing Value Handling**: Strategic imputation based on missing patterns
4. **Rare Category Handling**: Group rare categories to reduce dimensionality
5. **Final Encoding**: StandardScaler for numerical, OneHotEncoder for categorical

## Preprocessing Statistics

- **Original Features**: {self.preprocessing_stats_['original_features']}
- **Processed Features**: {self.preprocessing_stats_['processed_features']}
- **Feature Increase**: {self.preprocessing_stats_['feature_increase']}
- **Original Samples**: {self.preprocessing_stats_['original_samples']}
- **Processed Samples**: {self.preprocessing_stats_['processed_samples']}
- **Target Classes**: {self.preprocessing_stats_['target_classes']}

## Class Distribution

Original class distribution:
{pd.Series(self.preprocessing_stats_['class_distribution']).to_string()}

## Feature Engineering

### Created Features:
1. **Age-based**: age_group, is_senior, is_young_adult
2. **Education-based**: education_level, has_college_degree, has_advanced_degree
3. **Work-based**: is_self_employed, is_government_worker, is_unemployed
4. **Financial**: has_capital_gains, has_capital_losses, has_dividends (with log transforms)
5. **Family**: is_married, is_divorced_separated, has_children
6. **Geographic**: is_migrant (from migration features)
7. **Work Intensity**: work_intensity, is_full_year_worker
8. **Interactions**: age_education_interaction, married_with_children

## Missing Value Strategy

- **High missing rate (>50%)**: Constant imputation + missing indicator
- **Moderate missing rate (10-50%)**: KNN imputation for numerical, mode for categorical + missing indicator
- **Low missing rate (<10%)**: Simple imputation (median/mode)

## Recommendations

1. **Model Selection**: Consider tree-based models that handle mixed data types well
2. **Feature Selection**: Use feature importance to identify most relevant engineered features
3. **Cross-Validation**: Use stratified CV to maintain class distribution
4. **Monitoring**: Track feature drift in production, especially for engineered features
"""
        
        # Save report
        with open(REPORTS_DIR / "preprocessing_report.md", "w") as f:
            f.write(report_content)
        
        logger.info(f"Preprocessing report saved to {REPORTS_DIR / 'preprocessing_report.md'}")

def run_phase2():
    """Run complete Phase 2: Data Preprocessing & Feature Engineering."""
    logger.info("Starting Phase 2: Data Preprocessing & Feature Engineering")
    
    # Load data from Phase 1
    from src.data_pipeline import DataPipeline
    
    data_pipeline = DataPipeline()
    train_data, test_data = data_pipeline.load_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit and transform data
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_data(train_data, test_data)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(
        X_train, y_train, strategy='balanced_sample'
    )
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced
    }
    
    joblib.dump(processed_data, MODELS_DIR / "processed_data.pkl")
    logger.info(f"Processed data saved to {MODELS_DIR / 'processed_data.pkl'}")
    
    # Save preprocessing pipeline
    preprocessor.save_pipeline(MODELS_DIR / "preprocessing_pipeline.pkl")
    
    # Generate report
    preprocessor.generate_preprocessing_report()
    
    results = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'balanced_train_shape': X_train_balanced.shape,
        'feature_count': len(preprocessor.feature_names_),
        'preprocessing_stats': preprocessor.preprocessing_stats_,
        'phase_status': 'completed'
    }
    
    logger.info("Phase 2 completed successfully!")
    return results

if __name__ == "__main__":
    # Run Phase 2
    results = run_phase2()
    print("Phase 2: Data Preprocessing & Feature Engineering completed successfully!")
    print(f"Training data shape: {results['train_shape']}")
    print(f"Test data shape: {results['test_shape']}")
    print(f"Balanced training data shape: {results['balanced_train_shape']}")
    print(f"Total features: {results['feature_count']}")

