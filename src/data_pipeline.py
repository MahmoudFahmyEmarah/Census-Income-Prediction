"""
Phase 1: Data Infrastructure & EDA
Scalable data pipeline with comprehensive analysis and data leakage detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Scalable data pipeline for census income prediction."""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.data_quality_report = {}
        self.leakage_report = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and properly map column names for train and test data."""
        logger.info("Loading training and test datasets...")
        
        try:
            # Load training data
            self.train_data = pd.read_csv(TRAIN_FILE, header=None)
            self.train_data.columns = [COLUMN_MAPPING[i] for i in range(len(self.train_data.columns))]
            
            # Load test data
            self.test_data = pd.read_csv(TEST_FILE, header=None)
            self.test_data.columns = [COLUMN_MAPPING[i] for i in range(len(self.test_data.columns))]
            
            # Clean target variable
            self.train_data[TARGET_COLUMN] = self.train_data[TARGET_COLUMN].str.strip()
            self.test_data[TARGET_COLUMN] = self.test_data[TARGET_COLUMN].str.strip()
            
            # Handle inconsistent target format in test data
            self.test_data[TARGET_COLUMN] = self.test_data[TARGET_COLUMN].str.replace('- 50000', '-50000')
            self.test_data[TARGET_COLUMN] = self.test_data[TARGET_COLUMN].str.replace('50000+.', '50000+')
            
            logger.info(f"Training data shape: {self.train_data.shape}")
            logger.info(f"Test data shape: {self.test_data.shape}")
            
            return self.train_data, self.test_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        logger.info("Performing data quality assessment...")
        
        quality_report = {}
        
        for dataset_name, dataset in [("train", self.train_data), ("test", self.test_data)]:
            report = {
                "shape": dataset.shape,
                "missing_values": dataset.isnull().sum().to_dict(),
                "missing_percentage": (dataset.isnull().sum() / len(dataset) * 100).to_dict(),
                "duplicates": dataset.duplicated().sum(),
                "data_types": dataset.dtypes.to_dict(),
                "unique_values": {col: dataset[col].nunique() for col in dataset.columns},
                "memory_usage": dataset.memory_usage(deep=True).sum() / 1024**2  # MB
            }
            
            # Check for constant columns
            constant_cols = [col for col in dataset.columns if dataset[col].nunique() <= 1]
            report["constant_columns"] = constant_cols
            
            # Check for high cardinality categorical features
            high_cardinality = {
                col: dataset[col].nunique() 
                for col in CATEGORICAL_FEATURES 
                if col in dataset.columns and dataset[col].nunique() > MAX_CATEGORIES
            }
            report["high_cardinality_features"] = high_cardinality
            
            quality_report[dataset_name] = report
        
        self.data_quality_report = quality_report
        
        # Save quality report
        quality_df = pd.DataFrame({
            'Feature': self.train_data.columns,
            'Train_Missing_Count': [quality_report['train']['missing_values'][col] for col in self.train_data.columns],
            'Train_Missing_Pct': [quality_report['train']['missing_percentage'][col] for col in self.train_data.columns],
            'Test_Missing_Count': [quality_report['test']['missing_values'][col] for col in self.train_data.columns],
            'Test_Missing_Pct': [quality_report['test']['missing_percentage'][col] for col in self.train_data.columns],
            'Train_Unique': [quality_report['train']['unique_values'][col] for col in self.train_data.columns],
            'Test_Unique': [quality_report['test']['unique_values'][col] for col in self.train_data.columns],
            'Data_Type': [str(quality_report['train']['data_types'][col]) for col in self.train_data.columns]
        })
        
        quality_df.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)
        logger.info(f"Data quality report saved to {REPORTS_DIR / 'data_quality_report.csv'}")
        
        return quality_report
    
    def detect_data_leakage(self) -> Dict[str, Any]:
        """Detect potential data leakage by analyzing correlations and temporal relationships."""
        logger.info("Detecting potential data leakage...")
        
        leakage_report = {
            "high_correlations": {},
            "temporal_leakage": {},
            "target_leakage": {},
            "suspicious_features": []
        }
        
        # Prepare data for correlation analysis
        train_encoded = self.train_data.copy()
        
        # Encode categorical variables for correlation analysis
        for col in CATEGORICAL_FEATURES:
            if col in train_encoded.columns:
                train_encoded[col] = pd.Categorical(train_encoded[col]).codes
        
        # Encode target variable
        train_encoded[TARGET_COLUMN] = (train_encoded[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
        
        # Calculate correlations with target
        correlations = train_encoded.corr()[TARGET_COLUMN].abs().sort_values(ascending=False)
        high_corr_features = correlations[correlations > LEAKAGE_THRESHOLD]
        
        # Remove target column if it exists in the correlations
        if TARGET_COLUMN in high_corr_features.index:
            high_corr_features = high_corr_features.drop(TARGET_COLUMN)
        
        if len(high_corr_features) > 0:
            leakage_report["high_correlations"] = high_corr_features.to_dict()
            leakage_report["suspicious_features"].extend(high_corr_features.index.tolist())
        
        # Check for temporal leakage (future information)
        temporal_features = ["year", "tax_filer_status"]
        for feature in temporal_features:
            if feature in train_encoded.columns:
                # Check if feature values are suspiciously aligned with target
                contingency = pd.crosstab(train_encoded[feature], train_encoded[TARGET_COLUMN])
                if len(contingency) > 1:
                    chi2_stat = ((contingency - contingency.sum(axis=1).values.reshape(-1,1) * 
                                 contingency.sum(axis=0).values / contingency.sum().sum())**2 / 
                                (contingency.sum(axis=1).values.reshape(-1,1) * 
                                 contingency.sum(axis=0).values / contingency.sum().sum())).sum().sum()
                    leakage_report["temporal_leakage"][feature] = chi2_stat
        
        # Check specific features that might contain target information
        for feature in LEAKAGE_FEATURES_TO_CHECK:
            if feature in train_encoded.columns:
                correlation = abs(train_encoded[feature].corr(train_encoded[TARGET_COLUMN]))
                if correlation > LEAKAGE_THRESHOLD:
                    leakage_report["target_leakage"][feature] = correlation
                    if feature not in leakage_report["suspicious_features"]:
                        leakage_report["suspicious_features"].append(feature)
        
        self.leakage_report = leakage_report
        
        # Save leakage report
        with open(REPORTS_DIR / "data_leakage_report.txt", "w") as f:
            f.write("DATA LEAKAGE DETECTION REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write("HIGH CORRELATIONS WITH TARGET:\\n")
            for feature, corr in leakage_report["high_correlations"].items():
                f.write(f"  {feature}: {corr:.4f}\\n")
            
            f.write("\\nTEMPORAL LEAKAGE ANALYSIS:\\n")
            for feature, stat in leakage_report["temporal_leakage"].items():
                f.write(f"  {feature}: Chi-square statistic = {stat:.4f}\\n")
            
            f.write("\\nTARGET LEAKAGE ANALYSIS:\\n")
            for feature, corr in leakage_report["target_leakage"].items():
                f.write(f"  {feature}: Correlation = {corr:.4f}\\n")
            
            f.write("\\nSUSPICIOUS FEATURES:\\n")
            for feature in leakage_report["suspicious_features"]:
                f.write(f"  - {feature}\\n")
        
        logger.info(f"Data leakage report saved to {REPORTS_DIR / 'data_leakage_report.txt'}")
        
        return leakage_report
    
    def create_comprehensive_visualizations(self) -> None:
        """Create comprehensive visualizations for ALL features."""
        logger.info("Creating comprehensive visualizations for all features...")
        
        # Create visualization directory
        viz_dir = REPORTS_DIR / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Target distribution
        self._plot_target_distribution(viz_dir)
        
        # 2. Continuous features analysis
        self._plot_continuous_features(viz_dir)
        
        # 3. Categorical features analysis
        self._plot_categorical_features(viz_dir)
        
        # 4. Missing values heatmap
        self._plot_missing_values(viz_dir)
        
        # 5. Correlation matrix
        self._plot_correlation_matrix(viz_dir)
        
        # 6. Feature distributions by target
        self._plot_feature_target_relationships(viz_dir)
        
        # 7. Data quality overview
        self._plot_data_quality_overview(viz_dir)
        
        logger.info(f"All visualizations saved to {viz_dir}")
    
    def _plot_target_distribution(self, viz_dir: Path) -> None:
        """Plot target variable distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training data
        train_counts = self.train_data[TARGET_COLUMN].value_counts()
        axes[0].pie(train_counts.values, labels=train_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Training Data - Target Distribution')
        
        # Test data
        test_counts = self.test_data[TARGET_COLUMN].value_counts()
        axes[1].pie(test_counts.values, labels=test_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Test Data - Target Distribution')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "01_target_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_continuous_features(self, viz_dir: Path) -> None:
        """Plot distributions of continuous features."""
        continuous_cols = [col for col in CONTINUOUS_FEATURES if col in self.train_data.columns]
        
        if not continuous_cols:
            return
        
        n_cols = 3
        n_rows = (len(continuous_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(continuous_cols):
            if i < len(axes):
                # Distribution plot
                self.train_data[col].hist(bins=50, alpha=0.7, ax=axes[i])
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(continuous_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "02_continuous_features.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_categorical_features(self, viz_dir: Path) -> None:
        """Plot distributions of categorical features."""
        categorical_cols = [col for col in CATEGORICAL_FEATURES if col in self.train_data.columns]
        
        # Plot top categorical features (limit to avoid overcrowding)
        top_categorical = categorical_cols[:12]  # Top 12 for visualization
        
        n_cols = 3
        n_rows = (len(top_categorical) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(top_categorical):
            if i < len(axes):
                # Get top categories to avoid overcrowding
                top_cats = self.train_data[col].value_counts().head(10)
                top_cats.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{col} Distribution (Top 10)')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(top_categorical), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "03_categorical_features.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_missing_values(self, viz_dir: Path) -> None:
        """Plot missing values heatmap."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Training data missing values
        missing_train = self.train_data.isnull().sum().sort_values(ascending=False)
        missing_train = missing_train[missing_train > 0]
        
        if len(missing_train) > 0:
            missing_train.plot(kind='bar', ax=axes[0])
            axes[0].set_title('Missing Values - Training Data')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
        else:
            axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Missing Values - Training Data')
        
        # Test data missing values
        missing_test = self.test_data.isnull().sum().sort_values(ascending=False)
        missing_test = missing_test[missing_test > 0]
        
        if len(missing_test) > 0:
            missing_test.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Missing Values - Test Data')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Missing Values - Test Data')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "04_missing_values.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, viz_dir: Path) -> None:
        """Plot correlation matrix for continuous features."""
        continuous_cols = [col for col in CONTINUOUS_FEATURES if col in self.train_data.columns]
        
        if len(continuous_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = self.train_data[continuous_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix - Continuous Features')
        plt.tight_layout()
        plt.savefig(viz_dir / "05_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_target_relationships(self, viz_dir: Path) -> None:
        """Plot relationships between features and target variable."""
        # Select key features for target relationship analysis
        key_features = ['age', 'education', 'marital_status', 'sex', 'race', 'capital_gains']
        available_features = [f for f in key_features if f in self.train_data.columns]
        
        n_cols = 2
        n_rows = (len(available_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                if feature in CONTINUOUS_FEATURES:
                    # Box plot for continuous features
                    self.train_data.boxplot(column=feature, by=TARGET_COLUMN, ax=axes[i])
                    axes[i].set_title(f'{feature} by Income Level')
                else:
                    # Count plot for categorical features
                    pd.crosstab(self.train_data[feature], self.train_data[TARGET_COLUMN]).plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{feature} by Income Level')
                    axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "06_feature_target_relationships.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_quality_overview(self, viz_dir: Path) -> None:
        """Plot data quality overview."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Data completeness
        completeness = (1 - self.train_data.isnull().sum() / len(self.train_data)) * 100
        completeness.sort_values().tail(20).plot(kind='barh', ax=axes[0,0])
        axes[0,0].set_title('Data Completeness (Top 20 Features)')
        axes[0,0].set_xlabel('Completeness %')
        
        # 2. Unique value counts
        unique_counts = pd.Series({col: self.train_data[col].nunique() for col in self.train_data.columns})
        unique_counts.sort_values(ascending=False).head(20).plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Unique Value Counts (Top 20 Features)')
        axes[0,1].set_ylabel('Unique Values')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Data types distribution
        dtype_counts = self.train_data.dtypes.value_counts()
        axes[1,0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Data Types Distribution')
        
        # 4. Memory usage by feature
        memory_usage = self.train_data.memory_usage(deep=True).sort_values(ascending=False).head(15)
        memory_usage.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Memory Usage by Feature (Top 15)')
        axes[1,1].set_ylabel('Memory (bytes)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "07_data_quality_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_eda_report(self) -> None:
        """Generate comprehensive EDA report."""
        logger.info("Generating comprehensive EDA report...")
        
        report_content = f"""
# Census Income Prediction - Exploratory Data Analysis Report

## Dataset Overview

### Training Data
- **Shape**: {self.train_data.shape}
- **Memory Usage**: {self.data_quality_report['train']['memory_usage']:.2f} MB
- **Duplicates**: {self.data_quality_report['train']['duplicates']}

### Test Data
- **Shape**: {self.test_data.shape}
- **Memory Usage**: {self.data_quality_report['test']['memory_usage']:.2f} MB
- **Duplicates**: {self.data_quality_report['test']['duplicates']}

## Target Variable Analysis

### Training Data Distribution
{self.train_data[TARGET_COLUMN].value_counts().to_string()}

### Test Data Distribution
{self.test_data[TARGET_COLUMN].value_counts().to_string()}

## Data Quality Issues

### Missing Values (Training Data)
"""
        
        # Add missing values analysis
        missing_train = pd.Series(self.data_quality_report['train']['missing_values'])
        missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
        
        if len(missing_train) > 0:
            report_content += f"\\n{missing_train.to_string()}\\n"
        else:
            report_content += "\\nNo missing values found in training data.\\n"
        
        # Add high cardinality features
        high_card = self.data_quality_report['train']['high_cardinality_features']
        if high_card:
            report_content += f"\\n### High Cardinality Features\\n"
            for feature, count in high_card.items():
                report_content += f"- {feature}: {count} unique values\\n"
        
        # Add data leakage analysis
        report_content += f"""
## Data Leakage Analysis

### Suspicious Features
"""
        if self.leakage_report['suspicious_features']:
            for feature in self.leakage_report['suspicious_features']:
                report_content += f"- {feature}\\n"
        else:
            report_content += "No suspicious features detected.\\n"
        
        # Add high correlations
        if self.leakage_report['high_correlations']:
            report_content += f"\\n### High Correlations with Target\\n"
            for feature, corr in self.leakage_report['high_correlations'].items():
                report_content += f"- {feature}: {corr:.4f}\\n"
        
        # Add recommendations
        report_content += f"""
## Recommendations

### Data Preprocessing
1. Handle missing values in migration-related features
2. Consider feature engineering for high-cardinality categorical variables
3. Address class imbalance in target variable

### Feature Engineering
1. Create age groups from continuous age variable
2. Combine related categorical features
3. Create interaction features between education and work class

### Model Development
1. Use stratified sampling for cross-validation
2. Apply class balancing techniques
3. Consider ensemble methods for better performance

### Data Leakage Prevention
1. Remove or carefully handle features with high target correlation
2. Validate temporal consistency in features
3. Implement proper train/validation/test splits
"""
        
        # Save report
        with open(REPORTS_DIR / "eda_report.md", "w") as f:
            f.write(report_content)
        
        logger.info(f"EDA report saved to {REPORTS_DIR / 'eda_report.md'}")
    
    def run_phase1(self) -> Dict[str, Any]:
        """Run complete Phase 1: Data Infrastructure & EDA."""
        logger.info("Starting Phase 1: Data Infrastructure & EDA")
        
        # Load data
        train_data, test_data = self.load_data()
        
        # Assess data quality
        quality_report = self.assess_data_quality()
        
        # Detect data leakage
        leakage_report = self.detect_data_leakage()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Generate EDA report
        self.generate_eda_report()
        
        results = {
            "train_shape": train_data.shape,
            "test_shape": test_data.shape,
            "data_quality": quality_report,
            "leakage_analysis": leakage_report,
            "phase_status": "completed"
        }
        
        logger.info("Phase 1 completed successfully!")
        return results

if __name__ == "__main__":
    # Run Phase 1
    pipeline = DataPipeline()
    results = pipeline.run_phase1()
    print("Phase 1: Data Infrastructure & EDA completed successfully!")
    print(f"Training data shape: {results['train_shape']}")
    print(f"Test data shape: {results['test_shape']}")
    print(f"Suspicious features detected: {len(results['leakage_analysis']['suspicious_features'])}")

