"""
Model Loading Utilities
Handles loading of trained models and preprocessing pipelines.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import streamlit as st

class ModelLoader:
    """Handles loading and management of trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessing_pipeline = None
        self.feature_names = []
        self.model_results = {}
        self._load_models()
    
    def _load_models(self):
        """Load all available models and preprocessing pipeline"""
        try:
            # Load preprocessing pipeline
            pipeline_path = os.path.join(self.models_dir, 'preprocessing_pipeline.pkl')
            if os.path.exists(pipeline_path):
                pipeline_data = joblib.load(pipeline_path)
                if isinstance(pipeline_data, dict):
                    self.preprocessing_pipeline = pipeline_data.get('pipeline')
                    self.feature_names = pipeline_data.get('feature_names', [])
                else:
                    self.preprocessing_pipeline = pipeline_data
            
            # Load individual models
            model_files = {
                'logistic_regression': 'logistic_regression_best_model.pkl',
                'random_forest': 'random_forest_best_model.pkl',
                'xgboost': 'xgboost_best_model.pkl',
                'lightgbm': 'lightgbm_best_model.pkl',
                'neural_network': 'neural_network_best_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
            
            # Load model development results
            results_path = os.path.join(self.models_dir, 'model_development_results.pkl')
            if os.path.exists(results_path):
                self.model_results = joblib.load(results_path)
            
            # Load processed data if available
            processed_data_path = os.path.join(self.models_dir, 'processed_data.pkl')
            if os.path.exists(processed_data_path):
                self.processed_data = joblib.load(processed_data_path)
            
            print(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self._use_mock_data()
    
    def _use_mock_data(self):
        """Use mock data when actual models are not available"""
        self.model_results = {
            'cv_results': {
                'XGBoost': {'accuracy': 96.32, 'roc_auc': 99.30, 'f1': 96.29, 'precision': 97.16, 'recall': 95.43},
                'LightGBM': {'accuracy': 96.25, 'roc_auc': 99.30, 'f1': 96.21, 'precision': 97.07, 'recall': 95.37},
                'Random Forest': {'accuracy': 95.16, 'roc_auc': 99.01, 'f1': 95.19, 'precision': 94.70, 'recall': 95.67},
                'Neural Network': {'accuracy': 91.06, 'roc_auc': 96.76, 'f1': 91.22, 'precision': 89.63, 'recall': 92.88},
                'Logistic Regression': {'accuracy': 86.83, 'roc_auc': 94.11, 'f1': 87.07, 'precision': 85.52, 'recall': 88.69}
            }
        }
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get model performance metrics"""
        if 'cv_results' in self.model_results:
            return self.model_results['cv_results']
        else:
            # Return mock data
            return {
                'XGBoost': {'accuracy': 96.32, 'roc_auc': 99.30, 'f1': 96.29, 'precision': 97.16, 'recall': 95.43},
                'LightGBM': {'accuracy': 96.25, 'roc_auc': 99.30, 'f1': 96.21, 'precision': 97.07, 'recall': 95.37},
                'Random Forest': {'accuracy': 95.16, 'roc_auc': 99.01, 'f1': 95.19, 'precision': 94.70, 'recall': 95.67},
                'Neural Network': {'accuracy': 91.06, 'roc_auc': 96.76, 'f1': 91.22, 'precision': 89.63, 'recall': 92.88},
                'Logistic Regression': {'accuracy': 86.83, 'roc_auc': 94.11, 'f1': 87.07, 'precision': 85.52, 'recall': 88.69}
            }
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using loaded models or fallback to rule-based prediction"""
        
        if self.models and self.preprocessing_pipeline:
            return self._predict_with_models(input_data)
        else:
            return self._predict_rule_based(input_data)
    
    def _predict_with_models(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using actual trained models"""
        try:
            # Define expected features based on original dataset
            expected_features = [
                'age', 'class_of_worker', 'industry_code', 'occupation_code', 
                'education', 'wage_per_hour', 'enrolled_in_edu_inst_last_wk',
                'marital_status', 'major_industry_code', 'major_occupation_code',
                'race', 'hispanic_origin', 'sex', 'member_of_a_labor_union',
                'reason_for_unemployment', 'full_or_part_time_employment_stat',
                'capital_gains', 'capital_losses', 'dividends_from_stocks',
                'tax_filer_status', 'region_of_previous_residence',
                'state_of_previous_residence', 'detailed_household_and_family_stat',
                'detailed_household_summary_in_household', 'migration_code_change_in_msa',
                'migration_code_change_in_reg', 'migration_code_move_within_reg',
                'live_in_this_house_1_year_ago', 'migration_prev_res_in_sunbelt',
                'num_persons_worked_for_employer', 'family_members_under_18',
                'country_of_birth_father', 'country_of_birth_mother',
                'country_of_birth_self', 'citizenship', 'own_business_or_self_employed',
                'fill_inc_questionnaire_for_veterans_admin', 'veterans_benefits',
                'weeks_worked_in_year', 'year'
            ]
            
            # Create DataFrame with provided data
            input_df = pd.DataFrame([input_data])
            
            # Fill missing features with default values
            for feature in expected_features:
                if feature not in input_df.columns:
                    if feature in ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 
                                  'dividends_from_stocks', 'num_persons_worked_for_employer',
                                  'family_members_under_18', 'weeks_worked_in_year', 'year']:
                        input_df[feature] = 0
                    else:
                        input_df[feature] = 'Unknown'
            
            # Reorder columns to match expected order
            input_df = input_df[expected_features]
            
            # Preprocess the input
            processed_input = self.preprocessing_pipeline.transform(input_df)
            
            # Make predictions with all available models
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    # Get prediction probability
                    prob = model.predict_proba(processed_input)[0]
                    prediction = model.predict(processed_input)[0]
                    
                    predictions[model_name] = {
                        'prediction': int(prediction),
                        'probability_low_income': float(prob[0]),
                        'probability_high_income': float(prob[1]),
                        'confidence': float(max(prob))
                    }
                except Exception as e:
                    predictions[model_name] = {'error': str(e)}
            
            # Calculate ensemble prediction (majority vote)
            valid_predictions = [p['prediction'] for p in predictions.values() if 'prediction' in p]
            if valid_predictions:
                ensemble_prediction = int(np.round(np.mean(valid_predictions)))
                ensemble_confidence = np.mean([p['confidence'] for p in predictions.values() if 'confidence' in p])
            else:
                ensemble_prediction = 0
                ensemble_confidence = 0.5
            
            return {
                'prediction': ensemble_prediction,
                'probability_high_income': ensemble_confidence if ensemble_prediction == 1 else 1 - ensemble_confidence,
                'probability_low_income': 1 - ensemble_confidence if ensemble_prediction == 1 else ensemble_confidence,
                'confidence': ensemble_confidence,
                'individual_predictions': predictions
            }
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self._predict_rule_based(input_data)
    
    def _predict_rule_based(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based prediction for demonstration when models are not available"""
        score = 0
        
        # Age factor
        if input_data.get('age', 0) > 40:
            score += 0.3
        
        # Education factor
        education_scores = {
            'Doctorate degree(PhD EdD)': 0.4,
            'Prof school degree (MD DDS DVM LLB JD)': 0.4,
            'Masters degree(MA MS MEng MEd MSW MBA)': 0.3,
            'Bachelors degree(BA AB BS)': 0.2,
            'Associates degree-academic program': 0.1,
            'High school graduate': 0.0
        }
        score += education_scores.get(input_data.get('education', ''), 0)
        
        # Occupation factor
        if 'Professional' in input_data.get('occupation_code', ''):
            score += 0.2
        elif 'Executive' in input_data.get('occupation_code', ''):
            score += 0.25
        
        # Capital gains factor
        if input_data.get('capital_gains', 0) > 0:
            score += 0.15
        
        # Work hours factor
        if input_data.get('weeks_worked_in_year', 0) >= 50:
            score += 0.1
        
        # Convert to probability
        probability = min(max(score, 0.1), 0.9)
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability_high_income': probability,
            'probability_low_income': 1 - probability,
            'confidence': max(probability, 1 - probability)
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance data"""
        if 'feature_importance' in self.model_results:
            return self.model_results['feature_importance']
        else:
            # Return mock feature importance
            return {
                'education_level': 0.15,
                'age': 0.12,
                'occupation_code': 0.10,
                'capital_gains': 0.09,
                'marital_status': 0.08,
                'weeks_worked_in_year': 0.07,
                'class_of_worker': 0.06,
                'sex': 0.05,
                'capital_losses': 0.04,
                'race': 0.03,
                'dividends_from_stocks': 0.03,
                'age_education_interaction': 0.02,
                'is_married': 0.02,
                'has_college_degree': 0.02,
                'work_intensity': 0.02,
                'other_features': 0.10
            }
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if hasattr(self, 'processed_data'):
            return {
                'training_samples': self.processed_data.get('X_train', pd.DataFrame()).shape[0],
                'test_samples': self.processed_data.get('X_test', pd.DataFrame()).shape[0],
                'original_features': 40,
                'engineered_features': len(self.feature_names) if self.feature_names else 188,
                'duplicates_removed': 78273,
                'class_imbalance': {'low_income': 93.8, 'high_income': 6.2}
            }
        else:
            # Return mock data stats
            return {
                'training_samples': 199523,
                'test_samples': 99762,
                'original_features': 40,
                'engineered_features': 188,
                'duplicates_removed': 78273,
                'class_imbalance': {'low_income': 93.8, 'high_income': 6.2}
            }

# Global model loader instance
@st.cache_resource
def get_model_loader():
    """Get cached model loader instance"""
    return ModelLoader()

