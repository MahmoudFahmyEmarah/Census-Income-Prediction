
# Model Development & Selection Report

## Model Comparison Summary

              Model  Best_CV_Score  Training_Time  CV_accuracy_mean  CV_accuracy_std  CV_precision_mean  CV_precision_std  CV_recall_mean  CV_recall_std  CV_f1_mean  CV_f1_std  CV_roc_auc_mean  CV_roc_auc_std
logistic_regression       0.941264    3870.990673          0.868304         0.001630           0.855186          0.004913        0.886837       0.003397    0.870704   0.001000         0.941101        0.001248
      random_forest       0.990340     449.926093          0.951620         0.000480           0.947049          0.002343        0.956744       0.001690    0.951867   0.000381         0.990078        0.000358
            xgboost       0.993262     201.602162          0.963187         0.001368           0.971599          0.002475        0.954275       0.001586    0.962856   0.001354         0.993022        0.000508
           lightgbm       0.993380     114.746660          0.962459         0.000908           0.970677          0.001693        0.953734       0.001413    0.962129   0.000907         0.992955        0.000450
     neural_network       0.968174     753.110505          0.910630         0.003446           0.896271          0.007248        0.928842       0.001770    0.912243   0.002939         0.967624        0.002139

## Best Models and Hyperparameters


### Logistic Regression

**Best CV Score (ROC-AUC):** 0.9413
**Training Time:** 3870.99 seconds

**Best Hyperparameters:**
- C: 10
- class_weight: balanced
- penalty: l2
- solver: liblinear

### Random Forest

**Best CV Score (ROC-AUC):** 0.9903
**Training Time:** 449.93 seconds

**Best Hyperparameters:**
- n_estimators: 200
- min_samples_split: 2
- min_samples_leaf: 1
- max_depth: None
- class_weight: balanced

### Xgboost

**Best CV Score (ROC-AUC):** 0.9933
**Training Time:** 201.60 seconds

**Best Hyperparameters:**
- subsample: 1.0
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.2
- colsample_bytree: 0.8

### Lightgbm

**Best CV Score (ROC-AUC):** 0.9934
**Training Time:** 114.75 seconds

**Best Hyperparameters:**
- subsample: 1.0
- num_leaves: 31
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.2

### Neural Network

**Best CV Score (ROC-AUC):** 0.9682
**Training Time:** 753.11 seconds

**Best Hyperparameters:**
- learning_rate: constant
- hidden_layer_sizes: (100,)
- alpha: 0.001
- activation: relu

## Feature Importance Analysis

### Top 10 Features - Logistic Regression

                                                   feature  importance
                                         capital_gains_log    7.356671
                                         has_capital_gains    7.062048
                          class_of_worker_ Not in universe    4.457097
         education_ Prof school degree (MD DDS DVM LLB JD)    3.329144
                                           education_Other    3.201819
       major_occupation_code_ Farming forestry and fishing    3.093794
         education_ Masters degree(MA MS MEng MEd MSW MBA)    2.750250
 major_occupation_code_ Transportation and material moving    2.348282
major_occupation_code_ Precision production craft & repair    2.270709
                                           occupation_code    2.135221

### Top 10 Features - Random Forest

                              feature  importance
                            sex_ Male    0.066977
                      occupation_code    0.059754
                                  age    0.056523
                  is_full_year_worker    0.049826
education_ Bachelors degree(BA AB BS)    0.037735
                 weeks_worked_in_year    0.033214
      num_persons_worked_for_employer    0.032339
                        industry_code    0.030053
      education_ High school graduate    0.028085
                            dividends    0.026083

### Top 10 Features - Xgboost

                              feature  importance
                  is_full_year_worker    0.241799
                       is_young_adult    0.045225
                            sex_ Male    0.038205
                 weeks_worked_in_year    0.024234
           tax_filer_status_ Nonfiler    0.022210
education_ Bachelors degree(BA AB BS)    0.021593
      education_ High school graduate    0.020945
 major_occupation_code_ Other service    0.017932
                      occupation_code    0.016564
                      age_group_46-55    0.015920

### Top 10 Features - Lightgbm

                        feature  importance
                            age         630
                occupation_code         384
num_persons_worked_for_employer         298
                  industry_code         268
                      dividends         191
                  capital_gains         183
           weeks_worked_in_year         154
                      sex_ Male         152
                  dividends_log         116
                 capital_losses         110


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
