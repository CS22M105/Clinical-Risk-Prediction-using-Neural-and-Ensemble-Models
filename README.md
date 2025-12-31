# Clinical-Risk-Prediction-using-Neural-and-Ensemble-Models
An end-to-end machine learning project implementing state-of-the-art ensemble techniques and deep learning for heart disease prediction, achieving 93.16% AUC through hybrid stacked ensemble architecture.

## 🎯 Overview

This project demonstrates advanced machine learning techniques for predicting heart disease risk using the UCI Heart Disease dataset. The implementation showcases expertise in:

- **Advanced Ensemble Methods**: Hybrid stacking with neural network meta-learners
- **Deep Learning**: Custom MLP architectures with batch normalization and dropout
- **Feature Engineering**: Clinical feature creation and transformation
- **Model Optimization**: Hyperparameter tuning and cross-validation strategies
- **Production-Ready Code**: Modular, well-documented, and reproducible

**What this project demonstrates?:**
- 📊 Combines multiple modeling paradigms (tree-based, linear, and neural networks)
- 🧠 Implements sophisticated stacking ensemble with 3-level architecture
- 🔬 Demonstrates deep understanding of ML fundamentals and advanced techniques
- 📈 Achieves competitive performance (93.16% AUC) with proper validation
- 💻 Clean, professional code following best practices

## ✨ Key Features

### 1. **Multi-Model Ensemble Architecture**
```
Level 0: 9 Base Models (RF, XGBoost, LightGBM, CatBoost, etc.)
    ↓
Level 1: 3 Meta-Learners (Ridge, LightGBM, Logistic Regression)
    ↓
Level 2: Final Blender (XGBoost) → 93.16% AUC
```

### 2. **Advanced Feature Engineering**
- **Blood Pressure Features**: Pulse pressure, Mean Arterial Pressure (MAP)
- **Cardiovascular Ratios**: Cholesterol/Thalach, Oldpeak/Thalach ratios
- **Interaction Features**: Age-cholesterol interactions, angina severity scores
- **Transformations**: Standard scaling, quantile transformation for skewed features

### 3. **Four Modeling Approaches**
1. **Traditional ML Pipeline** (`project_code.py`)
2. **Deep Learning MLP** (`ML_project_mlp_model.ipynb`)
2. **Stacking Ensemble** (`stacking_ensemble.ipynb`)
3. **Hybrid Stacked Ensemble** (`hybrid_stacking.ipynb`)

## 📊 Results

### Performance Metrics

| Model | Test AUC | Test Accuracy | Key Strength |
|-------|----------|---------------|--------------|
| **Hybrid Stacking** | **93.16%** | **85.33%** | Best overall performance |
| Stacking + NN Meta | 93.16% | 85.33% | Complex pattern capture |
| RandomForest | 87.53% | 84.24% | Feature importance |
| XGBoost | 87.08% | 83.15% | Gradient boosting |
| Deep MLP | 89.34% | 77.17% | Non-linear relationships |
| Logistic Regression | 90.91% | 82.07% | Interpretability |

### Confusion Matrix (Hybrid Stacking)
```
              Predicted
            No Disease  Disease
Actual  No     64          18     (TPR: 78.0%)
      Disease   9          93     (TPR: 91.2%)

Precision: 88% (No Disease), 84% (Disease)
F1-Score: 0.83 (No Disease), 0.87 (Disease)
```

## 📁 Dataset

**UCI Heart Disease Dataset** (Cleveland, Hungarian, Switzerland, VA)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Total Samples**: 920 patients
- **Features**: 13 clinical attributes
- **Target**: Binary (0 = No Disease, 1 = Disease)

**Preprocessing Pipeline:**
1. Combined 4 datasets (Cleveland, Hungarian, Switzerland, VA)
2. Handled missing values (median imputation)
3. Feature engineering (9 new features created)
4. Scaling: StandardScaler + QuantileTransformer
5. 80-20 train-test split (stratified)

### Requirements
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
torch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📂 Project Structure

```
heart-disease-prediction/
│
├── notebooks/
│   ├── ML_project_mlp_model.ipynb      # Deep learning implementation
│   ├── stacking_ensemble.ipynb          # 2 level stacking ensemble
│   └── hybrid_stacking.ipynb           #  hybrid stacking ensemble
│
├── results/
│   ├── roc_curves.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── architecture.png
│
├── README.md
└── LICENSE
```

## 📈 Model Comparison

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Hybrid Stacking** | Highest accuracy, robust | Complex, slower training | Production deployment |
| **Deep MLP** | Captures non-linearities | Requires more data | Large datasets |
| **Traditional ML** | Fast, interpretable | Limited by linear assumptions | Quick prototyping |
