# Home Credit Default Risk

## 1. Project Introduction
This project focuses on the Kaggle competition "Home Credit Default Risk". The goal is to predict whether an applicant will be able to repay a loan. We utilize various data sources including application information, bureau data, previous applications, and credit card balance history to build predictive models. The project includes Exploratory Data Analysis (EDA), feature engineering, and model training using LightGBM.

## 2. Environment Setup
To set up the environment, please install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## 3. Dataset Download
You can download the dataset from the following link:
[Dataset Download Link](https://drive.google.com/drive/folders/1VZZ8yQoUKR1PR2ZX_nEENdAtWUrW8zEg?usp=sharing)

Please ensure the data is placed in the correct directory as expected by the scripts (typically `/home/liyiming/home-credit-default-risk/` or update the `DATA_PATH` variable in the scripts).

## 4. Model Download
Pre-trained models and related files can be downloaded here:
[Model Download Link](https://drive.google.com/drive/folders/1HlSipgFQhyeaofzNeOuqaVJ_QdSlKzKR?usp=sharing)

## 5. How to Run Training

### Run Baseline Model
The baseline model performs basic data processing, feature engineering, and trains a LightGBM model.

```bash
python home-credit-baseline.py
```

It will automatically:
1. Load and process all data tables.
2. Perform feature engineering.
3. Train the LightGBM model.
4. Generate predictions and save them to `submission.csv`.

### Run Advanced Model
For the advanced model with enhanced feature engineering (100+ new features) and K-Fold cross-validation:

```bash
python home-credit-advanced.py
```

This script includes:
- Advanced feature engineering
- 5-Fold Stratified Cross-Validation
- Optimized hyperparameters
- Feature importance analysis

### Run Data Analysis
To explore the data and generate visualization plots:

```bash
cd "data analysis"
python eda_analysis.py
```

## 6. Experimental Results

Below are the performance comparisons and ablation studies conducted during the project.

### Table 1: Performance Comparison within LightGBM Models
| Model | AUC ↑ |
| :--- | :--- |
| LightGBM* | 0.784 |
| LightGBM | **0.793** |
| LightGBM (five models ensembled) | **0.795** |

### Table 2: Performance Comparison with other Classification Models
| Model | AUC ↑ |
| :--- | :--- |
| Linear Regression | 0.737 |
| Support Vector Machine | 0.741 |
| Decision Tree | 0.743 |
| Random Forest | 0.760 |
| XGBoost | 0.790 |
| LightGBM | **0.793** |

### Table 3: Performance Comparison when Removing Specific Feature Sets (Ablation Study)
| Model | AUC ↑ |
| :--- | :--- |
| LightGBM w/o RS & FHI | 0.791 |
| LightGBM w/o Ext Source | 0.787 |
| LightGBM w/o Aggregations | 0.769 |
| LightGBM w/o Domain | 0.773 |
| LightGBM | **0.793** |

### Table 4: Performance Comparison with Different Training Techniques
| Model | AUC ↑ |
| :--- | :--- |
| LightGBM w/o regularization penalties | 0.788 |
| LightGBM w/o stratified sampling | 0.780 |
| LightGBM w focal loss | 0.791 |
| LightGBM | **0.793** |
