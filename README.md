# Credit-Card-Detection-Model      
## Introduction
Credit card fraud is a significant challenge in the financial industry, leading to substantial financial losses and security concerns. This project focuses on developing a machine learning-based fraud detection system to accurately identify fraudulent transactions while minimizing false positives. Using a real-world credit card fraud dataset, this model applies advanced data preprocessing techniques, feature engineering, and supervised learning algorithms to classify transactions as fraudulent or legitimate. The project explores various classification models, including Logistic Regression, Random Forest, XGBoost, and Neural Networks, comparing their performance in detecting fraud.
## Key Features for this Project
- Data Preprocessing – Handling imbalanced data, feature scaling, and removing anomalies.
- Exploratory Data Analysis (EDA) – Understanding transaction patterns and fraud indicators.
- Model Training & Evaluation – Implementing multiple machine learning models and assessing their accuracy, precision, recall, and F1-score.
- Hyperparameter Tuning – Optimizing model performance using Grid Search and Random Search.
- Deployment Ready – Building a real-time fraud detection system with Flask API integration.
This repository provides a step-by-step implementation of the fraud detection model, including Python code, Jupyter notebooks, and documentation. It serves as a valuable resource for data scientists, financial analysts, and machine learning enthusiasts looking to explore real-world fraud detection use cases.

- Tech Stack: Python | Pandas | Scikit-learn | TensorFlow | Flask | SQL

- Dataset: Credit Card Fraud Detection Dataset (Kaggle)
## Loading the dataset
This involves importing the necessary libriaries and loading the data set
```
import pandas as pd
df = pd.read_csv("creditcard_2023.csv")
df.head()
```
## Understanding the Dataset
This involves checking dataset information and handling missing values if any
```
df.info()
print(df.isnull().sum())
print(df["Class"].value_counts())
```
![image](https://github.com/user-attachments/assets/5bb28a2e-2ce6-47c3-8b3c-e0c148566f5e)

The results show that there are 568630 rows and 31 columns, and that there is no null value
## Data Preprocessing
Although fraud cases are rare but in this case, there is a balance between fraud and no fraud cases, and as such no need to use SMOTE command to balance the transactions. The dataset are divided into X and y segment for preparation for training and testing using the code snippet below:
```
X = df.drop(columns=['Class'])
y = df['Class']
```
## Splitting Data for Training and Testing
Here, the dataset is split into training and testing in preparation for building the predictive model
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Building Predictive Model (Regression Model)
Here, the predictive model (regression model) using the necessary libraries as shown:
```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
![image](https://github.com/user-attachments/assets/3e13b5dc-d7bd-48b9-b4fe-b9fd324aee16)The output above shows that a Logistic Regression model has successfully been created with a max_iter=500 parameter.
## Model Evaluation
Model evaluation is critical to building predictive model because it requires a highly reliable model that minimizes both false positives and false negatives. A well-evaluated model provides confidence that the predictions are reliable. To achive this, the following code snippets are used:
```
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

```
![image](https://github.com/user-attachments/assets/37b95694-cd4c-48f8-bd24-99135257b4f9)
- The model correctly classified 56,680 legitimate transactions as non-fraud and correctly classified 56,856 fraud transactions.
- 70 legitimate transactions were mistakenly classified as fraud.
- 120 fraud transactions were mistakenly classified as non-fraud.
- Precision (1.00 for both classes): The model rarely misclassifies fraud as non-fraud (few false positives).
- Recall (1.00 for both classes): The model correctly identifies almost all fraud cases (few false negatives).
- F1-score (1.00 for both classes): This is the balance between precision and recall, showing the model is excellent at fraud detection.
Other models including Random Forest and XGBoost are also evaluated to investigate their performances and select the best performing model.

## Building Predictive Model (Random Forest)
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
### Train and Evaluate Random Forest
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

### Evaluate Performance
```
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_rf))
```
![image](https://github.com/user-attachments/assets/5f363454-6ea0-47ea-8d3b-06dcee3f3669)
The results above shows the following:
- Precision (1.00 for both classes) → The model almost never misclassifies non-fraudulent transactions as fraud and vice versa.
- Recall (1.00 for both classes) → The model captures nearly all actual fraud cases.
- F1-score (1.00 for both classes) → Perfect balance between precision and recall.
- Overall Accuracy = 100% (Model correctly classified all cases).
- ROC-AUC Score = 0.9998 (~99.98%) → Measures how well the model separates fraud from non-fraud.
- Closer to 1 means a nearly perfect classifier.

## Train and Evaluate XGBoost
XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting algorithm that works exceptionally well on structured data.

### Importing the Necessary Libraries and Fit the Model
```
import xgboost as xgb
from sklearn.metrics import accuracy_score
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

### Evakyate Performance
The performance of this model is evaluted  using the code snippet below:
```
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_xgb))
```
![image](https://github.com/user-attachments/assets/5d6e4ecc-b41a-45ea-8fc9-1de3a60cf124)
The result above shows that the model
- has high precision level
- has high recall level
- F1-Score: 1.00 for both classes → This means a perfect balance between precision and recall.
- Accuracy: 1.00 (100%) → The model classified all 113,726 instances correctly.
- ROC-AUC Score: 0.9998 (~1.00) → The model has near-perfect discrimination between classes
