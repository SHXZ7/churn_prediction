# 🧠 Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn using classification models. This project uses Logistic Regression, Random Forest, and XGBoost models, and includes data preprocessing, feature engineering, and model serialization for deployment.

---

## 📂 Project Structure
├── app.py # Backend Flask/FastAPI app for predictions
├── churn.csv # Customer churn dataset
├── churn_model_lr.pkl # Trained Logistic Regression model
├── churn_model_rf.pkl # Trained Random Forest model
├── churn_model_xgb.pkl # Trained XGBoost model
├── data.ipynb # Jupyter Notebook for EDA, preprocessing, training
├── feature_names.pkl # Pickled feature name list for model input alignment
├── scaler.pkl # Pickled StandardScaler object for input scaling

---

## 🧪 Features Used

- Customer tenure
- Monthly charges
- Total charges
- Internet service type
- Payment method
- Contract type
- Gender, Senior Citizen, Partner status, etc.


---

## ⚙️ Models Trained

1. Logistic Regression  
2. Random Forest Classifier  
3. XGBoost Classifier  

Each model was evaluated using accuracy, precision, recall, F1-score, and ROC AUC.

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/SHXZ7/churn_prediction.git
cd churn_prediction
