# **Sales Prediction – Machine Learning Web App**

🔗 **[Live Demo](https://g-pujitha1102-sales-prediction-app-pfjfaf.streamlit.app)**  
🔗 **[GitHub Repository](https://github.com/G-Pujitha1102/Sales-Prediction)**  

---

## **📌 Project Overview**

This project predicts **future sales** using **Machine Learning models** – Linear Regression and Random Forest Regressor – based on provided sales datasets. It includes **data preprocessing, model training, evaluation metrics, and visualizations** in a **Streamlit web app**, making it interactive and accessible online.

---

## **🎯 Objective**

 Analyze historical advertising data through Exploratory Data Analysis (EDA), uncover patterns and relationships between TV, Radio, and Newspaper budgets and sales.Enable users to input budgets and predict sales using both Linear Regression and Random Forest models.Classify sales as Low, Average, or High.Visually compare the predictions through bar charts.Compare model performance using evaluation metrics (RMSE, MAE, R²) and indicate the most accurate model for data-driven marketing and business decisions.


---

## 📁 Files

- `data/advertising.csv` — Dataset containing advertising budgets and sales.
- `models/linear_regression_model.pkl` — Saved Linear Regression model.
- `models/random_forest_model.pkl` — Saved Random Forest model.
- `app.py` — Streamlit web app for predictions, comparisons, and EDA.
- `requirements.txt` — Python dependencies.
- `outputs/predictions.csv` — Generated file with actual vs predicted sales.
- `outputs/metrics.csv` — Generated file with model evaluation metrics.

---

## 🧪 Methodology

1. **Data Loading & Exploration**  
   - Loaded dataset with advertising budgets and sales data.  
   - Explored summary statistics and distributions.

2. **Model Training**  
   - Built Linear Regression and Random Forest models to predict sales.  
   - Trained models on the dataset features (TV, Radio, Newspaper budgets).

3. **Model Evaluation**  
   - Calculated metrics: R² Score, RMSE (Root Mean Squared Error), MAE (Mean Absolute Error).  
   - Compared models based on these metrics.

4. **Visualization & Prediction**  
   - Interactive web app to input advertising budgets and get sales predictions.  
   - Visual comparisons of model predictions and EDA plots.

---

## 🔧 Technologies Used

- **Programming Language:** Python  
- **Libraries:** Streamlit, pandas, scikit-learn, plotly, seaborn, matplotlib, numpy, joblib  
- **IDE:** VS Code / Any Python IDE  
- **Version Control:** Git and GitHub  

---

## 📊 Model Performance Comparison

| Model             | RMSE    | MAE     | R² Score |
|-------------------|---------|---------|----------|
| Linear Regression | 1.64    | 1.23    | 0.90     |
| Random Forest     | 0.43    | 0.30    | 0.99     |

🎉 **Random Forest** performed the best based on RMSE, MAE, and R² score.

comparision chart-<img width="556" height="497" alt="bar chart R2" src="https://github.com/user-attachments/assets/aefd9ac7-0056-4db9-abaa-533525a9e539" />

---

## 📊 Bar Chart Comparison

Bar charts were created using Plotly to visually compare RMSE, MAE, and R² values for both models.

---

## 📊 Sample Visualizations
**📌 Application Overview**
<img width="1680" height="923" alt="eda page 1" src="https://github.com/user-attachments/assets/8075e34d-48b7-4ada-9cdf-2deff99932ca" />
**📊 Exploratory Data Analysis (EDA)**
<img width="1680" height="923" alt="eda page 1" src="https://github.com/user-attachments/assets/3ad2e140-7723-4a84-8ee0-62f6d9b5fb9c" />
<img width="1676" height="925" alt="eda page 2" src="https://github.com/user-attachments/assets/b333a082-a100-4657-b164-8dd4ab3d3ec6" />
<img width="838" height="937" alt="eda page 3" src="https://github.com/user-attachments/assets/43336149-3b84-4dc6-bb14-73003c8cb2e3" />
<img width="838" height="934" alt="eda page 4" src="https://github.com/user-attachments/assets/6c9a4062-fbc8-4e13-b38b-e2234372d4b6" />
<img width="1680" height="923" alt="eda page 5" src="https://github.com/user-attachments/assets/08670aa2-3284-45e7-8ac0-8ab00f3097c8" />
<img width="1680" height="924" alt="eda page 6" src="https://github.com/user-attachments/assets/188345a3-873d-47e7-8e0a-06b8f5c7cc45" />
**⚙️ Model Comparison Metrics**
<img width="838" height="935" alt="comparision page 1" src="https://github.com/user-attachments/assets/d2b8f464-10ea-4ae0-9779-9c23de0e9b2b" />
<img width="838" height="936" alt="comparision page 2" src="https://github.com/user-attachments/assets/9a1fa7f7-03df-4310-8ae6-b342fb2b482c" />
**🧮 Sales Prediction Form**
<img width="1680" height="926" alt="prediction page 1" src="https://github.com/user-attachments/assets/44156753-6d46-43cf-9d6b-949922534943" />
**📈 Prediction Results Visualization**
<img width="1680" height="928" alt="prediction page 2" src="https://github.com/user-attachments/assets/50853ace-145d-4005-889a-441f6d09fb82" />

---

## 🚀 How to Run

1. Clone the repo:  
   ```bash
   git clone https://github.com/G-Pujitha1102/Sales-Prediction.git
2.Navigate to project folder:
       bash
cd Sales-Prediction
3.Create and activate a virtual environment (optional but recommended):
    bash    
python -m venv venv
source venv/scripts/activate
4.Install dependencies:
           bash     
pip install -r requirements.txt
5.Run the Streamlit app:
        bash
streamlit run app.py  

---
   
## 📎 Dataset Source
👉 Sales Prediction (Simple Linear Regression) - Kaggle

---

## 🗃️ Repository Structure

📁 Sales-Prediction/
├── data/
│   └── advertising.csv
├── models/
│   ├── linear_regression_model.pkl
│   └── random_forest_model.pkl
├── outputs/
│   ├── metrics.csv
│   └── predictions.csv
├── app.py
├── requirements.txt
├── README.md

---

## 💡 Challenges Faced & What I Learned

-Preprocessing and feature selection to improve model accuracy.
-Comparing regression models using multiple metrics.
-Visualizing model performance using Plotly and Seaborn.
-Building an interactive Streamlit dashboard for end-user predictions.
-Managing project structure and saving models with Joblib.

---

## 🙋‍♀️ Author
G. Pujitha
🎓 B.Tech - Computer Science Engineering
GitHub Profile

---

## 🔮 Future Improvements

-Add more advanced models like Gradient Boosting and XGBoost.
-Deploy on other platforms like Heroku or AWS.
-Add user authentication for saving predictions.
-Enable batch predictions via CSV upload.
-Improve visualizations and add interactive filtering options.
