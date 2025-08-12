# 📊 Sales Prediction App

This project predicts sales based on advertising budgets across TV, Radio, and Newspaper using Machine Learning models. It analyzes the dataset, visualizes insights, builds and compares Linear Regression and Random Forest models, and provides an interactive web app for sales prediction.

---

## 🎯 Objective  
To predict sales using advertising budgets as input features and identify which model performs better to help optimize marketing strategies.

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

## 📊 Bar Chart Comparison

Bar charts were created using Plotly to visually compare RMSE, MAE, and R² values for both models.



## 📊 Sample Visualizations

*Replace these with your own app screenshots or GIFs.*

- Sales Distribution Histogram <img width="1402" height="577" alt="histograms" src="https://github.com/user-attachments/assets/1e4c6c4e-6f39-4d60-8ba9-7669763d2a1d" />

- Correlation Heatmap <img width="1326" height="694" alt="correlation heatmap and plot" src="https://github.com/user-attachments/assets/b8164a10-28a7-4892-a0ae-7312b020d650" />

- Actual vs Predicted Sales Line Chart <img width="668" height="527" alt="actual vs predicted sales" src="https://github.com/user-attachments/assets/b62032a7-3ebb-4eb0-831d-f47dea99b3d8" />

- Model Prediction Comparison Bar Chart
 <img width="556" height="497" alt="bar chart R2" src="https://github.com/user-attachments/assets/d575df59-6007-48e4-b9c7-a89c9804d403" />
<img width="552" height="500" alt="bar chart RMSE" src="https://github.com/user-attachments/assets/e6c7f46f-96ed-43e6-9cf8-11185d0fb781" />
<img width="558" height="512" alt="bar chart MAE" src="https://github.com/user-attachments/assets/ff0cfa28-3b31-4a13-95d5-7ef73d452206" />

- Scatter 
Plot: news paper vs Sales<img width="701" height="678" alt="customscatterplot 1" src="https://github.com/user-attachments/assets/44a3414f-55ea-4c78-9017-bf352ad54e76" />

-  

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
   
📎 Dataset Source
👉 Sales Prediction (Simple Linear Regression) - Kaggle

🗃️ Repository Structure

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

💡 Challenges Faced & What I Learned

-Preprocessing and feature selection to improve model accuracy.
-Comparing regression models using multiple metrics.
-Visualizing model performance using Plotly and Seaborn.
-Building an interactive Streamlit dashboard for end-user predictions.
-Managing project structure and saving models with Joblib.


🔗 Live Demo
Try the app online:
👉 Sales Prediction Live Demo

🙋‍♀️ Author
G. Pujitha
🎓 B.Tech - Computer Science Engineering
GitHub Profile

🔮 Future Improvements

-Add more advanced models like Gradient Boosting and XGBoost.
-Deploy on other platforms like Heroku or AWS.
-Add user authentication for saving predictions.
-Enable batch predictions via CSV upload.
-Improve visualizations and add interactive filtering options.
