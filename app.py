import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# -----------------------
# Page Config 
# -----------------------

st.set_page_config(page_title="Sales Prediction App", layout="wide")

# -----------------------
# Fixed Header CSS
# -----------------------
st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
        top: 30px;
        left: 0;
        width: 100%;
        background-color: #0E1117;
        padding: 30px 0;
        text-align: center;
        border-bottom: 1px solid #444;
        z-index: 999;
        height: 120px;
    }
    .fixed-header h1 {
        color: white;
        font-size: 50px;
        margin: 0;
    }
    .block-container {
        padding-top: 140px !important;
    }
    @media screen and (max-width: 1024px) {
        .fixed-header { padding: 20px 0; height: 100px; }
        .fixed-header h1 { font-size: 35px; }
        .block-container { padding-top: 110px !important; }
    }
    @media screen and (max-width: 600px) {
        .fixed-header { padding: 15px 5px; height: auto; }
        .fixed-header h1 { font-size: 26px; line-height: 1.2; }
        .block-container { padding-top: 90px !important; }
    }
    </style>
    <div class="fixed-header">
        <h1>📊 Sales Prediction</h1>
    </div>     
""", unsafe_allow_html=True)

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/advertising.csv")

df = load_data()

# -----------------------
# Train Models 
# -----------------------
@st.cache_resource
def train_models(X_train, y_train):
    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    return lr_model, rf_model

# Prepare features & target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split into train/test (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on training set
lr_model, rf_model = train_models(X_train, y_train)

# Predictions on test set
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Create comparison DataFrame (only test data)
comparison_df = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "LR_Prediction": lr_pred,
    "RF_Prediction": rf_pred
})

# Save outputs
comparison_df.to_csv("outputs/predictions.csv", index=False)
joblib.dump(lr_model, "models/linear_regression_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")

# -----------------------
# Metrics calculation
# -----------------------
def calculate_metrics(y_true, y_pred):
    return {
        "R² Score": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

lr_metrics = calculate_metrics(y_test, lr_pred)
rf_metrics = calculate_metrics(y_test, rf_pred)
# Save metrics to CSV (based on test set)
metrics_df = pd.DataFrame([
    {"Metric": m, "Linear Regression": lr_metrics[m], "Random Forest": rf_metrics[m]}
    for m in lr_metrics.keys()
])
metrics_df.to_csv("outputs/metrics.csv", index=False)
print("✅ Metrics saved to metrics.csv (Test set only)")


# -----------------------
# Sidebar Navigation
# -----------------------
menu = st.sidebar.radio("Go to", ["Prediction", "Comparison Metrics", "EDA"])

# -----------------------
# PREDICTION PAGE
# -----------------------
if menu == "Prediction":
    st.header("📈 Prediction")

    # Metrics Table
    metrics_df = pd.DataFrame([
        {"Metric": m, "Linear Regression": lr_metrics[m], "Random Forest": rf_metrics[m]}
        for m in lr_metrics.keys()
    ])
    st.table(metrics_df)

    # Best model info
    lr_r2 = lr_metrics["R² Score"]
    rf_r2 = rf_metrics["R² Score"]
    if lr_r2 > rf_r2:
        st.info("✅ Linear Regression is performing more accurately based on R² score.")
    elif rf_r2 > lr_r2:
        st.info("✅ Random Forest is performing more accurately based on R² score.")
    else:
        st.info("ℹ️ Both models have equal accuracy based on R² score.")

    # Prediction form
    st.subheader("🔮 Make a Prediction")
    with st.form("prediction_form"):
        tv = st.number_input("TV Advertising Budget", min_value=0.0, value=100.0)
        radio = st.number_input("Radio Advertising Budget", min_value=0.0, value=25.0)
        newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, value=10.0)
        submit = st.form_submit_button("Predict Sales")

    if submit:
        lr_result = lr_model.predict([[tv, radio, newspaper]])[0]
        rf_result = rf_model.predict([[tv, radio, newspaper]])[0]

        colA, colB = st.columns(2)
        with colA:
            st.success(f"📈 Linear Regression: **{lr_result:.2f}**")
        with colB:
            st.success(f"🌲 Random Forest: **{rf_result:.2f}**")

        pred_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Predicted Sales": [lr_result, rf_result]
        })
        st.subheader("📊 Predicted Sales Comparison")
        fig = px.bar(pred_df, x="Model", y="Predicted Sales", text="Predicted Sales", color="Model",
                     width=600, height=400)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=False)

        avg_pred = (lr_result + rf_result) / 2
        if avg_pred < 10:
            st.warning("📉 Predicted Sales are **LOW**.")
        elif 10 <= avg_pred <= 20:
            st.info("📊 Predicted Sales are **AVERAGE**.")
        else:
            st.success("🚀 Predicted Sales are **HIGH**!")

# -----------------------
# COMPARISON METRICS PAGE
# -----------------------
elif menu == "Comparison Metrics":
    st.header("📊 Model Comparison")

    metrics_df = pd.DataFrame({
        "Metric": ["R² Score", "RMSE", "MAE"],
        "Linear Regression": [
            r2_score(y_test, lr_pred),
            np.sqrt(mean_squared_error(y_test, lr_pred)),
            mean_absolute_error(y_test, lr_pred)
        ],
        "Random Forest": [
            r2_score(y_test, rf_pred),
            np.sqrt(mean_squared_error(y_test, rf_pred)),
            mean_absolute_error(y_test, rf_pred)
        ]
    })

    st.dataframe(metrics_df.style.format({
        "Linear Regression": "{:.4f}",
        "Random Forest": "{:.4f}"
    }))

    for metric in metrics_df["Metric"]:
        df_metric = metrics_df[metrics_df["Metric"] == metric][["Linear Regression", "Random Forest"]]
        chart_df = df_metric.melt(var_name="Model", value_name="Score")
        fig = px.bar(chart_df, x="Model", y="Score", color="Model", text="Score",
                     title=f"{metric} Comparison", width=600, height=500)
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(yaxis_title=metric, xaxis_title="Model")
        st.plotly_chart(fig, use_container_width=False)

# -----------------------
# EDA PAGE
# -----------------------
elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    tabs = st.tabs([
        "Summary & Data",
        "Correlation & Pairplot",
        "Distributions",
        "Boxplots",
        "Custom Scatter",
        "Actual vs Predicted"
    ])

    with tabs[0]:
        st.subheader("📄 Dataset Overview")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Dataset CSV", data=csv, file_name="advertising_data.csv", mime="text/csv")
        st.subheader("📈 Summary Statistics")
        with st.expander("Show Summary Statistics"):
            st.write(df.describe())

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        with col2:
            sample_df = df.sample(min(50, len(df)), random_state=42)
            pairplot_fig = sns.pairplot(sample_df, diag_kind="kde", palette="pastel")
            st.pyplot(pairplot_fig.fig)

    with tabs[2]:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(numeric_cols[i:i+2]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.histplot(df[col_name], kde=True, ax=ax)
                    ax.set_title(f"Distribution of {col_name}", fontsize=14, fontweight='bold')
                    st.pyplot(fig)

    with tabs[3]:
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(numeric_cols[i:i+2]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.boxplot(x=df[col_name], ax=ax, color="#a3cee2")
                    ax.set_title(f"Boxplot of {col_name}", fontsize=14, fontweight='bold')
                    st.pyplot(fig)

    with tabs[4]:
        columns = list(df.columns)
        x_axis = st.selectbox("Select X-axis", columns, index=0)
        y_axis = st.selectbox("Select Y-axis", columns, index=1)
        fig = px.scatter(df, x=x_axis, y=y_axis, size=y_axis, color=y_axis,
                         title=f"Scatter plot of {x_axis} vs {y_axis}",
                         width=700, height=500,
                         color_continuous_scale=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=False)

    with tabs[5]:
        fig = px.line(comparison_df,
                      y=["Actual_Sales", "LR_Prediction", "RF_Prediction"],
                      title="Actual vs Predicted Sales (Test Set)",
                      width=700, height=500)
        fig.update_layout(xaxis_title="Record Index", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=False)
