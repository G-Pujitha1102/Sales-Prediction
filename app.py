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
menu = st.sidebar.radio("Go to", ["EDA", "Prediction", "Comparison Metrics"])

# -----------------------
#EDA PAGE
# -----------------------
if menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    # Global CSS to limit max width and center content + tab styling
    st.markdown("""
    <style>
    /* Container width and centering */
    .main > div.block-container {
        max-width: 900px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 15px !important;
        padding-right: 15px !important;
    }

       /* Tab buttons - general */
    button[role="tab"] {
        background-color: #f0f4f8 !important; /* Light gray default */
        color: #334155 !important;             /* Dark text */
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
        margin-right: 4px !important;
        transition: background-color 0.3s ease, color 0.3s ease !important;
    }

    /* Active tab */
    button[role="tab"][aria-selected="true"] {
        background-color: #1e3a8a !important; /* Deep navy blue */
        color: #dbeafe !important;            /* Light blue text */
        font-weight: 700 !important;
        border-bottom: 3px solid #3b82f6 !important; /* Bright blue underline */
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.4) !important;
    }

    /* Hover effect for inactive tabs */
    button[role="tab"]:not([aria-selected="true"]):hover {
        background-color: #3b82f6 !important; /* Bright blue */
        color: white !important;
        cursor: pointer !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Tabs for EDA sections
    tabs = st.tabs([
        "Summary & Data",
        "Correlation & Pairplot",
        "Distributions",
        "Boxplots",
        "Custom Scatter",
        "Actual vs Predicted"
    ])

    # Tab 1: Summary & Dataset with download
    with tabs[0]:
        with st.container():
            st.subheader("📄 Dataset Overview")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download Dataset CSV", data=csv, file_name="advertising_data.csv", mime="text/csv")

            st.subheader("📈 Summary Statistics")
            with st.expander("Show Summary Statistics"):
                st.write(df.describe())

           

    # Tab 2: Correlation Heatmap & Pairplot side by side
    with tabs[1]:
        with st.container():
            st.subheader("🔗 Correlation Heatmap & Pair Plot")
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

    # Tab 3: Histograms side by side (2 per row)
    with tabs[2]:
        with st.container():
            st.subheader("📉 Histograms")
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
            sns.set_palette("pastel")
            for i in range(0, len(numeric_cols), 2):
                cols = st.columns(2)
                for j, col_name in enumerate(numeric_cols[i:i+2]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.histplot(df[col_name], kde=True, ax=ax)
                        ax.set_title(f"Distribution of {col_name}", fontsize=14, fontweight='bold')
                        ax.set_xlabel(col_name)
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)

    # Tab 4: Boxplots side by side (2 per row)
    with tabs[3]:
        with st.container():
            st.subheader("📦 Boxplots for Numerical Features")
            for i in range(0, len(numeric_cols), 2):
                cols = st.columns(2)
                for j, col_name in enumerate(numeric_cols[i:i+2]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.boxplot(x=df[col_name], ax=ax, color="#a3cee2")
                        ax.set_title(f"Boxplot of {col_name}", fontsize=14, fontweight='bold')
                        st.pyplot(fig)

    # Tab 5: Custom Scatter Plot with selections
    with tabs[4]:
        with st.container():
            st.subheader("📌 Custom Scatter Plot")
            columns = list(df.columns)
            x_axis = st.selectbox("Select X-axis", columns, index=0)
            y_axis = st.selectbox("Select Y-axis", columns, index=1)
            fig = px.scatter(df, x=x_axis, y=y_axis, size=y_axis, color=y_axis,
                             title=f"Scatter plot of {x_axis} vs {y_axis}",
                             width=700, height=500,
                             color_continuous_scale=px.colors.sequential.Blues  # Changed here
            )
            fig.update_layout(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                title_font_size=18,
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=False)

    # Tab 6: Actual vs Predicted Sales Line Chart
    with tabs[5]:
        with st.container():
            st.subheader("📊 Actual vs Predicted Sales (Line Chart)")
            fig = px.line(
                comparison_df,
                y=["Actual_Sales", "LR_Prediction", "RF_Prediction"],
                title="Actual vs Predicted Sales Over Records",
                width=700,
                height=500
            )
            fig.update_layout(
                xaxis_title="Record Index",
                yaxis_title="Sales",
                title_font_size=18,
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=False)

            # Optional insight box
        st.markdown(
            """
            <div style='
                margin-top: 20px;
                padding: 12px 15px;
                background-color: #1e3a8a;
                color: #dbeafe;
                border-left: 6px solid #3b82f6;
                border-radius: 8px;
                font-weight: 600;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 2px 6px rgba(59, 130, 246, 0.4);
            '>
                Insight: The line chart shows how closely each model's predictions align with the actual sales. Both models perform well, with Random Forest generally fitting better on some records.
            </div>
            """,
            unsafe_allow_html=True
        )
# -----------------------
#  PREDICTION PAGE
# -----------------------
elif menu == "Prediction":
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
