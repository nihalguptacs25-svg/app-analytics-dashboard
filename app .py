
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="App Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df.drop_duplicates(subset="App")
    df = df[df["Category"] != "1.9"]

    df = df.dropna(subset=["Category", "Type"])
    df["Category"] = df["Category"].astype(str)
    df["Type"] = df["Type"].astype(str)

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    df["Installs_Num"] = (
        df["Installs"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Installs_Num"] = pd.to_numeric(df["Installs_Num"], errors="coerce")

    df["Price_Num"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
    )
    df["Price_Num"] = pd.to_numeric(df["Price_Num"], errors="coerce")

    df = df.dropna(subset=["Rating", "Reviews", "Installs_Num", "Price_Num"])
    df["Highly_Rated"] = (df["Rating"] >= 4.5).astype(int)
    return df

@st.cache_resource
def train_model(df):
    features = ["Reviews", "Installs_Num", "Price_Num", "Category", "Type"]
    X = df[features]
    y = df["Highly_Rated"]

    numeric_features = ["Reviews", "Installs_Num", "Price_Num"]
    categorical_features = ["Category", "Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm, len(X_train)

df = load_and_clean_data()
model, accuracy, cm, train_size = train_model(df)

st.title("📊 App Analytics Dashboard")
st.write(
    "Explore Google Play Store apps, ratings, "
    "installs, pricing and machine learning predictions."
)
st.divider()

st.sidebar.header("🔎 Filters")

categories = sorted(df["Category"].unique())
selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)
selected_type = st.sidebar.selectbox("Select Type", ["All", "Free", "Paid"])

filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

if selected_type != "All":
    filtered_df = filtered_df[filtered_df["Type"] == selected_type]

total_apps = len(filtered_df)
average_rating = filtered_df["Rating"].mean() if total_apps > 0 else 0
total_installs = filtered_df["Installs_Num"].sum() if total_apps > 0 else 0
free_apps = len(filtered_df[filtered_df["Type"] == "Free"])
paid_apps = len(filtered_df[filtered_df["Type"] == "Paid"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📱 Total Apps", f"{total_apps:,}")
with col2:
    st.metric("⭐ Average Rating", f"{average_rating:.2f}")
with col3:
    st.metric("⬇️ Total Installs", f"{total_installs:,.0f}")
with col4:
    st.metric("💰 Free / Paid", f"{free_apps:,} / {paid_apps:,}")

st.divider()

st.subheader("⭐ Average Rating by Category")

if total_apps > 0:
    rating_category = (
        filtered_df.groupby("Category")["Rating"]
        .mean()
        .reset_index()
        .sort_values("Rating", ascending=False)
    )

    fig1 = px.bar(
        rating_category,
        x="Category",
        y="Rating",
        title="Average Rating by Category",
        labels={"Rating": "Average Rating", "Category": "Category"},
        color="Rating",
        color_continuous_scale="Viridis"
    )
    fig1.update_yaxes(range=[0, 5])
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No data available for the selected filters.")

st.subheader("💰 Free vs Paid Apps")

if total_apps > 0:
    type_count = filtered_df["Type"].value_counts().reset_index()
    type_count.columns = ["Type", "Count"]

    fig2 = px.pie(
        type_count,
        names="Type",
        values="Count",
        title="Free vs Paid Apps",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("⬇️ Installs by Category")

if total_apps > 0:
    install_category = (
        filtered_df.groupby("Category")["Installs_Num"]
        .sum()
        .reset_index()
        .sort_values("Installs_Num", ascending=False)
    )

    fig3 = px.bar(
        install_category,
        x="Category",
        y="Installs_Num",
        title="Total Installs by Category",
        labels={"Installs_Num": "Installs", "Category": "Category"},
        color="Installs_Num",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("⭐ Rating vs Installs")

if total_apps > 0:
    scatter_df = filtered_df[
        (filtered_df["Rating"] > 0) &
        (filtered_df["Installs_Num"] > 0)
    ].copy()

    fig4 = px.scatter(
        scatter_df,
        x="Installs_Num",
        y="Rating",
        hover_name="App",
        color="Type",
        size="Reviews",
        title="Rating vs Number of Installs",
        labels={"Installs_Num": "Installs", "Rating": "Rating"},
        log_x=True
    )
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("🏆 Highly Rated Apps")

high_rated = filtered_df[filtered_df["Rating"] >= 4.5].copy()
high_rated = high_rated.sort_values(
    ["Rating", "Reviews"],
    ascending=[False, False]
)

display_columns = ["App", "Category", "Rating", "Reviews", "Installs", "Type", "Price"]

if len(high_rated) > 0:
    st.dataframe(
        high_rated[display_columns].head(20),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No highly rated apps found for the selected filters.")

st.divider()
st.header("🤖 Logistic Regression Model")
st.write(
    "The model predicts whether an app is "
    "**Highly Rated** (rating ≥ 4.5) using "
    "reviews, installs, price, category and type."
)

m1, m2 = st.columns(2)
with m1:
    st.metric("🤖 Model Accuracy", f"{accuracy * 100:.2f}%")
with m2:
    st.metric("📚 Training Samples", f"{train_size:,}")

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not Highly Rated", "Actual: Highly Rated"],
    columns=["Predicted: Not Highly Rated", "Predicted: Highly Rated"]
)
st.dataframe(cm_df, use_container_width=True)

st.subheader("🔮 Predict Whether an App Will Be Highly Rated")

col1, col2 = st.columns(2)

with col1:
    prediction_category = st.selectbox(
        "Category",
        sorted(df["Category"].unique()),
        key="prediction_category"
    )
    prediction_type = st.selectbox(
        "Type",
        ["Free", "Paid"],
        key="prediction_type"
    )
    prediction_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=1000
    )

with col2:
    prediction_installs = st.number_input(
        "Number of Installs",
        min_value=0,
        value=100000
    )
    prediction_price = st.number_input(
        "Price ($)",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

if st.button("🔮 Predict", type="primary"):
    input_data = pd.DataFrame({
        "Reviews": [prediction_reviews],
        "Installs_Num": [prediction_installs],
        "Price_Num": [prediction_price],
        "Category": [prediction_category],
        "Type": [prediction_type]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"⭐ **Prediction: Highly Rated**\n\n"
            f"Probability: **{probability * 100:.2f}%**"
        )
    else:
        st.warning(
            f"**Prediction: Not Highly Rated**\n\n"
            f"Probability of being highly rated: **{probability * 100:.2f}%**"
        )

st.divider()
st.caption(
    "Dataset: Google Play Store | "
    "Machine Learning: Logistic Regression"
)
