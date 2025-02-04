import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from plots import scatter_plot
import os
import urllib.request


# Page Layout 
st.set_page_config(layout="wide")

# Title
st.title("🌍 Worldwide Analysis of Quality of Life and Economic Factors")

# Subtitle
st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)

# Load The model
@st.cache_resource
def load_model():
    model_path = "model.pkl"

    # إذا لم يكن الملف موجودًا محليًا، قم بتحميله من GitHub
    if not os.path.exists(model_path):
        st.warning("⚠ Model file not found locally. Downloading from GitHub...")
        github_url = "https://raw.githubusercontent.com/lion-os2/StreamLib/main/model.pkl"
        try:
            urllib.request.urlretrieve(github_url, model_path)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


model = load_model()

# Load Data 
def load_data():
    file_path = "global_development_data.csv"
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("⚠ Local file not found, loading from GitHub...")
        file_url = "https://raw.githubusercontent.com/lion-os2/StreamLib/main/global_development_data.csv"
        data = pd.read_csv(file_url)
    
    return data

data = load_data()

# ========== 🌍 Global Overview ==========
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write("This section provides a global perspective on quality of life indicators.")

    selected_year = st.slider("Select Year", int(data["year"].min()), int(data["year"].max()))

    # Filtering data by Year
    year_data = data[data["year"] == selected_year]

    # Calculating 
    mean_life_expectancy = year_data["Healthy Life Expectancy (IHME)"].mean()
    median_gdp_per_capita = year_data["GDP per capita"].median()
    if year_data["headcount_ratio_upper_mid_income_povline"].max() > 1:
        mean_poverty_ratio = year_data["headcount_ratio_upper_mid_income_povline"].mean() / 100
    else:
        mean_poverty_ratio = year_data["headcount_ratio_upper_mid_income_povline"].mean()
    num_countries = year_data["country"].nunique()

    # Displaying Data
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📈 Mean Life Expectancy", value=f"{mean_life_expectancy:.2f} years")

    with col2:
        st.metric(label="💰 Median GDP per Capita", value=f"${median_gdp_per_capita:,.2f}")

    with col3:
        st.metric(label="📊 Mean Poverty Ratio", value=f"{mean_poverty_ratio:.2%}")  

    with col4:
        st.metric(label="🌍 Number of Countries", value=num_countries)

    # Create the Figure and display it
    st.write("### 📉 GDP per Capita vs Life Expectancy")
    fig = scatter_plot(year_data)  
    st.plotly_chart(fig, use_container_width=True)

    # ========== 🔮 Prediction Model ==========
    st.write("## Predict Life Expectancy")

    col_left, col_right = st.columns([1, 1])

    with col_right:
        st.write("### Input Features for Prediction")
        
        gdp = st.number_input(
            "GDP per Capita", 
            min_value=int(data["GDP per capita"].min()), 
            max_value=int(data["GDP per capita"].max()), 
            value=int(data["GDP per capita"].median()), 
            step=1000
        )

        poverty_ratio = st.number_input(
            "Poverty Ratio", 
            min_value=float(data["headcount_ratio_upper_mid_income_povline"].min()), 
            max_value=float(data["headcount_ratio_upper_mid_income_povline"].max()), 
            value=float(data["headcount_ratio_upper_mid_income_povline"].median()), 
            step=0.1
        )

        year = st.number_input(
            "Year", 
            min_value=int(data["year"].min()), 
            max_value=int(data["year"].max()), 
            value=int(data["year"].median()), 
            step=1
        )

        # Predict
        if st.button("Predict"):
            input_data = pd.DataFrame([[gdp, poverty_ratio, year]], columns=['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year'])
            prediction = model.predict(input_data)[0]

            st.success(f"Estimated Life Expectancy: **{prediction:.2f} years**")

    with col_left:
        # st.write("### Model Performance")

        # 🔹  Mean Squared Error (MSE) 
        from sklearn.metrics import mean_squared_error

        features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
        X_test = data[features]
        y_test = data["Healthy Life Expectancy (IHME)"]
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        st.metric(label="Model Performance", value=f"{mse:.2f}")

        # Show the importance of the Features
        # st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            "Feature": ['GDP per capita', 'Poverty Ratio', 'Year'],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            feature_importance, 
            x="Importance", 
            y="Feature", 
            orientation="h", 
            title="Feature Importance",
            color="Feature"
        )
        st.plotly_chart(fig)

# ========== :bar_chart: Country Deep Dive ==========
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

# ========== 📂 Data Explorer ==========
with tab3:
    st.write("### 📂 Data Explorer")
    st.write("Explore the dataset of global development indicators:")
    
    # Display Dataset 
    st.dataframe(data)

    # Create multiselectbox to select the country names
    countries = st.multiselect("Select Countries", options=data["country"].unique())

    # Create slider to select the year range
    min_year, max_year = int(data["year"].min()), int(data["year"].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    # Create filtered dataset
    filtered_data = data[
        (data["year"] >= year_range[0]) & (data["year"] <= year_range[1])
    ]
    if countries:
        filtered_data = filtered_data[filtered_data["country"].isin(countries)]
    
    # Display Filtered Data 
    st.write("### Filtered Data")
    st.dataframe(filtered_data)

    # Download Data 
    csv = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_global_data.csv",
        mime="text/csv",
    )
