import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from plots import scatter_plot
import os
import urllib.request
import plotly.graph_objects as go
import plotly.figure_factory as ff


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

# ========== 🌍 1. Global Overview ==========
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Global Overview"

# Use radio buttons to control tab selection
selected_tab = st.radio(
    "Choose a Tab", 
    ["Global Overview", "Country Deep Dive", "Data Explorer"], 
    index=["Global Overview", "Country Deep Dive", "Data Explorer"].index(st.session_state.active_tab),
    key="tab_selector",
    horizontal=True
)

# Update session state when user selects a different tab
st.session_state.active_tab = selected_tab

if st.session_state.active_tab == "Global Overview":
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
        if model is not None:
            try:
                # التحقق من وجود بيانات قبل التنبؤ
                X_test = data[features].copy()

                if X_test.empty:
                    st.error("❌ No data available for prediction.")
                else:
                    y_pred = model.predict(X_test)

                    from sklearn.metrics import mean_squared_error
                    mse = mean_squared_error(data["Healthy Life Expectancy (IHME)"], y_pred)
                    st.metric(label="📉 Mean Model Squared Error", value=f"{mse:.2f}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        else:
            st.error("❌ Model could not be loaded. Please check if model.pkl exists.")

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

# ========== :2. bar_chart: Country Deep Dive ==========
elif st.session_state.active_tab == "Country Deep Dive":
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

    selected_country = st.selectbox(
        "🌍 Select a Country", 
        options=sorted(data["country"].unique()),
        key="country_deep_dive"
    )

    country_data = data[data["country"] == selected_country].groupby("year", as_index=False).agg({
        "Healthy Life Expectancy (IHME)": "mean",
        "GDP per capita": "mean"
    })

    if not country_data.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=country_data["year"],
            y=country_data["Healthy Life Expectancy (IHME)"],
            mode="lines+markers",
            name="Life Expectancy",
            line=dict(width=2, color="blue"),
        ))

        fig.add_trace(go.Scatter(
            x=country_data["year"],
            y=country_data["GDP per capita"],
            mode="lines+markers",
            name="GDP per Capita",
            line=dict(width=2, color="red"),
            yaxis="y2"
        ))

        fig.update_layout(
            title=f"Life Expectancy & GDP per Capita in {selected_country}",
            xaxis_title="Year",
            yaxis=dict(title="Life Expectancy (Years)", side="left"),
            yaxis2=dict(
                title="GDP per Capita",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(x=0.1, y=1.1)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠ No data available for {selected_country}.")

    # Slider to select a single year
    selected_year = st.slider("Select Year", int(data["year"].min()), int(data["year"].max()), int(data["year"].max()))

    # Filter data for the selected year
    yearly_data = data[data["year"] == selected_year]

    st.write(f"### Analysis for {selected_country} in {selected_year}")

    # Ensure the selected country has data for the selected year
    country_year_data = yearly_data[yearly_data["country"] == selected_country]

    if not country_year_data.empty:
        # Extract values for the selected country
        selected_gdp = country_year_data["GDP per capita"].values[0]
        selected_life_expectancy = country_year_data["Healthy Life Expectancy (IHME)"].values[0]

        # Display key statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📈 Life Expectancy", value=f"{selected_life_expectancy:.2f} years")
        with col2:
            st.metric(label="💰 GDP per Capita", value=f"${selected_gdp:,.2f}")

        # ========== Histogram for GDP per Capita ==========
        st.write(f"### GDP per Capita Distribution in {selected_year}")
        fig_gdp = go.Figure()

        fig_gdp.add_trace(go.Histogram(
            x=yearly_data["GDP per capita"],
            nbinsx=30,
            marker=dict(color="lightblue"),
            name="All Countries"
        ))

        # Add arrow marker for the selected country's GDP
        fig_gdp.add_trace(go.Scatter(
            x=[selected_gdp],
            y=[0],
            mode="markers+text",
            marker=dict(color="red", size=10, symbol="arrow-bar-up"),
            text=[selected_country],
            textposition="top center",
            name=f"{selected_country} GDP"
        ))

        fig_gdp.update_layout(
            title="GDP per Capita Distribution",
            xaxis_title="GDP per Capita",
            yaxis_title="Number of Countries",
            showlegend=True
        )
        st.plotly_chart(fig_gdp, use_container_width=True)

        # ========== Histogram for Life Expectancy ==========
        st.write(f"### Life Expectancy Distribution in {selected_year}")
        fig_life = go.Figure()

        fig_life.add_trace(go.Histogram(
            x=yearly_data["Healthy Life Expectancy (IHME)"],
            nbinsx=30,
            marker=dict(color="green"),
            name="All Countries"
        ))

        # Add arrow marker for the selected country's Life Expectancy
        fig_life.add_trace(go.Scatter(
            x=[selected_life_expectancy],
            y=[0],
            mode="markers+text",
            marker=dict(color="red", size=10, symbol="arrow-bar-up"),
            text=[selected_country],
            textposition="top center",
            name=f"{selected_country} Life Expectancy"
        ))

        fig_life.update_layout(
            title="Life Expectancy Distribution",
            xaxis_title="Life Expectancy (Years)",
            yaxis_title="Number of Countries",
            showlegend=True
        )
        st.plotly_chart(fig_life, use_container_width=True)
        
        # ========== Histogram for Poverty Headcount Ratio ==========
        st.write(f"### Poverty Headcount Ratio Distribution in {selected_year}")

        # Get Poverty Headcount Ratio of the selected country
        selected_poverty = country_year_data["poverty_gap_index_100"].values[0]

        # Create Histogram for Poverty Headcount Ratio
        fig_poverty = go.Figure()

        fig_poverty.add_trace(go.Histogram(
            x=yearly_data["poverty_gap_index_100"],
            nbinsx=30,
            marker=dict(color="lightcoral"),
            name="All Countries"
        ))

        # Add arrow marker for the selected country's Poverty Headcount Ratio
        fig_poverty.add_trace(go.Scatter(
            x=[selected_poverty],
            y=[0],
            mode="markers+text",
            marker=dict(color="red", size=10, symbol="arrow-bar-up"),
            text=[selected_country],
            textposition="top center",
            name=f"{selected_country} Poverty Ratio"
        ))

        fig_poverty.update_layout(
            title="Poverty Headcount Ratio Distribution",
            xaxis_title="Poverty Headcount Ratio",
            yaxis_title="Number of Countries",
            showlegend=True
        )

        st.plotly_chart(fig_poverty, use_container_width=True)

    else:
        st.warning(f"⚠ No data available for {selected_country} in {selected_year}.")


# ========== 📂 3. Data Explorer ==========
elif st.session_state.active_tab == "Data Explorer":
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
