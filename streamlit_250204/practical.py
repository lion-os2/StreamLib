import streamlit as st
import pandas as pd

# Page Layaut 
st.set_page_config(layout="wide")


# Title
st.title("🌍 Worldwide Analysis of Quality of Life and Economic Factors")

# Subtitle
st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

# Download Data 
def load_data():
    file_path = "global_development_data.csv"
    
    try:
        # محاولة تحميل البيانات من الملف المحلي
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("⚠ Local file not found, loading from GitHub...")
        # تحميل البيانات من GitHub
        file_url = "https://raw.githubusercontent.com/lion-os2/StreamLib/main/global_development_data.csv"
        data = pd.read_csv(file_url)
    
    return data

data = load_data()

with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write("This section provides a global perspective on quality of life indicators.")

    selected_year = st.slider("Select Year", int(data["year"].min()), int(data["year"].max()), int(data["year"].median()))

    # Filtering data by Year
    year_data = data[data["year"] == selected_year]

    # Calculating 
    mean_life_expectancy = year_data["Healthy Life Expectancy (IHME)"].mean()
    median_gdp_per_capita = year_data["GDP per capita"].median()
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

with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

with tab3:
    st.write("### 📂 Data Explorer")

    st.write("Explore the dataset of global development indicators:")
    
    # Display Dataset 
    st.dataframe(data)

    # Create multiselectbox to select the country names
    countries = st.multiselect("Select Countries", options=data["country"].unique())

    #Create slider to select the year range
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
