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
@st.cache_data
def load_data():
    return pd.read_csv("global_development_data.csv")

data = load_data()

with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write("This section provides a global perspective on quality of life indicators.")
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

with tab3:
    st.write("### 📂 Data Explorer")

    st.write("Explore the dataset of global development indicators:")
    

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
