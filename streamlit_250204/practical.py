import streamlit as st 

# st.write("Hello!")
# st.write("Hello again!")
# st.write("Hello Hello!")

st.set_page_config(layout="wide")

st.title("🌍 Worldwide Analysis of Quality of Life and Economic Factors")

st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)

tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

with tab1:
    st.write("### 🌎 Global Overview")
    st.write("This section provides a global perspective on quality of life indicators.")

with tab2:
    st.write("### 📊 Country Deep Dive")
    st.write("Analyze specific countries in detail.")

with tab3:
    st.write("### 📂 Data Explorer")
    st.write("Explore raw data and trends over time.")
