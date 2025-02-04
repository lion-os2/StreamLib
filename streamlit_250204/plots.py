import plotly.express as px

# px.defaults.template = None  

def scatter_plot(data):
    """
    Creates a scatter plot of GDP per capita vs Life Expectancy.

    Args:
        data (DataFrame): Filtered dataset containing GDP and Life Expectancy.
    
    Returns:
        fig (plotly Figure): Scatter plot figure.
    """
    fig = px.scatter(
        data,
        x="GDP per capita",
        y="Healthy Life Expectancy (IHME)",
        size="GDP per capita", 
        color="Healthy Life Expectancy (IHME)",  
        hover_name="country",
        log_x=True,
        title="GDP per Capita vs Healthy Life Expectancy",
        labels={"GDP per capita": "GDP per Capita (log scale)", "Life Expectancy": "Life Expectancy (years)"},
    )

    fig.update_layout(
        template="seaborn",
        autosize=True,
        margin=dict(l=100, r=100, t=50, b=100),  
        width=900,  
        height=600,  
    )

    return fig  
