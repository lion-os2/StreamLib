import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model(data_path="global_development_data.csv"):
    """
        Train the Model RandomForestRegressor
    """
    # Upload Data
    data = pd.read_csv(data_path)

    # Specify the features and target
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    target = 'Healthy Life Expectancy (IHME)'

    X = data[features]
    y = data[target]

    # Train the model 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the model in PKL File 
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved as model.pkl")

# Run the train 
if __name__ == "__main__":
    train_and_save_model()
