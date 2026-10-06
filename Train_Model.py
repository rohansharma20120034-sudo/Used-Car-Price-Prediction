import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).replace('₹', '').replace(',', '').strip()
    if 'Lakh' in price_str:
        return float(price_str.replace('Lakh', '').strip())
    elif 'Crore' in price_str:
        return float(price_str.replace('Crore', '').strip()) * 100
    else:
        return float(price_str)

def clean_kms(kms_str):
    if pd.isna(kms_str):
        return np.nan
    return int(str(kms_str).replace('km', '').replace(',', '').strip())

def run_car_price_prediction():
    print("1. Loading dataset...")
    try:
        df = pd.read_csv('used_car_dataset.csv')
    except FileNotFoundError:
        print("Error: 'used_car_dataset.csv' not found. Please place your dataset in the script's directory.")
        return

    df['car_price_in_rupees'] = df['car_price_in_rupees'].apply(clean_price)
    df['kms_driven'] = df['kms_driven'].apply(clean_kms)
    
    if 'car_name' in df.columns:
        df = df.drop(columns=['car_name'])

    print(f"Dataset Shape: {df.shape}")
    
    target_column = 'car_price_in_rupees'
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found. Columns available: {list(df.columns)}")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical Features: {numerical_cols}")
    print(f"Categorical Features: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("\n2. Training the Random Forest model pipeline...")
    pipeline.fit(X_train, y_train)

    print("\n3. Evaluating model performance...")
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% variance explained)")

    print("\n4. Saving evaluation plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7, edgecolors='k', label='Predictions')
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', linewidth=2, label='Ideal Fit')

    plt.xlabel('Actual Price (in Lakhs)', fontsize=12)
    plt.ylabel('Predicted Price (in Lakhs)', fontsize=12)
    plt.title('Actual vs. Predicted Used Car Prices (Random Forest)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('car_price_prediction.png')

if __name__ == "__main__":
    run_car_price_prediction()
