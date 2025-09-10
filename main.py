import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



# Dataset
try:
    df = pd.read_csv("used_cars_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'used_cars_data.csv' not found. Please ensure the file is in the same directory.")
    exit()

cars=df.copy()
print(f'There are {cars.shape[0]} rows and {cars.shape[1]} columns')

print ("Rows     : " , cars.shape[0])
print ("Columns  : " , cars.shape[1])
print ("#"*40,"\n","Features : \n\n", cars.columns.tolist())
print ("#"*40,"\nMissing values :\n\n", cars.isnull().sum().sort_values(ascending=False))
print( "#"*40,"\nPercent of missing :\n\n", round(cars.isna().sum() / cars.isna().count() * 100, 2))
print ("#"*40,"\nUnique values :  \n\n", cars.nunique())


# Data Preprocessing
df.drop(["S.No.", "New_Price"], axis=1, inplace=True)
df.dropna(inplace=True)
print(f"\nShape of data after dropping missing values: {df.shape}")

def clean_numeric_column(series):

    cleaned_series = series.astype(str)
    cleaned_series = cleaned_series.str.replace(r'[^0-9.]', '', regex=True)
    return pd.to_numeric(cleaned_series, errors='coerce')

df['Mileage'] = clean_numeric_column(df['Mileage'])
df['Engine'] = clean_numeric_column(df['Engine'])
df['Power'] = clean_numeric_column(df['Power'])

df.dropna(subset=['Mileage', 'Engine', 'Power'], inplace=True)

current_year = 2025
df['car_age'] = current_year - df['Year']
df.drop('Year', axis=1, inplace=True)

df['Brand'] = df['Name'].apply(lambda x: x.split(" ")[0])
df.drop('Name', axis=1, inplace=True)


# Feature Extraction
numerical_features = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'car_age']
categorical_features = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)

X = df.drop('Price', axis=1)
y = df['Price']

X_encoded = preprocessor.fit_transform(X)


# Split Data into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully.")

#Prediction
y_pred = model.predict(X_test)

# Evaluation of the model using the R-squared score and MSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel R-squared score: {r2:.4f}")
print(f'Mean Squared Error (MSE): {mse:.2f}')

#Visualization
# --- Visualization of Model Performance ---

y_train_pred = model.predict(X_train)


plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")


plt.scatter(y_train, y_train_pred, color='blue', label='Training Data', alpha=0.6)
plt.scatter(y_test, y_pred, color='red', label='Testing Data', alpha=0.8)


plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')


plt.title('Actual vs. Predicted Used Car Prices', fontsize=16)
plt.xlabel('Actual Price (in Lakhs)', fontsize=12)
plt.ylabel('Predicted Price (in Lakhs)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('performance_plot.png')
print("Graph saved to performance_plot.png")