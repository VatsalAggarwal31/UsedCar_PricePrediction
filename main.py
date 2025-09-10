import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import re


# Dataset
try:
    df = pd.read_csv("used_cars_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'used_cars_data.csv' not found. Please ensure the file is in the same directory.")
    exit()