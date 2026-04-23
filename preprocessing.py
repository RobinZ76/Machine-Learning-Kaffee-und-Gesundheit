import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# load the dataset
df = pd.read_csv('CoffeeAndHealth.csv', keep_default_na=False)

# Ordinal Encoding
# Map Stress_Level to numerical values
stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Stress_Level'] = df['Stress_Level'].map(stress_mapping)

# Map Health_Issues to numerical values
health_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
df['Health_Issues'] = df['Health_Issues'].map(health_mapping)

# Map Sleep_Quality to numerical values
quality_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
df['Sleep_Quality'] = df['Sleep_Quality'].map(quality_mapping)

# One-Hot Encoding
# One-Hot encode the Gender
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Feature Creation and Transformation
df['Caffeine_per_Age'] = df['Caffeine_mg'] / (df['Age'] + 1)
df['Caffeine_Stress_Interaction'] = df['Caffeine_mg'] * df['Stress_Level']
df['Age_Activity_Interaction'] = df['Age'] * df['Physical_Activity_Hours']
df['Log_Caffeine'] = np.log1p(df['Caffeine_mg'])

# Drop 
cols_to_drop = ['ID', 'Country', 'Occupation', 'Coffee_Intake','Caffeine_mg']
df_reduced = df.drop(columns=cols_to_drop)

# Data Splitting
X = df_reduced.drop('Sleep_Hours', axis=1)
y = df_reduced['Sleep_Hours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler = RobustScaler() # extreme samples in Coffeine_mg and Sleep_Hours
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Preprocessing complete. {len(X.columns)} features were used")

# Save the preprocessed data and scaler for later use
data_to_save = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': X.columns.tolist() 
}

# check for missing values in the preprocessed data
print("\nMissing values in each column:")
print(X.isnull().sum())

joblib.dump(data_to_save, 'preprocessed_data.pkl')
print("\nPreprocessed data and scaler saved successfully.")