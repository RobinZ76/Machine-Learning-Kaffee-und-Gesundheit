import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

data = joblib.load('preprocessed_data.pkl')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

# Model Initialization
models = {
    # 1. Linear Regression
    'Linear Regression': LinearRegression(),
    # 2. Support Vector Regression
    'SVR': SVR(),
    # 3. Random Forest Regressor
    'Random Forest Regressor': RandomForestRegressor(random_state=42)
}

# Hyperparameter Definitions
param_grids = {
    'Linear Regression': {}, # empty
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    },
    'Random Forest Regressor': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
}

best_estimators = {}

# Model Training and Hyperparameter Tuning
for name, model in models.items():
    print(f"Processing {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_estimators[name] = grid.best_estimator_
    print(f"{name} Best Parameter: {grid.best_params_}")

summary = []
for name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    summary.append({'Model': name, 'RMSE': rmse, 'MSE': mse, 'R2': r2})

# Model Performance Comparison
df_results = pd.DataFrame(summary)
print("\n" + "="*40)
print("Model Performance Results (Look here!):")
print("="*40)
print(df_results.sort_values(by='R2', ascending=False))
print("="*40 + "\n")
df_results.to_csv('model_results_table.csv', index=False)

# Visualization of the best model's predictions
best_model_info = df_results.sort_values(by='R2', ascending=False).iloc[0]
best_model_name = best_model_info['Model']
best_model = best_estimators[best_model_name]

# Predictions and Residuals for the best model
y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

plt.figure(figsize=(14, 6))

# Graph 1: Actual vs Predicted 
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6, color='royalblue')
# Add a reference line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs. Predicted (Best Model: {best_model_name})')
plt.xlabel('Actual Sleep Hours')
plt.ylabel('Predicted Sleep Hours')
plt.grid(True, linestyle='--', alpha=0.7)

# Graph 2: Residual Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.6, color='darkorange')
# Add a horizontal line at y=0 to indicate perfect predictions
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.title(f'Residual Plot (Best Model: {best_model_name})')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Error)')
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('model_evaluation_plots.png', dpi=300) 
plt.show()

print(f"\nModel evaluation plots saved to 'model_evaluation_plots.png'.")