import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for saving plots
output_dir = "eda_visualizations_summary"
os.makedirs(output_dir, exist_ok=True)

def save_plot(name):
    """Helper function to save the current figure before showing it"""
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}.png", dpi=300)
    print(f"Saved: {output_dir}/{name}.png")

# load the dataset
df = pd.read_csv('CoffeeAndHealth.csv', keep_default_na=False)

# data overview
print("data distribution:")
print(df.info())
print(df.describe())
# missing value analysis
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_stats = pd.concat([missing_count, missing_percentage], axis=1, keys=['Total', 'Percentage (%)'])
print("\n missing value statistics:")
print(missing_stats)
# dupicate value analysis
duplicate_count = df.duplicated().sum()
print(f"\n Total duplicate rows: {duplicate_count}")
if duplicate_count > 0:
    print("\n Duplicate rows:")
    print(df[df.duplicated()])

# setting seaborn style
sns.set_theme(style="whitegrid")

# outlier analysis using boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Age'], color='lightblue')
plt.title('Age Outlier Detection')
plt.xlabel('Age')
save_plot("1_age_outlier") # Save before show
plt.show()

# Histogram Plot: Caffeine_mg, Age and Sleep_Hours distribution
plt.figure(figsize=(8, 6)) # Added figure for clean saving
sns.histplot(df['Caffeine_mg'], kde=True, color='red')
plt.xlabel('Caffeine (mg)')
plt.ylabel('Count of Individuals')
plt.title('Caffeine (mg) Distribution')
save_plot("2_caffeine_distribution") # Save before show
plt.show()

mean_age = df['Age'].mean()
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, color='orange')
plt.axvline(df['Age'].mean(), color='black', linestyle='--', label='Mean Age')
plt.text(mean_age + 1, 1000, f'Mean Age: {mean_age:.2f}', color='black')
plt.title('Age Distribution with Mean Age')
plt.legend()
save_plot("3_age_distribution") # Save before show
plt.show()

mean_sleep_hours = df['Sleep_Hours'].mean()
plt.figure(figsize=(8, 6))
sns.histplot(df['Sleep_Hours'], kde=True, color='lightgreen')
plt.axvline(df['Sleep_Hours'].mean(), color='black', linestyle='--', label='Mean Sleep Hours')
plt.text(mean_sleep_hours + 0.5, 600, f'Mean Sleep Hours: {mean_sleep_hours:.2f}', color='black')
plt.title('Sleep Hours Distribution with Mean Sleep Hours')
plt.legend()
save_plot("4_sleep_hours_distribution") # Save before show
plt.show()


# Bar Plot: Gender distribution
plt.figure(figsize=(8, 6))
plt.bar(df['Gender'].value_counts().index, df['Gender'].value_counts().values, color=['aquamarine', 'pink', 'lemonchiffon'])
plt.xlabel('Gender')
plt.ylabel('Count of Individuals')
plt.title('Gender Distribution')
save_plot("5_gender_distribution") # Save before show
plt.show()

# Count Plot: Sleep Quality distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Sleep_Quality', data=df, palette='Set2')
plt.xlabel('Sleep Quality')
plt.ylabel('Count of Individuals')
plt.title('Sleep Quality Distribution')
save_plot("6_sleep_quality_distribution") # Save before show
plt.show()

# Scatter Plot: relationship between Caffeine_mg and Sleep_Hours
correlation = df['Caffeine_mg'].corr(df['Sleep_Hours'])
plt.figure(figsize=(10, 6))
sns.regplot(x='Caffeine_mg', y='Sleep_Hours', data=df, scatter_kws={'alpha': 0.2 }, line_kws={'color': 'salmon'})
plt.text(600, 10, f'Correlation (r) = {correlation:.3f}')
plt.xlabel('Caffeine (mg)')
plt.ylabel('Sleep Hours')
save_plot("7_caffeine_vs_sleep") # Save before show
plt.show()

# Box plot: relationship between Sleep_Hours and Sleep_Quality
order = ['Excellent', 'Good', 'Fair', 'Poor']
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sleep_Quality', y='Sleep_Hours', data=df, order=order, palette='viridis')
plt.title('Relationship between Sleep Hours and Quality')
plt.xlabel('Sleep Quality')
plt.ylabel('Sleep Hours')
save_plot("8_sleep_quality_vs_hours") # Save before show
plt.show()

# Scatter Plot: relationship between Caffeine_mg and Coffee_Intake
correlation = df['Caffeine_mg'].corr(df['Coffee_Intake'])
plt.figure(figsize=(10, 6))
sns.regplot(x='Caffeine_mg', y='Coffee_Intake', data=df, scatter_kws={'alpha': 0.2 }, line_kws={'color': 'salmon'})
plt.text(600, 8, f'Correlation (r) = {correlation:.3f}')
plt.xlabel('Caffeine (mg)')
plt.ylabel('Coffee Intake')
save_plot("9_caffeine_vs_intake") # Save before show
plt.show()

# Data Preparation: Map categorical variables to numerical values for correlation calculation
# Note: This step is for visualization only and does not affect the subsequent modeling pipeline
df_eda = df.copy()

# Map ordinal variables
stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
health_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
quality_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

df_eda['Stress_Level'] = df_eda['Stress_Level'].map(stress_mapping)
df_eda['Health_Issues'] = df_eda['Health_Issues'].map(health_mapping)
df_eda['Sleep_Quality'] = df_eda['Sleep_Quality'].map(quality_mapping)

# Select numerical columns for the heatmap
numeric_cols = df_eda.select_dtypes(include=['number']).columns
if 'ID' in numeric_cols: numeric_cols = numeric_cols.drop('ID') # Exclude ID

# --- 1. Plot Heatmap ---
plt.figure(figsize=(12, 10))

# Calculate correlation matrix
corr_matrix = df_eda[numeric_cols].corr()

# Plot heatmap: annot=True shows values, fmt=".2f" formats to 2 decimal places
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix Heatmap')
save_plot("10_correlation_heatmap") # Call the previously defined save function
plt.show()

# --- 2. Plot Pair Plot ---
# Select key variables to avoid an overcrowded plot
key_vars = ['Caffeine_mg', 'Sleep_Hours', 'Age', 'Stress_Level', 'Sleep_Quality']

# hue='Gender' colors data by gender, ideal for observing patterns across groups
# diag_kind='kde' shows the density distribution on the diagonal
pair_plot = sns.pairplot(df_eda[key_vars + ['Gender']], hue='Gender', palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6})

pair_plot.fig.suptitle('Pair Plot of Key Variables', y=1.02)
pair_plot.savefig(f"{output_dir}/11_pair_plot.png") # Save the PairGrid object
plt.show()