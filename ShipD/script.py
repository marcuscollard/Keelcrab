import pandas as pd
import sys
import os

# Load the CSV file into a DataFrame

dataset_num = 4
datasets = ['Constrained_Randomized_Set_1', 'Constrained_Randomized_Set_2', 'Constrained_Randomized_Set_3', 
            "Diffusion_Aug_Set_1", "Diffusion_Aug_Set_2"]

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(
    script_dir,
    'Ship_D_Dataset',
    datasets[dataset_num-1],
    'Input_Vectors.csv'
)
df = pd.read_csv(csv_path)

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.std())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for constant values
print("\nConstant Value Columns:")
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
print(constant_columns)

