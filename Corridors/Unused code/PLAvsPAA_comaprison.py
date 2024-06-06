import pandas as pd
import matplotlib.pyplot as plt

# Convert Excel file to CSV
def excel_to_csv(excel_file, csv_file):
    df = pd.read_excel(excel_file)
    df.to_csv(csv_file, index=False)

excel_file_path = 'Excels/PLAvsPAA.xlsx'
csv_file_path = 'csv_conversions/PLAvsPAA_csv.csv'
excel_to_csv(excel_file_path, csv_file_path)

# Load the data from Excel file into a pandas DataFrame
df = pd.read_excel(excel_file_path, header=0)  # Use the first row as column names

# Extract column names from the DataFrame (assuming they are in the first row)
column_names = df.columns.tolist()
x_column_name = column_names[1] 
y_column_name = column_names[2]  

# Get the values for X and Y based on the column names
x_values = df[x_column_name]
y_values = df[y_column_name]

# Plotting the scatter graph
plt.figure(figsize=(8, 6))  
plt.scatter(x_values, y_values, color='blue', alpha=0.7)  
plt.title(f'Scatter Plot of {x_column_name} vs {y_column_name}')  
plt.xlabel(x_column_name)  
plt.ylabel(y_column_name)  
plt.grid(True)  

plt.show() 