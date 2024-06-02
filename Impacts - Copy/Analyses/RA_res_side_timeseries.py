import pandas as pd
import matplotlib.pyplot as plt

# Convert Excel file to CSV
def excel_to_csv(excel_file, csv_file):
    # Read Excel file into a DataFrame
    df = pd.read_excel(excel_file)
    # Write DataFrame to CSV file
    df.to_csv(csv_file, index=False)

excel_file_path = 'RA_Res_SideATD.xlsx'
csv_file_path = 'ATDcsv/RA_Res_SideATD_csv.csv'

# Convert Excel file to CSV
excel_to_csv(excel_file_path, csv_file_path)


# Convert Excel file to CSV
excel_to_csv(excel_file_path, csv_file_path)

# Load signals from the first CSV file into a DataFrame
df1 = pd.read_csv(csv_file_path)

# Exclude the first column
exclude_columns1 = ['Time']  

# Extract signals from df1
signals1 = [df1[col].tolist() for col in df1.columns if col not in exclude_columns1]

# Names of the columns from df1
signal_names1 = [col for col in df1.columns if col not in exclude_columns1]

# Use the first column from df1 as x-axis values
x_values1 = df1['Time'].tolist()

# Load signals from the second CSV file into a DataFrame
df2 = pd.read_csv('RA_Res_Front_Corridor.csv')  # Replace 'second_csv_file.csv' with the actual file path

# Exclude the first column
exclude_columns2 = ['Time']  

# Extract signals from df2
signals2 = [df2[col].tolist() for col in df2.columns if col not in exclude_columns2]

# Names of the columns from df2
signal_names2 = [col for col in df2.columns if col not in exclude_columns2]

# Use the first column from df2 as x-axis values
x_values2 = df2['Time'].tolist()


# Plotting signals
plt.figure(figsize=(10, 6))


# Plotting signals from df1 with x-axis values
for idx, signal in enumerate(signals1):
    if signal_names1[idx].startswith('Relaxed'):
        color = 'blue'
    elif signal_names1[idx].startswith('Clenched'):
        color = 'red'
    else:
        color = 'black'  # Default color for other cases
    plt.plot(x_values1, signal, label=signal_names1[idx], linestyle='--', color=color)

plt.title('Side RA (res) Time Series')
plt.xlabel('Time (ms)')
plt.ylabel('Rotational Acceleration rad/s2')
#plt.legend()
plt.grid(True)
plt.show()