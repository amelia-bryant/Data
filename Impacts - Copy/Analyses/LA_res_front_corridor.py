import pandas as pd
import matplotlib.pyplot as plt

# Convert Excel to CSV
def excel_to_csv(excel_file, csv_file):
    # Read Excel file into a DataFrame
    df = pd.read_excel(excel_file)
    # Write df to CSV
    df.to_csv(csv_file, index=False)

excel_file_path = 'LA_Res_FrontATD.xlsx'
csv_file_path = 'ATDcsv/LA_Res_FrontATD_csv.csv'

# Convert Excel to CSV
excel_to_csv(excel_file_path, csv_file_path)

# Load signals from the first CSV into df
df1 = pd.read_csv(csv_file_path)

# Exclude first column
exclude_columns1 = ['Time']  

# Extract signals from df1
signals1 = [df1[col].tolist() for col in df1.columns if col not in exclude_columns1]

# Names of the columns from df1
signal_names1 = [col for col in df1.columns if col not in exclude_columns1]

# First column from df1 as x-axis values
x_values1 = df1['Time'].tolist()

# Load signals from the second CSV into df
df2 = pd.read_csv('LA_Res_Front_Corridor.csv')  

# Exclude the first column
exclude_columns2 = ['Time']  

# Extract signals from df2
signals2 = [df2[col].tolist() for col in df2.columns if col not in exclude_columns2]

# Names of the columns from df2
signal_names2 = [col for col in df2.columns if col not in exclude_columns2]

# First column from df2 as x-axis values
x_values2 = df2['Time'].tolist()


# Plotting signals
plt.figure(figsize=(10, 6))


# Plotting signals from df1 
for idx, signal in enumerate(signals1):
    if signal_names1[idx].startswith('Relaxed'):
        color = 'blue'
    elif signal_names1[idx].startswith('Clenched'):
        color = 'red'
    else:
        color = 'black'  
    plt.plot(x_values1, signal, label=signal_names1[idx], linestyle='--', color=color)

# Plotting only first signal from df2 
if signals2:
    plt.plot(x_values2, signals2[0], label=signal_names2[0], linestyle='-', color='green')

plt.title('Linear Acceleration (Resultant) Corridor')
plt.xlabel('Time (ms)')
plt.ylabel('Linear Acceleration g')
#plt.legend()
plt.grid(True)
plt.show()