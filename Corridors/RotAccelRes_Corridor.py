import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Convert Excel to CSV
def excel_to_csv(excel_file, csv_file):
    # Read Excel into df
    df = pd.read_excel(excel_file)
    # Write df to CSV file
    df.to_csv(csv_file, index=False)

excel_file_path = 'Excels/RotAccel_Res.xlsx'
csv_file_path = 'csv_conversions/RotAccelRes_csv.csv'

# Convert Excel to CSV
excel_to_csv(excel_file_path, csv_file_path)

# Load signals from CSV into df
df = pd.read_csv(csv_file_path)

# Exclude the first column
exclude_columns = ['Time']  

# Extract signals from df
signals = [df[col].tolist() for col in df.columns if col not in exclude_columns]

# Names of columns and lists
signal_names = [col for col in df.columns if col not in exclude_columns]
upper_bound_signal = []
lower_bound_signal = []

# Iterate over each position in the signals
for i in range(len(signals[0])):  
    max_value_at_position = float('-inf')
    min_value_at_position = float('inf')
    # Iterate over each signal to find the maximum and minimum val
    for signal in signals:
        if signal[i] > max_value_at_position:
            max_value_at_position = signal[i]
        if signal[i] < min_value_at_position:
            min_value_at_position = signal[i]
    # Append the maxi value to upper_bound_signal
    upper_bound_signal.append(max_value_at_position)
    # Append the min value to the lower_bound_signal
    lower_bound_signal.append(min_value_at_position)

sigma = 1.5  # Standard deviation of Gaussian kernel 
smoothed_upper_signal = gaussian_filter(upper_bound_signal, sigma=sigma)
smoothed_lower_signal = gaussian_filter(lower_bound_signal, sigma=sigma)

# Save 
graph_data = {
    'Time': range(len(smoothed_upper_signal)),
    'Smoothed_Upper_Signal': smoothed_upper_signal,
    'Smoothed_Lower_Signal': smoothed_lower_signal
}

graph_df = pd.DataFrame(graph_data)
graph_df.to_csv('RA_Res_Corridor.csv', index=False)

'''
Plotting signals
plt.figure(figsize=(10, 6))

for idx, signal in enumerate(signals):
    plt.plot(signal, label=signal_names[idx], linestyle='--')
plt.plot(upper_bound_signal, label='Upper Bound Signal', color='blue', marker='o')
plt.plot(lower_bound_signal, label='Lower Bound Signal', color='red', marker='o')
'''

#Plotting corridors
plt.plot(smoothed_upper_signal, label=f'Smoothed Curve (Gaussian, sigma={sigma})', color='green', linewidth=2)
plt.plot(smoothed_lower_signal, label=f'Smoothed Curve (Gaussian, sigma={sigma})', color='red', linewidth=2)


plt.title('Rotational Acceleration (Resultant) Corridor')
plt.xlabel('Time (ms)')
plt.ylabel('Acceleration rad/s^2')
plt.legend()
plt.grid(True)
plt.show()