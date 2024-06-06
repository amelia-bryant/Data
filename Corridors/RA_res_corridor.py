import pandas as pd
import matplotlib.pyplot as plt

# Load graph data from CSV
graph_df = pd.read_csv('RA_Res_Corridor.csv')

# Extract 
time = graph_df['Time'].to_numpy()
smoothed_upper_signal = graph_df['Smoothed_Upper_Signal'].to_numpy()
smoothed_lower_signal = graph_df['Smoothed_Lower_Signal'].to_numpy()

# Plotting signals
plt.figure(figsize=(10, 6))

# Plotting corridors
plt.plot(time, smoothed_upper_signal, label='Smoothed Upper Signal', color='green', linewidth=2)
plt.plot(time, smoothed_lower_signal, label='Smoothed Lower Signal', color='red', linewidth=2)

plt.title('Rotational Acceleration (Resultant) Corridor')
plt.xlabel('Time (ms)')
plt.ylabel('Linear Acceleration g')
plt.legend()
plt.grid(True)
plt.show()