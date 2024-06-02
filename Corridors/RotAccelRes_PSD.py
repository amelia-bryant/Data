import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert Excel file to CSV
def excel_to_csv(excel_file, csv_file):
    df = pd.read_excel(excel_file)
    df.to_csv(csv_file, index=False)

excel_file_path = 'Excels/RotAccel_Res.xlsx'
csv_file_path = 'csv_conversions/RotAccelResPSD_csv.csv'
excel_to_csv(excel_file_path, csv_file_path)
df = pd.read_csv(csv_file_path)
column_names = df.columns[1:].tolist()

# Compute and Plot Power Spectral Density (PSD) for each column
plt.figure(figsize=(10, 6)) 

for column in column_names:
    acceleration_data = df[column].values 

    # Compute FFT and PSD
    fft_result = np.fft.fft(acceleration_data)
    psd = np.abs(fft_result)**2

    # Frequency Axis Calculation, positive only
    n = len(acceleration_data)  
    dt = 0.001  
    freqs = np.fft.fftfreq(n, dt)  
    positive_freqs = freqs[:n//2]
    psd_positive = psd[:n//2]

    # Plot 
    plt.plot(positive_freqs[positive_freqs > 0], psd_positive[positive_freqs > 0], label=column)

# Show graph
plt.title('Power Spectral Density (PSD) of Lin Acceleration Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.legend() 
plt.show()